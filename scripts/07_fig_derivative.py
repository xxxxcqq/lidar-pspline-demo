import json, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.append("scripts")
from utils_pspline import (
    cubic_spline_basis, cubic_deriv_basis,
    fit_cubic_penalty, fit_cubic_local, knots_quantile,
)

df = pd.read_csv("data/lidar.csv")
x = df["range"].to_numpy()
y = df["logratio"].to_numpy()
n = len(x)

# 15-knot 三次样条（按第 6.8.3 节）
K = 15
knots = knots_quantile(x, K)
K_len = len(knots)

# 100 个点求导数 + 带（按第 6.14 节 M=100）
x_deriv = np.linspace(x.min(), x.max(), 100)
C_deriv = cubic_deriv_basis(x_deriv, knots)

def save(name, obj):
    json.dump(obj, open(f"assets/{name}.json", "w"), indent=2)

def gcv_global(log_lam):
    return fit_cubic_penalty(x, y, knots, np.exp(log_lam))["gcv"]

res = minimize_scalar(
    gcv_global,
    bounds=(np.log(0.1), np.log(500)),
    method="bounded",
)
lam_global = float(np.exp(res.x))
fit_g = fit_cubic_penalty(x, y, knots, lam_global)

# ============ Fig 6.13：导数估计 ============
def deriv_with_band(fit, x_deriv, C_deriv, sigma2_scale=1.0):
    beta = fit["beta"]
    d = C_deriv @ beta
    cov = fit["sigma2"] * fit["Ainv"]  # 偏差修正版
    var_per_point = np.einsum("ij,jk,ik->i", C_deriv, cov, C_deriv)
    se = np.sqrt(np.clip(var_per_point, 1e-20, None)) * np.sqrt(sigma2_scale)
    return -d, -d + 2 * se, -d - 2 * se  # 注意：书里画的是 -f'

d_g, hi_g, lo_g = deriv_with_band(fit_g, x_deriv, C_deriv)

# --- 空间自适应局部惩罚 ---
M = 5
subknots = np.quantile(knots, np.linspace(0, 1, M))
log_lams = np.full(M, np.log(lam_global))
lam_lo, lam_hi = np.log(lam_global) - 3.0, np.log(lam_global) + 3.0

def interp_lams(log_lams):
    return np.exp(np.interp(knots, subknots, log_lams))

def gcv_local(log_lams):
    lams = interp_lams(log_lams)
    return fit_cubic_local(x, y, knots, lams)["gcv"]

for iteration in range(2):
    for m in range(M):
        def obj(ll):
            lams_try = log_lams.copy()
            lams_try[m] = ll
            return gcv_local(lams_try)
        res_m = minimize_scalar(obj, bounds=(lam_lo, lam_hi), method="bounded")
        log_lams[m] = res_m.x

lams_local = interp_lams(log_lams)
fit_l = fit_cubic_local(x, y, knots, lams_local)

# 局部惩罚 SE 的“保守”校正：把局部平均方差拉到与全局一致
def se_global(xe):
    Cg = cubic_deriv_basis(xe, knots)
    v = np.einsum("ij,jk,ik->i", Cg, fit_g["sigma2"] * fit_g["Ainv"], Cg)
    return np.sqrt(np.clip(v, 1e-20, None))

def se_local(xe):
    Cg = cubic_deriv_basis(xe, knots)
    v = np.einsum("ij,jk,ik->i", Cg, fit_l["sigma2"] * fit_l["Ainv"], Cg)
    return np.sqrt(np.clip(v, 1e-20, None))

se_g_grid = se_global(x_deriv)
se_l_grid = se_local(x_deriv)
correction = (np.mean(se_g_grid**2) / np.mean(se_l_grid**2))
se_l_adj = se_l_grid * np.sqrt(correction)
d_l = -(C_deriv @ fit_l["beta"])

save("fig_6_13", {
    "fig_id": "6.13", "chapter": 6,
    "title": "LIDAR 数据：−f'(range) 估计与置信带",
    "panels": [
        {
            "label": "Global penalty (GCV)",
            "x": x_deriv.tolist(),
            "y": d_g.tolist(),
            "y_hi": hi_g.tolist(),
            "y_lo": lo_g.tolist(),
            "lambda": lam_global,
        },
        {
            "label": "Spatially adaptive penalty",
            "x": x_deriv.tolist(),
            "y": d_l.tolist(),
            "y_hi": (d_l + 2 * se_l_adj).tolist(),
            "y_lo": (d_l - 2 * se_l_adj).tolist(),
            "log_lams_subknots": log_lams.tolist(),
            "subknots": subknots.tolist(),
        },
    ],
    "ylim": [-0.005, 0.015],
    "notes": ("三次样条 K=15；上图为全局 λ 由 GCV 选定；"
              "下图用 M=5 次结点局部惩罚，2 轮 1D GCV 网格搜索，"
              "SE 做保守校正（把平均方差拉齐到全局）。"
              "阴影为逐点 ±2 SE。"),
})

# ============ Fig 6.14：逐点 vs 同时置信带 ============
# 用同一全局惩罚拟合；M=100 网格；N=10000 模拟 m_0.95
rng = np.random.default_rng(20240601)
N = 10000
d_param = C_deriv.shape[1]
cov_bias = fit_g["sigma2"] * fit_g["Ainv"]
eigvals, eigvecs = np.linalg.eigh(cov_bias)
eigvals = np.clip(eigvals, 0, None)
L = eigvecs * np.sqrt(eigvals)

z = rng.standard_normal((N, d_param))  # N x d
delta = z @ L.T                         # N x d
deriv_err = delta @ C_deriv.T           # N x M
se_for_m = se_g_grid                    # 逐点 SE
normalized = np.abs(deriv_err) / se_for_m[None, :]
max_abs = normalized.max(axis=1)
m_095 = float(np.percentile(max_abs, 95))

save("fig_6_14", {
    "fig_id": "6.14", "chapter": 6,
    "title": "LIDAR 数据：−f'(range) 的逐点与同时置信限",
    "x": x_deriv.tolist(),
    "y": d_g.tolist(),
    "point_hi": (d_g + 2 * se_g_grid).tolist(),
    "point_lo": (d_g - 2 * se_g_grid).tolist(),
    "sim_hi":   (d_g + m_095 * se_g_grid).tolist(),
    "sim_lo":   (d_g - m_095 * se_g_grid).tolist(),
    "m_095":    m_095,
    "lambda":   lam_global,
    "notes": (f"三次样条 K=15；M=100 网格，N=10000 模拟，"
              f"得 m_0.95 ≈ {m_095:.3f}（书中约 3.17）。"
              f"深色为逐点 ±2 SE，浅色为同时 ±m_0.95 SE。"),
})

print(f"lam_global = {lam_global:.3f}, m_0.95 = {m_095:.3f}")
print("done")
