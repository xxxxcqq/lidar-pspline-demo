import json, sys
import numpy as np
import pandas as pd

sys.path.append("scripts")
from utils_pspline import (
    linear_spline_basis, penalty_matrix, fit_pspline,
    fit_weighted, knots_quantile, df_fit
)

df = pd.read_csv("data/lidar.csv")
x = df["range"].to_numpy()
y = df["logratio"].to_numpy()
n = len(x)

K = 24
knots = knots_quantile(x, K)
C = linear_spline_basis(x, knots)
D = penalty_matrix(1, K)
x_grid = np.linspace(x.min(), x.max(), 400)
C_grid = linear_spline_basis(x_grid, knots)

lam_mean = 30.0
lam_var  = 30.0
eps_var  = 1e-8
# log(r^2) 的期望比 log(sigma^2) 低 E[log chi2_1] = -1.2704，
# 所以 exp(平滑值) 是几何均值 0.281*sigma^2；乘 kappa 还原为方差均值
kappa = float(np.exp(1.2703628454614782))

def save(name, obj):
    json.dump(obj, open(f"assets/{name}.json", "w"), indent=2)

# ========== 同方差模型 ==========
beta0, yhat0, Ainv0 = fit_weighted(C, y, D, lam_mean, np.ones(n))
df0 = float(np.trace(C @ Ainv0 @ C.T))
sigma2_0 = float(np.sum((y - yhat0) ** 2) / (n - df0))
f_grid_0 = C_grid @ beta0

# ========== 异方差模型：迭代 3 轮 ==========
beta, yhat, Ainv = beta0, yhat0, Ainv0
beta_g = np.zeros(C.shape[1])
w_scale = 1.0
for it in range(3):
    r2 = (y - yhat) ** 2
    log_r2 = np.log(r2 + eps_var)
    beta_g, g_hat, S_g = fit_pspline(C, log_r2, D, lam_var)
    var_hat_data = np.maximum(kappa * np.exp(g_hat), 1e-8)
    w_raw = 1.0 / var_hat_data
    # 权重归一到均值 1，保证 lam_mean 的含义与同方差模型可比
    w_scale = float(w_raw.mean())
    w = w_raw / w_scale
    beta, yhat, Ainv = fit_weighted(C, y, D, lam_mean, w)

f_grid_h = C_grid @ beta
var_fn   = np.maximum(kappa * np.exp(C_grid @ beta_g), 1e-8)   # 方差函数

# ========== Fig 14.1：散点 + 平滑 ==========
save("fig_14_1", {
    "fig_id": "14.1", "chapter": 14,
    "title": "LIDAR 散点 + 平滑，展示异方差",
    "scatter": {"x": x.tolist(), "y": y.tolist()},
    "fit": {"x": x_grid.tolist(), "y": f_grid_0.tolist()},
    "xlim": [390, 720], "ylim": [-1.0, 0.05],
    "notes": "变异性随 range 增大"
})

# ========== Fig 14.2：常数方差假设的带 ==========
se2_0 = np.einsum("ij,jk,ik->i", C_grid, Ainv0, C_grid) * sigma2_0
band0 = 2 * np.sqrt(se2_0 + sigma2_0)
save("fig_14_2", {
    "fig_id": "14.2", "chapter": 14,
    "title": "常数方差假设下的拟合与变异性带",
    "scatter": {"x": x.tolist(), "y": y.tolist()},
    "x": x_grid.tolist(),
    "fit": f_grid_0.tolist(),
    "band_lo": (f_grid_0 - band0).tolist(),
    "band_hi": (f_grid_0 + band0).tolist(),
    "xlim": [390, 720], "ylim": [-1.2, 0.3],
    "notes": "带在全范围近似等宽；x≈400 处过宽，x≈700 处过窄"
})

# ========== Fig 14.3：异方差模型下的带 ==========
# 加权拟合的拟合值方差用三明治公式，再除以权重缩放因子回到原始方差尺度
M = Ainv @ (C.T * w) @ C @ Ainv / w_scale
se2_h = np.einsum("ij,jk,ik->i", C_grid, M, C_grid)
bandh = 2 * np.sqrt(se2_h + var_fn)
save("fig_14_3", {
    "fig_id": "14.3", "chapter": 14,
    "title": "方差函数建模后的拟合与变异性带",
    "scatter": {"x": x.tolist(), "y": y.tolist()},
    "x": x_grid.tolist(),
    "fit": f_grid_h.tolist(),
    "band_lo": (f_grid_h - bandh).tolist(),
    "band_hi": (f_grid_h + bandh).tolist(),
    "xlim": [390, 720], "ylim": [-1.2, 0.3],
    "notes": ("带随 range 增大而变宽，与数据一致。"
              "异方差模型用 log-spline 近似替代书中 Gamma GLMM；"
              "方差函数的形式和迭代思路与 14.2 节一致，"
              "曲线和带宽形态与书中结论吻合，数值上可能有小差异。")
})

# ========== Fig 14.4：同方差标准化绝对残差 ==========
z0 = np.abs(y - yhat0) / np.sqrt(sigma2_0)
beta_z0, smooth_z0, _ = fit_pspline(C, z0, D, lam_mean)
save("fig_14_4", {
    "fig_id": "14.4", "chapter": 14,
    "title": "同方差标准化绝对残差 + 平滑",
    "scatter": {"x": x.tolist(), "y": z0.tolist()},
    "smooth": {"x": x_grid.tolist(), "y": (C_grid @ beta_z0).tolist()},
    "xlim": [390, 720], "ylim": [0, 3.2],
    "notes": "平滑线有明显上升趋势 → 同方差不足"
})

# ========== Fig 14.5：异方差标准化绝对残差 ==========
var_hat_data = np.maximum(kappa * np.exp(C @ beta_g), 1e-8)
zh = np.abs(y - yhat) / np.sqrt(var_hat_data)
beta_zh, _, _ = fit_pspline(C, zh, D, lam_mean)
save("fig_14_5", {
    "fig_id": "14.5", "chapter": 14,
    "title": "异方差标准化绝对残差 + 平滑",
    "scatter": {"x": x.tolist(), "y": zh.tolist()},
    "smooth": {"x": x_grid.tolist(), "y": (C_grid @ beta_zh).tolist()},
    "xlim": [390, 720], "ylim": [0, 3.2],
    "notes": ("平滑线在全范围基本水平（约 0.6–0.8，|Z| 的期望为 0.8），"
              "无明显上升趋势 → 方差函数已充分解释")
})

print("sigma2_0 =", round(sigma2_0, 6))
print("df weighted =", round(float(np.trace(C @ Ainv @ C.T @ np.diag(w))), 3))
print("band 14.2 (full) =", round(float(2 * band0[0]), 3), "→", round(float(2 * band0[-1]), 3))
print("band 14.3 (full) =", round(float(2 * bandh[0]), 3), "→", round(float(2 * bandh[-1]), 3))
print("14.4 smooth ends =", round(float((C_grid @ beta_z0)[0]), 3), "→",
      round(float((C_grid @ beta_z0)[-1]), 3))
print("14.5 smooth ends =", round(float((C_grid @ beta_zh)[0]), 3), "→",
      round(float((C_grid @ beta_zh)[-1]), 3))
print("done")
