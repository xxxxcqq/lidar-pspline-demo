import json, sys
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.append("scripts")
from utils_pspline import (
    quadratic_spline_basis, penalty_matrix, knots_quantile,
    ig_sample,
)

# ===================== 数据与基 =====================
df = pd.read_csv("data/lidar.csv")
x = df["range"].to_numpy()
y = df["logratio"].to_numpy()
n = len(x)

# 对 x 做中心化 + 缩放（仅改善数值条件，不改变拟合空间）
SHIFT, SCALE = 555.0, 100.0
xs = (x - SHIFT) / SCALE

K = 30
knots = knots_quantile(xs, K)
C = quadratic_spline_basis(xs, knots)     # n × 33
D = penalty_matrix(2, K)                    # 33 × 33，前 3 项不惩罚
CtC = C.T @ C
Cty = C.T @ y

x_grid = np.linspace(x.min(), x.max(), 400)
xs_grid = (x_grid - SHIFT) / SCALE
C_grid = quadratic_spline_basis(xs_grid, knots)

# ===================== GCV 起始值 =====================
lam_grid = np.logspace(-2, 4, 80)
best = {"gcv": np.inf}
for lam in lam_grid:
    A = CtC + (lam ** 4) * D
    try:
        Ainv = np.linalg.inv(A)
        beta = Ainv @ Cty
        yhat = C @ beta
        rss = float(np.sum((y - yhat) ** 2))
        df_fit = float(np.trace(C @ Ainv @ C.T))
        gcv = rss / (1 - df_fit / n) ** 2
        if gcv < best["gcv"]:
            best = {"gcv": gcv, "lam": lam, "beta": beta,
                    "rss": rss, "df_fit": df_fit}
    except np.linalg.LinAlgError:
        continue

sigma2_eps_init = best["rss"] / (n - best["df_fit"])
rho_init        = best["lam"] ** 4       # σ²ε/σ²u = λ^4（p=2）
sigma2_u_init   = sigma2_eps_init / rho_init
nu_init         = best["beta"].copy()
print(f"GCV: lam={best['lam']:.3f}, df={best['df_fit']:.2f}, "
      f"sigma2_eps={sigma2_eps_init:.6f}, sigma2_u={sigma2_u_init:.6f}")

# ===================== Gibbs 采样 =====================
# σ²u 以 GCV 的方差成分为尺度中心，先验 IG(2,2)（均值 1，方差有限）
N_iter = 3000
A_eps, B_eps = 0.1, 0.1
A_u, B_u = 2.0, 2.0
rng = np.random.default_rng(20240601)

p_dim = C.shape[1]
nu_all   = np.zeros((N_iter, p_dim))
s2e_all  = np.zeros(N_iter)
tau_all  = np.zeros(N_iter)      # τ = σ²u / σ²u(GCV)

nu_cur, s2e_cur, tau_cur = nu_init, sigma2_eps_init, 1.0
nu_all[0] = nu_cur
s2e_all[0] = s2e_cur
tau_all[0] = tau_cur

for it in range(1, N_iter):
    # (1) (β, u) | σ²ε, σ²u ~ N(mean, cov)
    rho = s2e_cur / (tau_cur * sigma2_u_init)
    A = CtC + rho * D
    Ainv = np.linalg.inv(A)
    mean_nu = Ainv @ Cty
    cov_nu = s2e_cur * Ainv
    L = np.linalg.cholesky(cov_nu + 1e-12 * np.eye(p_dim))
    nu_cur = mean_nu + L @ rng.standard_normal(p_dim)

    # (2) τ | u ~ IG(Au + K/2, Bu + ||u||²/(2σ²u(GCV)))
    u_part = nu_cur[3:]
    tau_cur = ig_sample(A_u + K / 2,
                        B_u + 0.5 * float(u_part @ u_part) / sigma2_u_init, rng)

    # (3) σ²ε | y, ν ~ IG(Aε + n/2, Bε + ||y - Cν||²/2)
    resid = y - C @ nu_cur
    s2e_cur = ig_sample(A_eps + n / 2, B_eps + 0.5 * float(resid @ resid), rng)

    nu_all[it] = nu_cur
    s2e_all[it] = s2e_cur
    tau_all[it] = tau_cur

# 丢弃 burn-in
burn = 500
nu_post  = nu_all[burn:]
s2e_post = s2e_all[burn:]
tau_post = tau_all[burn:]

# ===================== 后验 f(x) 摘要 =====================
f_grid_post = C_grid @ nu_post.T          # 400 × 2500
f_mean = f_grid_post.mean(axis=1)
f_sd   = f_grid_post.std(axis=1, ddof=1)
f_lo   = np.percentile(f_grid_post, 2.5,  axis=1)
f_hi   = np.percentile(f_grid_post, 97.5, axis=1)

# 5 条后验实现
idx_real = rng.choice(nu_post.shape[0], 5, replace=False)
realizations = [(C_grid @ nu_post[i]).tolist() for i in idx_real]

# ===================== 近似贝叶斯 =====================
rho_approx = sigma2_eps_init / sigma2_u_init
A_approx = CtC + rho_approx * D
Ainv_approx = np.linalg.inv(A_approx)
var_approx = sigma2_eps_init * np.einsum(
    "ij,jk,ik->i", C_grid, Ainv_approx, C_grid
)
sd_approx = np.sqrt(np.clip(var_approx, 1e-20, None))
rel_err = (f_sd ** 2) / var_approx - 1.0

# ===================== 后验密度（σ_ε 与标准化 σ_u） =====================
s_eps_post = np.sqrt(s2e_post)
s_u_post   = np.sqrt(tau_post)          # τ 的平方根，相对 GCV 方差成分
kde_e = gaussian_kde(s_eps_post)
kde_u = gaussian_kde(s_u_post)
grid_e = np.linspace(s_eps_post.min(), s_eps_post.max(), 200)
grid_u = np.linspace(s_u_post.min(), s_u_post.max(), 200)

# ===================== Fig 16.1 =====================
json.dump({
    "fig_id": "16.1", "chapter": 16,
    "title": "LIDAR 贝叶斯分析六面板",
    "x_grid": x_grid.tolist(),
    "scatter": {"x": x.tolist(), "y": y.tolist()},
    "panel_a": {"realizations": realizations},
    "panel_b": {"mean": f_mean.tolist(),
                "lo": f_lo.tolist(), "hi": f_hi.tolist()},
    "panel_c": {"sd": f_sd.tolist()},
    "panel_d": {"rel_err": rel_err.tolist(),
                "sd_full": f_sd.tolist(),
                "sd_approx": sd_approx.tolist()},
    "panel_e": {"x": grid_e.tolist(), "density": kde_e(grid_e).tolist()},
    "panel_f": {"x": grid_u.tolist(), "density": kde_u(grid_u).tolist()},
    "xlim": [390, 720], "ylim": [-1.0, 0.05],
    "notes": ("Gibbs 3000 次迭代，前 500 为 burn-in；"
              "30 结点二次样条；σ²ε 先验 IG(0.1,0.1)，σ²u 先验 IG(2,2)。"
              "近似贝叶斯用 GCV 选出的 σ²ε、σ²u 代入 (16.8)。"
              "密度用高斯核平滑替代书中 200-bin 直方图 + GLMM 平滑；"
              "(e) 画 σ_ε 的后验密度；(f) 画 σ_u 相对 GCV 方差成分的比值尺度。"),
}, open("assets/fig_16_1.json", "w"), indent=2)

# ===================== Fig 16.2 =====================
C_620 = quadratic_spline_basis(np.array([(620.0 - SHIFT) / SCALE]), knots)
f620_chain = (nu_all @ C_620[0]).tolist()

json.dump({
    "fig_id": "16.2", "chapter": 16,
    "title": "Gibbs 链诊断：range=620 处估计均值",
    "iterations": list(range(N_iter)),
    "chain": f620_chain,
    "overall_mean": float(np.mean(f620_chain)),
    "notes": ("左：每 30 次迭代取一点；"
              "右：1501–1600 连续 100 次迭代；"
              "水平线为 3000 次样本均值。"
              "书中 lag1–4 自相关约 −0.013、−0.008、0.024、0.018。"),
}, open("assets/fig_16_2.json", "w"), indent=2)

print("done")
print("posterior mean sigma_eps =", round(float(s_eps_post.mean()), 4),
      "| sigma_u =", round(float(s_u_post.mean()), 4))
print("lag1 =", round(float(np.corrcoef(np.array(f620_chain[:-1]),
                                       np.array(f620_chain[1:]))[0, 1]), 4))
