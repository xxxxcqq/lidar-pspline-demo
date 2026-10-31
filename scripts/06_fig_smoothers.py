import json, sys
import numpy as np
import pandas as pd

sys.path.append("scripts")
from utils_pspline import (
    linear_spline_basis, quadratic_spline_basis, penalty_matrix,
    fit_pspline, df_fit, knots_quantile,
    bisect_log_df, safe_inv
)

# ===================== 数据 =====================
df = pd.read_csv("data/lidar.csv")
x = df["range"].to_numpy()
y = df["logratio"].to_numpy()
n = len(x)
x_grid = np.linspace(x.min(), x.max(), 400)
xlim, ylim = [390, 720], [-1.0, 0.05]
scatter = {"x": x.tolist(), "y": y.tolist()}

def save(name, obj):
    json.dump(obj, open(f"assets/{name}.json", "w"), indent=2)

# ===================== 平滑器 1：24 结点惩罚样条 =====================
K = 24
knots = knots_quantile(x, K)
C1 = linear_spline_basis(x, knots)
C1g = linear_spline_basis(x_grid, knots)
D1 = penalty_matrix(1, K)

def smoother_pspline(target_df):
    def df_fn(lam):
        _, _, S = fit_pspline(C1, y, D1, lam, p=1)
        return df_fit(S)
    lam = bisect_log_df(df_fn, target_df, 1e-2, 1e4)
    beta, _, S = fit_pspline(C1, y, D1, lam, p=1)
    return {
        "name": f"24-knot penalized spline (λ≈{lam:.1f})",
        "fit_x": x_grid.tolist(),
        "fit_y": (C1g @ beta).tolist(),
        "df": float(df_fit(S)),
    }

# ===================== 平滑器 2：离散二阶差分 smoothing spline =====================
D2 = np.diff(np.eye(n), n=2, axis=0)
K2 = D2.T @ D2

def smoother_smoothing_spline(target_df):
    def df_fn(lam):
        S = safe_inv(np.eye(n) + lam * K2)
        return float(np.trace(S))
    lam = bisect_log_df(df_fn, target_df, 1e-4, 1e6)
    S = safe_inv(np.eye(n) + lam * K2)
    y_hat = S @ y
    # 排序后插值到 x_grid
    order = np.argsort(x)
    y_grid = np.interp(x_grid, x[order], y_hat[order])
    return {
        "name": f"Smoothing spline approx (λ≈{lam:.3g})",
        "fit_x": x_grid.tolist(),
        "fit_y": y_grid.tolist(),
        "df": float(np.trace(S)),
    }

# ===================== 平滑器 3：local linear（高斯核） =====================
def local_linear_eval(x_eval, x_data, y_data, bw, want_S=True):
    n_eval = len(x_eval)
    y_hat = np.zeros(n_eval)
    if want_S:
        S = np.zeros((n_eval, len(x_data)))
    for i, xe in enumerate(x_eval):
        u = (x_data - xe) / bw
        w = np.exp(-0.5 * u ** 2)
        Xloc = np.column_stack([np.ones_like(x_data), x_data - xe])
        XtW = Xloc.T * w
        A = XtW @ Xloc
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            Ainv = np.linalg.pinv(A)
        l = Ainv[0] @ XtW
        y_hat[i] = l @ y_data
        if want_S:
            S[i] = l
    return (y_hat, S) if want_S else y_hat

def smoother_local_linear(target_df):
    def df_fn(bw):
        _, S = local_linear_eval(x, x, y, bw)
        return float(np.trace(S))
    bw = bisect_log_df(df_fn, target_df, 1.0, 300.0, tol=0.1, maxiter=40)
    y_hat, S = local_linear_eval(x, x, y, bw)
    y_grid = local_linear_eval(x_grid, x, y, bw, want_S=False)
    return {
        "name": f"Local linear (bw≈{bw:.1f})",
        "fit_x": x_grid.tolist(),
        "fit_y": y_grid.tolist(),
        "df": float(np.trace(S)),
    }

# ===================== 平滑器 4：Fourier series =====================
def smoother_fourier(target_df):
    J = int(round((target_df - 1) / 2))
    t = (x - x.min()) / (x.max() - x.min())
    tg = (x_grid - x.min()) / (x.max() - x.min())
    cols = [np.ones_like(x)]
    cols_g = [np.ones_like(x_grid)]
    for j in range(1, J + 1):
        cols += [np.cos(j * np.pi * t), np.sin(j * np.pi * t)]
        cols_g += [np.cos(j * np.pi * tg), np.sin(j * np.pi * tg)]
    B = np.column_stack(cols)
    Bg = np.column_stack(cols_g)
    S = B @ np.linalg.pinv(B.T @ B) @ B.T
    beta = np.linalg.lstsq(B, y, rcond=None)[0]
    return {
        "name": f"Fourier series (J={J}, df={1+2*J})",
        "fit_x": x_grid.tolist(),
        "fit_y": (Bg @ beta).tolist(),
        "df": float(np.trace(S)),
    }

# ===================== Fig 3.25 =====================
target_325 = 11
curves_325 = [
    smoother_pspline(target_325),
    smoother_smoothing_spline(target_325),
    smoother_local_linear(target_325),
    smoother_fourier(target_325),
]
save("fig_3_25", {
    "layout": "overlay", "fig_id": "3.25", "chapter": 3,
    "title": "四种平滑器各 11 df 的对比",
    "scatter": scatter, "curves": curves_325,
    "xlim": xlim, "ylim": ylim,
    "notes": ("smoothing spline 用离散二阶差分惩罚近似；"
              "local linear 用高斯核；Fourier 用 J=5 谐波。"
              "四种平滑器固定 df 时视觉几乎重合。")
})

# ===================== Fig 13.13 公共 =====================
target_1313 = 12

# --- 两种惩罚样条 ---
def smoother_pspline_general(target_df, degree):
    kn = knots_quantile(x, 24)
    if degree == 1:
        C = linear_spline_basis(x, kn)
        Cg = linear_spline_basis(x_grid, kn)
        p = 1
    else:
        C = quadratic_spline_basis(x, kn)
        Cg = quadratic_spline_basis(x_grid, kn)
        p = 2
    D = penalty_matrix(degree, 24)
    def df_fn(lam):
        _, _, S = fit_pspline(C, y, D, lam, p=p)
        return df_fit(S)
    lam = bisect_log_df(df_fn, target_df, 1e-2, 1e4, tol=0.05)
    beta, _, S = fit_pspline(C, y, D, lam, p=p)
    return {
        "name": f"{'Linear' if degree==1 else 'Quadratic'} penalized spline",
        "fit_x": x_grid.tolist(),
        "fit_y": (Cg @ beta).tolist(),
        "df": float(df_fit(S)),
    }

# --- 通用 kriging smoother（BLP 形式，含线性趋势项） ---
def kriging_smoother(C_data, C_eval, target_df, name, jitter=1e-6):
    n = C_data.shape[0]
    P = np.column_stack([np.ones_like(x), x - x.mean()])
    Pe = np.column_stack([np.ones_like(x_grid), x_grid - x.mean()])

    def design(sigma2):
        Vi = safe_inv(C_data + (sigma2 + jitter) * np.eye(n))
        XtVi = P.T @ Vi
        S = (C_data @ Vi
             + (P - C_data @ Vi @ P) @ safe_inv(XtVi @ P) @ XtVi)
        return S, Vi

    def df_fn(sigma2):
        S, _ = design(sigma2)
        return float(np.trace(S))

    sigma2 = bisect_log_df(df_fn, target_df, 1e-6, 1e8, tol=0.1, maxiter=60)
    S, Vi = design(sigma2)
    beta = np.linalg.solve(P.T @ Vi @ P, P.T @ Vi @ y)
    y_grid = C_eval @ Vi @ (y - P @ beta) + Pe @ beta
    return {
        "name": f"{name} (σ²≈{sigma2:.3g})",
        "fit_x": x_grid.tolist(),
        "fit_y": y_grid.tolist(),
        "df": float(np.trace(S)),
    }

# --- 四种协方差矩阵 ---
r_data = np.abs(x[:, None] - x[None, :])
r_eval = np.abs(x_grid[:, None] - x[None, :])
rho = 50.0

C_thin = r_data ** 3
C_thin_e = r_eval ** 3

C_gen = -r_data + np.abs(x)[:, None] + np.abs(x)[None, :]
C_gen_e = -r_eval + np.abs(x_grid)[:, None] + np.abs(x)[None, :]

C_exp = np.exp(-r_data / rho)
C_exp_e = np.exp(-r_eval / rho)

C_gauss = np.exp(-0.5 * (r_data / rho) ** 2)
C_gauss_e = np.exp(-0.5 * (r_eval / rho) ** 2)

curves_1313 = [
    smoother_pspline_general(target_1313, degree=1),
    smoother_pspline_general(target_1313, degree=2),
    kriging_smoother(C_thin, C_thin_e, target_1313, "Thin plate (r³)"),
    kriging_smoother(C_gen, C_gen_e, target_1313, "Generalized cov (−|r|+…)"),
    kriging_smoother(C_exp, C_exp_e, target_1313, "Exponential cov (ρ=50)"),
    kriging_smoother(C_gauss, C_gauss_e, target_1313, "Gaussian cov (ρ=50)"),
]

save("fig_13_13", {
    "layout": "overlay", "fig_id": "13.13", "chapter": 13,
    "title": "四种径向平滑 + 线性/二次惩罚样条，各 12 df",
    "scatter": scatter, "curves": curves_1313,
    "xlim": xlim, "ylim": ylim,
    "notes": ("四种径向平滑器用 kriging BLP 形式实现：S = C V⁻¹ + (P - C V⁻¹P)(PᵀV⁻¹P)⁻¹PᵀV⁻¹，"
              "P 为线性趋势项（通用 kriging），避免纯 BLP 在右边界的外推偏差；"
              "V = C + σ²I，二分搜 σ² 使 tr(S)=12。ρ 固定为 50（range 跨度的约 15%）。"
              "具体协方差形式按第 13.4 节推定，页面标注为近似复现。")
})

print("done")
