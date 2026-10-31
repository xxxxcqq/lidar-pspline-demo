import json, sys
import numpy as np
import pandas as pd

sys.path.append("scripts")
from utils_pspline import (
    linear_spline_basis, quadratic_spline_basis, penalty_matrix,
    fit_pspline, df_fit, df_res, knots_quantile,
    cv_score, gcv_score, rss_score, reml_ml_lambda
)

df = pd.read_csv("data/lidar.csv")
x = df["range"].to_numpy()
y = df["logratio"].to_numpy()
xlim = [390, 720]
ylim = [-1.0, 0.05]
scatter = {"x": x.tolist(), "y": y.tolist()}
x_grid = np.linspace(x.min(), x.max(), 400)

def save(name, obj):
    json.dump(obj, open(f"assets/{name}.json", "w"), indent=2)

def fit_at(C, D, lam, p=1):
    beta, _, S = fit_pspline(C, y, D, lam, p=p)
    return beta, S

def scan_lambda(C, D, p=1, lam_lo=1e-1, lam_hi=1e4, n_lam=120):
    lams = np.logspace(np.log10(lam_lo), np.log10(lam_hi), n_lam)
    rows = []
    for lam in lams:
        _, S = fit_at(C, D, lam, p=p)
        rows.append({
            "lambda": float(lam),
            "df_fit": df_fit(S),
            "df_res": df_res(S),
            "cv": cv_score(S, y),
            "gcv": gcv_score(S, y),
            "rss": rss_score(S, y),
        })
    return rows

# ===== 公共：24-knot 线性样条 =====
K24 = 24
knots24 = knots_quantile(x, K24)
C24 = linear_spline_basis(x, knots24)
D24 = penalty_matrix(1, K24)
C24g = linear_spline_basis(x_grid, knots24)

# ===== Fig 3.20：λ=32 与 λ=48 =====
panels = []
for lam in [32, 48]:
    beta, S = fit_at(C24, D24, lam)
    panels.append({
        "label": f"λ={lam}",
        "fit_x": x_grid.tolist(),
        "fit_y": (C24g @ beta).tolist(),
        "df_fit": df_fit(S),
    })
save("fig_3_20", {
    "layout": "panels", "fig_id": "3.20", "chapter": 3,
    "title": "λ=32 与 λ=48 的 24 结点线性样条",
    "scatter": scatter, "panels": panels,
    "xlim": xlim, "ylim": ylim,
    "notes": "书中 df 分别约 9.86 和 8.34，两图视觉几乎相同"
})

# ===== 公共 λ 扫描 =====
scan = scan_lambda(C24, D24, p=1)
log_lam = np.log10([r["lambda"] for r in scan])

# ===== Fig 3.21：df_fit vs log10(λ) =====
save("fig_3_21", {
    "layout": "curve", "fig_id": "3.21", "chapter": 3,
    "title": "df_fit 对 log10(λ)",
    "x": log_lam.tolist(), "xlabel": "log10(λ)",
    "curves": [{"name": "df_fit", "y": [r["df_fit"] for r in scan]}],
    "vlines": [],
    "notes": "24 结点线性样条，单调下降"
})

# ===== Fig 3.22：df_fit 与 n - df_res =====
n = len(y)
save("fig_3_22", {
    "layout": "curve", "fig_id": "3.22", "chapter": 3,
    "title": "df_fit 与 n - df_res 对比",
    "x": log_lam.tolist(), "xlabel": "log10(λ)",
    "curves": [
        {"name": "df_fit", "y": [r["df_fit"] for r in scan]},
        {"name": "n - df_res", "y": [n - r["df_res"] for r in scan],
         "dash": "dash"},
    ],
    "vlines": [],
    "notes": "24 结点线性样条；中间平滑区间两者差异最大"
})

# ===== Fig 5.2：REML 与 ML 自动选 λ =====
X_poly = np.column_stack([np.ones_like(x), x])
Z = np.column_stack([np.maximum(x - k, 0.0) for k in knots24])
lam_reml = reml_ml_lambda(X_poly, Z, y, method="REML")
lam_ml   = reml_ml_lambda(X_poly, Z, y, method="ML")

overlay = []
for lam, tag in [(lam_reml, "REML"), (lam_ml, "ML")]:
    beta, S = fit_at(C24, D24, lam)
    overlay.append({
        "name": f"{tag} (λ≈{lam:.1f}, df={df_fit(S):.2f})",
        "fit_x": x_grid.tolist(),
        "fit_y": (C24g @ beta).tolist(),
    })
save("fig_5_2", {
    "layout": "overlay", "fig_id": "5.2", "chapter": 5,
    "title": "REML 与 ML 自动选 λ",
    "scatter": scatter, "curves": overlay,
    "xlim": xlim, "ylim": ylim,
    "notes": f"书中参考 λ_REML≈38.7、λ_ML≈40.8；本页 λ_REML≈{lam_reml:.1f}、λ_ML≈{lam_ml:.1f}"
})

# ===== Fig 5.3：CV 与 RSS vs log10(λ) =====
cv_vals = [r["cv"] for r in scan]
idx_min = int(np.argmin(cv_vals))
save("fig_5_3", {
    "layout": "curve", "fig_id": "5.3", "chapter": 5,
    "title": "CV 与 RSS 对 log10(λ)",
    "x": log_lam.tolist(), "xlabel": "log10(λ)",
    "curves": [
        {"name": "CV", "y": cv_vals},
        {"name": "RSS", "y": [r["rss"] for r in scan], "dash": "dash"},
    ],
    "vlines": [{"x": float(log_lam[idx_min]),
                "label": f"CV 最小 λ≈{scan[idx_min]['lambda']:.1f}"}],
    "notes": "书中 CV 最小 λ≈44.7；RSS 单调增"
})

# ===== Fig 5.9：24 与 48 结点的 CV vs df_fit =====
K48 = 48
knots48 = knots_quantile(x, K48)
C48 = linear_spline_basis(x, knots48)
D48 = penalty_matrix(1, K48)
scan48 = scan_lambda(C48, D48, p=1)
save("fig_5_9", {
    "layout": "curve", "fig_id": "5.9", "chapter": 5,
    "title": "24 与 48 结点的 CV 对 df_fit",
    "x": [r["df_fit"] for r in scan], "xlabel": "df_fit",
    "curves": [
        {"name": "24 knots", "y": [r["cv"] for r in scan]},
        {"name": "48 knots", "y": [r["cv"] for r in scan48], "dash": "dash"},
    ],
    "vlines": [],
    "notes": "两条曲线几乎重合"
})

# ===== Fig 5.10：6/12/24 结点的最小 CV 拟合 =====
overlay = []
for K in [6, 12, 24]:
    kn = knots_quantile(x, K)
    C = linear_spline_basis(x, kn)
    D = penalty_matrix(1, K)
    Cg = linear_spline_basis(x_grid, kn)
    rows = scan_lambda(C, D, p=1)
    best = rows[int(np.argmin([r["cv"] for r in rows]))]
    beta, _ = fit_at(C, D, best["lambda"])
    overlay.append({
        "name": f"{K} knots (λ≈{best['lambda']:.1f})",
        "fit_x": x_grid.tolist(),
        "fit_y": (Cg @ beta).tolist(),
    })
save("fig_5_10", {
    "layout": "overlay", "fig_id": "5.10", "chapter": 5,
    "title": "6/12/24 结点的最小 CV 拟合",
    "scatter": scatter, "curves": overlay,
    "xlim": xlim, "ylim": ylim,
    "notes": "6 结点明显更折线；12 与 24 接近"
})

# ===== 24-knot 二次样条 CV 扫描 =====
C24q = quadratic_spline_basis(x, knots24)
D24q = penalty_matrix(2, K24)
C24qg = quadratic_spline_basis(x_grid, knots24)
scan_q = scan_lambda(C24q, D24q, p=2)

# ===== Fig 5.11：24 结点线性 vs 二次 CV vs df_fit =====
save("fig_5_11", {
    "layout": "curve", "fig_id": "5.11", "chapter": 5,
    "title": "24 结点线性 vs 二次样条的 CV 对 df_fit",
    "x": [r["df_fit"] for r in scan], "xlabel": "df_fit",
    "curves": [
        {"name": "线性样条", "y": [r["cv"] for r in scan]},
        {"name": "二次样条", "y": [r["cv"] for r in scan_q], "dash": "dash"},
    ],
    "vlines": [],
    "notes": "固定 df 时两者 CV 几乎相等"
})

# ===== Fig 5.12：24 结点线性 vs 二次最小 CV 拟合 =====
overlay = []
for tag, C, D, Cg, rows in [
    ("线性", C24, D24, C24g, scan),
    ("二次", C24q, D24q, C24qg, scan_q),
]:
    best = rows[int(np.argmin([r["cv"] for r in rows]))]
    beta, _ = fit_at(C, D, best["lambda"], p=(1 if tag == "线性" else 2))
    overlay.append({
        "name": f"{tag} (λ≈{best['lambda']:.1f})",
        "fit_x": x_grid.tolist(),
        "fit_y": (Cg @ beta).tolist(),
    })
save("fig_5_12", {
    "layout": "overlay", "fig_id": "5.12", "chapter": 5,
    "title": "24 结点线性与二次最小 CV 拟合",
    "scatter": scatter, "curves": overlay,
    "xlim": xlim, "ylim": ylim,
    "notes": "视觉几乎不可区分"
})

# ===== Fig 5.13：6 结点线性 vs 二次最小 CV 拟合 =====
K6 = 6
knots6 = knots_quantile(x, K6)
C6 = linear_spline_basis(x, knots6)
D6 = penalty_matrix(1, K6)
C6g = linear_spline_basis(x_grid, knots6)
scan6 = scan_lambda(C6, D6, p=1)

C6q = quadratic_spline_basis(x, knots6)
D6q = penalty_matrix(2, K6)
C6qg = quadratic_spline_basis(x_grid, knots6)
scan6q = scan_lambda(C6q, D6q, p=2)

overlay = []
for tag, C, D, Cg, rows, p in [
    ("线性", C6, D6, C6g, scan6, 1),
    ("二次", C6q, D6q, C6qg, scan6q, 2),
]:
    best = rows[int(np.argmin([r["cv"] for r in rows]))]
    beta, _ = fit_at(C, D, best["lambda"], p=p)
    overlay.append({
        "name": f"{tag} (λ≈{best['lambda']:.1f})",
        "fit_x": x_grid.tolist(),
        "fit_y": (Cg @ beta).tolist(),
    })
save("fig_5_13", {
    "layout": "overlay", "fig_id": "5.13", "chapter": 5,
    "title": "6 结点线性与二次最小 CV 拟合",
    "scatter": scatter, "curves": overlay,
    "xlim": xlim, "ylim": ylim,
    "notes": "二次样条消除线性样条的折角"
})

print("done")
