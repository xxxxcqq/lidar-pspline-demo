import json, sys
import numpy as np
import pandas as pd

sys.path.append("scripts")
from utils_pspline import (
    linear_spline_basis, penalty_matrix, fit_pspline,
    df_fit, df_res, knots_quantile
)

df = pd.read_csv("data/lidar.csv")
x = df["range"].to_numpy()
y = df["logratio"].to_numpy()

K = 24
knots = knots_quantile(x, K)
C = linear_spline_basis(x, knots)
D = penalty_matrix(1, K)

# 书中 Figure 3.11 的四个 λ
lambdas = [0, 10, 30, 1000]

# 细网格用于画曲线
x_grid = np.linspace(x.min(), x.max(), 400)
C_grid = linear_spline_basis(x_grid, knots)

panels = []
for lam in lambdas:
    beta, _, S = fit_pspline(C, y, D, lam)
    y_grid = C_grid @ beta
    panels.append({
        "label": f"λ={lam}",
        "lambda": lam,
        "fit_x": x_grid.tolist(),
        "fit_y": y_grid.tolist(),
        "df_fit": df_fit(S),
        "df_res": df_res(S),
    })

# 为第四步交互滑条预计算连续 λ 网格（对数间隔 60 个点）
lam_grid = np.logspace(-1, 3, 60)
grid_fits = []
for lam in lam_grid:
    beta, _, S = fit_pspline(C, y, D, lam)
    grid_fits.append({
        "lambda": float(lam),
        "fit_y": (C_grid @ beta).tolist(),
        "df_fit": df_fit(S),
    })

out = {
    "fig_id": "3.11",
    "chapter": 3,
    "title": "λ=0,10,30,1000 四联图",
    "book_params": {"K": K, "degree": 1, "lambda": lambdas,
                    "knots": knots.tolist()},
    "scatter": {"x": x.tolist(), "y": y.tolist()},
    "x_grid": x_grid.tolist(),
    "knots": knots.tolist(),
    "panels": panels,
    "lambda_grid": grid_fits,
    "xlim": [390, 720],
    "ylim": [-1.0, 0.05],
    "notes": "书中 Figure 3.11；Figure 5.1 与本图相同，可复用"
}
json.dump(out, open("assets/fig_3_11.json", "w"), indent=2)

for p in panels:
    print(p["label"], "df_fit =", round(p["df_fit"], 2))
