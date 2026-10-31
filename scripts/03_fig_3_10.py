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
lam = 30.0

C = linear_spline_basis(x, knots)
D = penalty_matrix(1, K)
beta, y_hat, S = fit_pspline(C, y, D, lam)

# 在细网格上画曲线
x_grid = np.linspace(x.min(), x.max(), 400)
C_grid = linear_spline_basis(x_grid, knots)
y_grid = C_grid @ beta

out = {
    "fig_id": "3.10",
    "chapter": 3,
    "title": "惩罚线性样条，λ=30",
    "book_params": {"K": K, "degree": 1, "lambda": lam, "knots": knots.tolist()},
    "scatter": {"x": x.tolist(), "y": y.tolist()},
    "fit": {"x": x_grid.tolist(), "y": y_grid.tolist()},
    "knots": knots.tolist(),
    "df_fit": df_fit(S),
    "df_res": df_res(S),
    "xlim": [390, 720],
    "ylim": [-1.0, 0.05],
    "notes": "书中 Figure 3.10，24 knots，λ=30"
}
json.dump(out, open("assets/fig_3_10.json", "w"), indent=2)
print("df_fit =", out["df_fit"], "df_res =", out["df_res"])
