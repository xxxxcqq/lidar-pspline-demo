import json, sys
import numpy as np
import pandas as pd

sys.path.append("scripts")
from utils_pspline import (
    linear_spline_basis, knots_quantile
)

df = pd.read_csv("data/lidar.csv")
x = df["range"].to_numpy()
y = df["logratio"].to_numpy()
x_grid = np.linspace(x.min(), x.max(), 400)
xlim, ylim = [390, 720], [-1.0, 0.05]
scatter = {"x": x.tolist(), "y": y.tolist()}

def save(name, obj):
    json.dump(obj, open(f"assets/{name}.json", "w"), indent=2)

# ---------- Fig 2.25 散点 ----------
save("fig_2_25", {
    "fig_id": "2.25", "chapter": 2,
    "title": "LIDAR 散点图",
    "scatter": scatter,
    "xlim": xlim, "ylim": ylim,
    "notes": "书中 Figure 2.25"
})

# ---------- Fig 2.26 多项式 ----------
def poly_fit(degree):
    t = (x - x.min()) / (x.max() - x.min())
    tg = (x_grid - x.min()) / (x.max() - x.min())
    Xp = np.column_stack([t ** d for d in range(degree + 1)])
    Xpg = np.column_stack([tg ** d for d in range(degree + 1)])
    beta = np.linalg.lstsq(Xp, y, rcond=None)[0]
    return {"degree": degree,
            "x": x_grid.tolist(),
            "y": (Xpg @ beta).tolist()}

save("fig_2_26", {
    "fig_id": "2.26", "chapter": 2,
    "title": "三次、四次、十次多项式拟合",
    "scatter": scatter,
    "curves": [poly_fit(d) for d in (3, 4, 10)],
    "xlim": xlim, "ylim": ylim,
    "notes": "三次/四次边界 wiggle；十次多项式振荡"
})

# ---------- 3.6 / 3.7 / 3.8 / 3.9 ----------
def ols_fit(knots):
    C = linear_spline_basis(x, knots)
    Cg = linear_spline_basis(x_grid, knots)
    beta = np.linalg.lstsq(C, y, rcond=None)[0]
    S = C @ np.linalg.pinv(C.T @ C) @ C.T
    return Cg @ beta, float(np.trace(S))

for name, fig_id, kn, note in [
    ("fig_3_6", "3.6", [575.0, 600.0],
     "2 个结点，OLS"),
    ("fig_3_7", "3.7", [500.0, 550.0, 600.0, 650.0],
     "4 个结点，OLS"),
    ("fig_3_8", "3.8", knots_quantile(x, 24).tolist(),
     "24 结点，OLS 过拟合"),
]:
    kn = np.array(kn)
    yg, dfv = ols_fit(kn)
    save(name, {
        "fig_id": fig_id, "chapter": 3,
        "title": f"线性样条，{note}",
        "scatter": scatter,
        "x": x_grid.tolist(), "y": yg.tolist(),
        "knots": kn.tolist(), "df_fit": dfv,
        "xlim": xlim, "ylim": ylim,
        "notes": note
    })

# 3.9：从 24 结点里删掉最接近 612.5/650/662.5/687.5 的四个
kn24 = list(knots_quantile(x, 24))
for v in [612.5, 650.0, 662.5, 687.5]:
    if len(kn24) > 0:
        idx = int(np.argmin(np.abs(np.array(kn24) - v)))
        del kn24[idx]
kn39 = np.array(kn24)
yg39, df39 = ols_fit(kn39)
save("fig_3_9", {
    "fig_id": "3.9", "chapter": 3,
    "title": "删除四个结点后的线性样条",
    "scatter": scatter,
    "x": x_grid.tolist(), "y": yg39.tolist(),
    "knots": kn39.tolist(), "df_fit": df39,
    "xlim": xlim, "ylim": ylim,
    "notes": "从 Fig 3.8 基中删除 612.5, 650, 662.5, 687.5 附近的四个结点"
})

# ---------- manifest ----------
manifest = [
    {"fig_id": "2.25", "chapter": 2, "file": "fig_2_25.json", "title": "LIDAR 散点图"},
    {"fig_id": "2.26", "chapter": 2, "file": "fig_2_26.json", "title": "三次/四次/十次多项式"},
    {"fig_id": "3.1",  "chapter": 3, "file": "fig_3_1.json",  "title": "LIDAR 散点图（引入平滑）"},
    {"fig_id": "3.6",  "chapter": 3, "file": "fig_3_6.json",  "title": "线性样条 2 结点"},
    {"fig_id": "3.7",  "chapter": 3, "file": "fig_3_7.json",  "title": "线性样条 4 结点"},
    {"fig_id": "3.8",  "chapter": 3, "file": "fig_3_8.json",  "title": "24 结点线性样条"},
    {"fig_id": "3.9",  "chapter": 3, "file": "fig_3_9.json",  "title": "删结点后线性样条"},
    {"fig_id": "3.10", "chapter": 3, "file": "fig_3_10.json", "title": "惩罚线性样条 λ=30"},
    {"fig_id": "3.11", "chapter": 3, "file": "fig_3_11.json", "title": "λ=0/10/30/1000 四联图"},
    {"fig_id": "3.20", "chapter": 3, "file": "fig_3_20.json", "title": "λ=32 与 λ=48"},
    {"fig_id": "3.21", "chapter": 3, "file": "fig_3_21.json", "title": "df_fit 对 log λ"},
    {"fig_id": "3.22", "chapter": 3, "file": "fig_3_22.json", "title": "df_fit 与 n−df_res"},
    {"fig_id": "3.25", "chapter": 3, "file": "fig_3_25.json", "title": "四种平滑器各 11 df"},
    {"fig_id": "5.1",  "chapter": 5, "file": "fig_3_11.json", "title": "同 Fig 3.11"},
    {"fig_id": "5.2",  "chapter": 5, "file": "fig_5_2.json",  "title": "REML / ML 自动 λ"},
    {"fig_id": "5.3",  "chapter": 5, "file": "fig_5_3.json",  "title": "CV / RSS 对 log λ"},
    {"fig_id": "5.9",  "chapter": 5, "file": "fig_5_9.json",  "title": "24 与 48 结点 CV"},
    {"fig_id": "5.10", "chapter": 5, "file": "fig_5_10.json", "title": "6/12/24 结点最小 CV"},
    {"fig_id": "5.11", "chapter": 5, "file": "fig_5_11.json", "title": "线性 vs 二次 CV"},
    {"fig_id": "5.12", "chapter": 5, "file": "fig_5_12.json", "title": "24 结点最小 CV"},
    {"fig_id": "5.13", "chapter": 5, "file": "fig_5_13.json", "title": "6 结点最小 CV"},
    {"fig_id": "6.13", "chapter": 6, "file": "fig_6_13.json", "title": "导数推断（全局/局部）"},
    {"fig_id": "6.14", "chapter": 6, "file": "fig_6_14.json", "title": "逐点 vs 同时置信带"},
    {"fig_id": "13.13", "chapter": 13, "file": "fig_13_13.json", "title": "径向平滑 + 样条各 12 df"},
    {"fig_id": "14.1", "chapter": 14, "file": "fig_14_1.json", "title": "散点 + 平滑"},
    {"fig_id": "14.2", "chapter": 14, "file": "fig_14_2.json", "title": "同方差带"},
    {"fig_id": "14.3", "chapter": 14, "file": "fig_14_3.json", "title": "异方差带"},
    {"fig_id": "14.4", "chapter": 14, "file": "fig_14_4.json", "title": "同方差标准化残差"},
    {"fig_id": "14.5", "chapter": 14, "file": "fig_14_5.json", "title": "异方差标准化残差"},
    {"fig_id": "16.1", "chapter": 16, "file": "fig_16_1.json", "title": "贝叶斯六面板"},
    {"fig_id": "16.2", "chapter": 16, "file": "fig_16_2.json", "title": "Gibbs 链诊断"},
]
json.dump(manifest, open("assets/manifest.json", "w"), indent=2)
print("done; manifest =", len(manifest), "entries")
print("fig_3_9 knots =", len(kn39))
