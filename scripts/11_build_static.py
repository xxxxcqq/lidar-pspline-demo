# -*- coding: utf-8 -*-
"""把 Dash 应用导出为纯静态站点：dist/public/{index.html,app.js,data.js,plotly.min.js}"""
import json, os, sys, shutil
import numpy as np
import plotly
import plotly.utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import app as A  # 复用 app.py 里的渲染函数与数据加载

OUT = os.path.join(ROOT, "dist", "public")
os.makedirs(OUT, exist_ok=True)

FIG = A.FIG
MANIFEST = A.MANIFEST

FIG_FUNCS = {
    "2.25": A.fig_scatter,
    "2.26": A.fig_poly,
    "3.1":  A.fig_scatter,
    "3.6":  A.fig_ols_spline,
    "3.7":  A.fig_ols_spline,
    "3.8":  A.fig_ols_spline,
    "3.9":  A.fig_ols_spline,
    "3.10": A.fig_pspline,
    "3.11": A.fig_quad,
    "3.20": A.fig_panels,
    "3.21": A.fig_curve,
    "3.22": A.fig_curve,
    "3.25": A.fig_overlay,
    "5.2":  A.fig_overlay,
    "5.3":  A.fig_curve,
    "5.9":  A.fig_curve,
    "5.10": A.fig_overlay,
    "5.11": A.fig_curve,
    "5.12": A.fig_overlay,
    "5.13": A.fig_overlay,
    "6.13": A.fig_deriv_panels,
    "6.14": A.fig_deriv_sim,
    "13.13": A.fig_overlay,
    "14.1": A.fig_scatter_smooth,
    "14.2": A.fig_band,
    "14.3": A.fig_band,
    "14.4": A.fig_abs_resid,
    "14.5": A.fig_abs_resid,
    "16.1": A.fig_bayes_6panel,
    "16.2": A.fig_bayes_chain,
}

def slim(obj):
    """把 numpy 数组转 list，并把浮点压到 6 位有效数字以减小体积"""
    if isinstance(obj, dict):
        return {k: slim(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [slim(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return slim(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if v != 0 and np.isfinite(v):
            return float(f"{v:.6g}")
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

figures = {}
for fid, fn in FIG_FUNCS.items():
    d = FIG.get(fid)
    if d is None:
        print("missing data:", fid); continue
    fig = fn(d)
    js = fig.to_plotly_json()
    figures[fid] = {
        "figure": slim({"data": js["data"], "layout": js["layout"]}),
        "notes": d.get("notes", ""),
    }

QUAD = FIG["3.11"]
explore = {
    "scatter": slim(QUAD["scatter"]),
    "knots": slim(QUAD["knots"]),
    "xlim": slim(QUAD["xlim"]),
    "ylim": slim(QUAD["ylim"]),
    "x_grid": slim(QUAD["x_grid"]),
    "grid": [{"lambda": slim(g["lambda"]),
              "df_fit": slim(g["df_fit"]),
              "y": slim(g["fit_y"])} for g in QUAD["lambda_grid"]],
    "ols": {"y": slim(QUAD["panels"][0]["fit_y"]),
            "df_fit": slim(QUAD["panels"][0]["df_fit"])},
}

pages = [
    {"id": "ch2", "label": "第 2 章", "cards": [
        {"label": "Fig 2.25", "fig": "2.25"},
        {"label": "Fig 2.26", "fig": "2.26"},
    ]},
    {"id": "ch3", "label": "第 3 章", "cards": [
        {"label": "Fig 3.1", "fig": "3.1"},
        {"label": "Fig 3.6", "fig": "3.6"},
        {"label": "Fig 3.7", "fig": "3.7"},
        {"label": "Fig 3.8", "fig": "3.8"},
        {"label": "Fig 3.9", "fig": "3.9"},
        {"label": "Fig 3.10", "fig": "3.10"},
        {"label": "Fig 3.11", "fig": "3.11"},
        {"label": "Fig 3.20", "fig": "3.20"},
        {"label": "Fig 3.21", "fig": "3.21"},
        {"label": "Fig 3.22", "fig": "3.22"},
        {"label": "Fig 3.25", "fig": "3.25"},
    ]},
    {"id": "ch5", "label": "第 5 章", "cards": [
        {"label": "Fig 5.1（同 3.11）", "fig": "3.11"},
        {"label": "Fig 5.2", "fig": "5.2"},
        {"label": "Fig 5.3", "fig": "5.3"},
        {"label": "Fig 5.9", "fig": "5.9"},
        {"label": "Fig 5.10", "fig": "5.10"},
        {"label": "Fig 5.11", "fig": "5.11"},
        {"label": "Fig 5.12", "fig": "5.12"},
        {"label": "Fig 5.13", "fig": "5.13"},
    ]},
    {"id": "ch6", "label": "第 6 章", "cards": [
        {"label": "Fig 6.13", "fig": "6.13"},
        {"label": "Fig 6.14", "fig": "6.14"},
    ]},
    {"id": "ch13", "label": "第 13 章", "cards": [
        {"label": "Fig 13.13", "fig": "13.13"},
    ]},
    {"id": "ch14", "label": "第 14 章", "cards": [
        {"label": "Fig 14.1", "fig": "14.1"},
        {"label": "Fig 14.2", "fig": "14.2"},
        {"label": "Fig 14.3", "fig": "14.3"},
        {"label": "Fig 14.4", "fig": "14.4"},
        {"label": "Fig 14.5", "fig": "14.5"},
    ]},
    {"id": "ch16", "label": "第 16 章", "cards": [
        {"label": "Fig 16.1", "fig": "16.1"},
        {"label": "Fig 16.2", "fig": "16.2"},
    ]},
    {"id": "explore", "label": "探索模式", "cards": []},
    {"id": "index", "label": "全图索引", "cards": []},
]

data = {
    "meta": {
        "title": "LIDAR 惩罚样条全流程复现",
        "subtitle": "《Semiparametric Regression》LIDAR 数据（Session2.7_lidar.txt，n=221）· 预计算 JSON 静态渲染",
    },
    "figs": figures,
    "pages": pages,
    "explore": explore,
    "manifest": slim(MANIFEST),
}

with open(os.path.join(OUT, "data.js"), "w", encoding="utf-8") as f:
    f.write("window.__DATA=")
    json.dump(data, f, ensure_ascii=False, separators=(",", ":"),
              cls=plotly.utils.PlotlyJSONEncoder)
    f.write(";\n")

# plotly.js 用 Python 包里自带的那份，保证 schema 版本一致
src = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
shutil.copyfile(src, os.path.join(OUT, "plotly.min.js"))

print("figures:", len(figures), "pages:", len(pages))
print("data.js MB:", round(os.path.getsize(os.path.join(OUT, "data.js")) / 1e6, 2))
print("plotly.min.js MB:", round(os.path.getsize(os.path.join(OUT, "plotly.min.js")) / 1e6, 2))
