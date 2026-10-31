import json
import os
import numpy as np
import dash
from dash import dcc, html, Input, Output, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 通用加载 ----------
def load(name):
    path = os.path.join(BASE_DIR, "assets", name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

MANIFEST = load("manifest.json") or []

FIG = {
    "2.25": load("fig_2_25.json"),
    "2.26": load("fig_2_26.json"),
    "3.1":  load("fig_3_1.json"),
    "3.6":  load("fig_3_6.json"),
    "3.7":  load("fig_3_7.json"),
    "3.8":  load("fig_3_8.json"),
    "3.9":  load("fig_3_9.json"),
    "3.10": load("fig_3_10.json"),
    "3.11": load("fig_3_11.json"),
    "3.20": load("fig_3_20.json"),
    "3.21": load("fig_3_21.json"),
    "3.22": load("fig_3_22.json"),
    "3.25": load("fig_3_25.json"),
    "5.2":  load("fig_5_2.json"),
    "5.3":  load("fig_5_3.json"),
    "5.9":  load("fig_5_9.json"),
    "5.10": load("fig_5_10.json"),
    "5.11": load("fig_5_11.json"),
    "5.12": load("fig_5_12.json"),
    "5.13": load("fig_5_13.json"),
    "6.13": load("fig_6_13.json"),
    "6.14": load("fig_6_14.json"),
    "13.13": load("fig_13_13.json"),
    "14.1": load("fig_14_1.json"),
    "14.2": load("fig_14_2.json"),
    "14.3": load("fig_14_3.json"),
    "14.4": load("fig_14_4.json"),
    "14.5": load("fig_14_5.json"),
    "16.1": load("fig_16_1.json"),
    "16.2": load("fig_16_2.json"),
}

GREY_MARKER = dict(size=3.5, symbol="circle-open", color="rgba(120,120,120,0.75)")

# ---------- 渲染函数 ----------
def fig_scatter(d):
    if d is None: return _missing("scatter")
    f = go.Figure()
    f.add_trace(go.Scatter(x=d["scatter"]["x"], y=d["scatter"]["y"],
                           mode="markers", marker=GREY_MARKER))
    f.update_layout(xaxis_title="range", yaxis_title="logratio",
                    xaxis_range=d["xlim"], yaxis_range=d["ylim"],
                    template="simple_white")
    return f

def fig_poly(d):
    if d is None: return _missing("poly")
    f = go.Figure()
    f.add_trace(go.Scatter(x=d["scatter"]["x"], y=d["scatter"]["y"],
                           mode="markers", marker=GREY_MARKER, name="data"))
    for c in d["curves"]:
        f.add_trace(go.Scatter(x=c["x"], y=c["y"], mode="lines",
                               line=dict(width=1.6),
                               name=f"degree={c['degree']}"))
    f.update_layout(xaxis_title="range", yaxis_title="logratio",
                    xaxis_range=d["xlim"], yaxis_range=d["ylim"],
                    template="simple_white")
    return f

def fig_ols_spline(d):
    if d is None: return _missing("ols")
    f = go.Figure()
    f.add_trace(go.Scatter(x=d["scatter"]["x"], y=d["scatter"]["y"],
                           mode="markers", marker=GREY_MARKER, name="data"))
    f.add_trace(go.Scatter(x=d["x"], y=d["y"], mode="lines",
                           line=dict(width=1.8, color="black"), name="fit"))
    ymin, ymax = d["ylim"]
    tick = (ymax - ymin) * 0.04
    for k in d["knots"]:
        f.add_trace(go.Scatter(x=[k, k], y=[ymin, ymin + tick],
                               mode="lines", line=dict(color="grey", width=1),
                               showlegend=False))
    f.update_layout(xaxis_title="range", yaxis_title="logratio",
                    xaxis_range=d["xlim"], yaxis_range=d["ylim"],
                    template="simple_white",
                    title=f"Fig {d['fig_id']}：{d['title']}（df={d['df_fit']:.1f}）")
    return f

def fig_pspline(d):
    if d is None: return _missing("pspline")
    f = go.Figure()
    f.add_trace(go.Scatter(x=d["scatter"]["x"], y=d["scatter"]["y"],
                           mode="markers", marker=GREY_MARKER, name="data"))
    f.add_trace(go.Scatter(x=d["fit"]["x"], y=d["fit"]["y"], mode="lines",
                           line=dict(width=1.8, color="black"),
                           name=f"λ={d['book_params']['lambda']}"))
    ymin, ymax = d["ylim"]
    tick = (ymax - ymin) * 0.04
    for k in d["knots"]:
        f.add_trace(go.Scatter(x=[k, k], y=[ymin, ymin + tick],
                               mode="lines", line=dict(color="grey", width=1),
                               showlegend=False))
    f.update_layout(xaxis_title="range", yaxis_title="logratio",
                    xaxis_range=d["xlim"], yaxis_range=d["ylim"],
                    template="simple_white",
                    title=f"Fig {d['fig_id']}：{d['title']}（df={d['df_fit']:.1f}）")
    return f

def fig_quad(d):
    if d is None: return _missing("quad")
    labels = [p["label"] for p in d["panels"]]
    f = make_subplots(rows=2, cols=2, subplot_titles=labels,
                      shared_xaxes=True, shared_yaxes=True,
                      horizontal_spacing=0.06, vertical_spacing=0.12)
    x_s, y_s = d["scatter"]["x"], d["scatter"]["y"]
    ymin, ymax = d["ylim"]
    tick = (ymax - ymin) * 0.04
    for i, p in enumerate(d["panels"]):
        r, c = divmod(i, 2); r += 1; c += 1
        f.add_trace(go.Scatter(x=x_s, y=y_s, mode="markers",
                               marker=dict(size=3, symbol="circle-open", color="grey"),
                               showlegend=False), row=r, col=c)
        f.add_trace(go.Scatter(x=p["fit_x"], y=p["fit_y"], mode="lines",
                               line=dict(width=1.8, color="black"),
                               showlegend=False), row=r, col=c)
        for k in d["knots"]:
            f.add_trace(go.Scatter(x=[k, k], y=[ymin, ymin + tick],
                                   mode="lines", line=dict(color="grey", width=1),
                                   showlegend=False), row=r, col=c)
    f.update_xaxes(range=d["xlim"])
    f.update_yaxes(range=d["ylim"])
    f.update_layout(height=650, template="simple_white",
                    title=f"Fig {d['fig_id']}：{d['title']}")
    return f

DASH_STYLE = {"solid": "solid", "dash": "dash", "dot": "dot"}

def fig_overlay(d):
    if d is None: return _missing("overlay")
    f = go.Figure()
    f.add_trace(go.Scatter(x=d["scatter"]["x"], y=d["scatter"]["y"],
                           mode="markers", marker=GREY_MARKER, name="data"))
    for c in d["curves"]:
        f.add_trace(go.Scatter(
            x=c["fit_x"], y=c["fit_y"], mode="lines",
            line=dict(width=1.8,
                      dash=DASH_STYLE.get(c.get("dash", "solid"), "solid")),
            name=c["name"]))
    f.update_layout(xaxis_title="range", yaxis_title="logratio",
                    xaxis_range=d["xlim"], yaxis_range=d["ylim"],
                    template="simple_white",
                    legend=dict(x=0.02, y=0.98),
                    title=f"Fig {d['fig_id']}：{d['title']}")
    return f

def fig_panels(d):
    if d is None: return _missing("panels")
    labels = [p["label"] for p in d["panels"]]
    f = make_subplots(rows=1, cols=len(labels), subplot_titles=labels,
                      shared_yaxes=True, horizontal_spacing=0.06)
    for i, p in enumerate(d["panels"]):
        r, c = 1, i + 1
        f.add_trace(go.Scatter(x=d["scatter"]["x"], y=d["scatter"]["y"],
                               mode="markers",
                               marker=dict(size=3, symbol="circle-open", color="grey"),
                               showlegend=False), row=r, col=c)
        f.add_trace(go.Scatter(x=p["fit_x"], y=p["fit_y"], mode="lines",
                               line=dict(width=1.8, color="black"),
                               showlegend=False), row=r, col=c)
        f.update_xaxes(range=d["xlim"], row=r, col=c)
        f.update_yaxes(range=d["ylim"], row=r, col=c)
    f.update_layout(height=400, template="simple_white",
                    title=f"Fig {d['fig_id']}：{d['title']}")
    return f

def fig_curve(d):
    if d is None: return _missing("curve")
    f = go.Figure()
    for c in d["curves"]:
        f.add_trace(go.Scatter(
            x=d["x"], y=c["y"], mode="lines",
            line=dict(width=1.8,
                      dash=DASH_STYLE.get(c.get("dash", "solid"), "solid")),
            name=c["name"]))
    for v in d.get("vlines", []):
        f.add_vline(x=v["x"], line=dict(color="grey", dash="dot"),
                    annotation_text=v.get("label", ""),
                    annotation_position="top")
    f.update_layout(xaxis_title=d["xlabel"], yaxis_title="value",
                    template="simple_white",
                    title=f"Fig {d['fig_id']}：{d['title']}")
    return f

def fig_deriv_panels(d):
    if d is None: return _missing("deriv_panels")
    labels = [p["label"] for p in d["panels"]]
    f = make_subplots(rows=len(labels), cols=1, subplot_titles=labels,
                      shared_xaxes=True, vertical_spacing=0.10)
    for i, p in enumerate(d["panels"]):
        r = i + 1
        f.add_trace(go.Scatter(x=p["x"], y=p["y_lo"], mode="lines",
                               line=dict(width=0), showlegend=False),
                    row=r, col=1)
        f.add_trace(go.Scatter(x=p["x"], y=p["y_hi"], mode="lines",
                               line=dict(width=0), fill="tonexty",
                               fillcolor="rgba(120,120,120,0.35)",
                               showlegend=False), row=r, col=1)
        f.add_trace(go.Scatter(x=p["x"], y=p["y"], mode="lines",
                               line=dict(width=1.6, color="black"),
                               showlegend=False), row=r, col=1)
        f.update_xaxes(range=[390, 720], row=r, col=1)
        f.update_yaxes(range=d["ylim"], row=r, col=1)
    f.update_layout(height=520, template="simple_white",
                    title=f"Fig {d['fig_id']}：{d['title']}")
    return f

def fig_deriv_sim(d):
    if d is None: return _missing("deriv_sim")
    f = go.Figure()
    f.add_trace(go.Scatter(x=d["x"], y=d["sim_lo"], mode="lines",
                           line=dict(width=0), showlegend=False))
    f.add_trace(go.Scatter(x=d["x"], y=d["sim_hi"], mode="lines",
                           line=dict(width=0), fill="tonexty",
                           fillcolor="rgba(150,150,150,0.25)",
                           name=f"Simultaneous (m={d['m_095']:.2f})"))
    f.add_trace(go.Scatter(x=d["x"], y=d["point_lo"], mode="lines",
                           line=dict(width=0), showlegend=False))
    f.add_trace(go.Scatter(x=d["x"], y=d["point_hi"], mode="lines",
                           line=dict(width=0), fill="tonexty",
                           fillcolor="rgba(60,60,60,0.45)",
                           name="Pointwise (±2 SE)"))
    f.add_trace(go.Scatter(x=d["x"], y=d["y"], mode="lines",
                           line=dict(width=1.6, color="black"),
                           name="−f'(range)"))
    f.add_hline(y=0, line=dict(color="black", width=0.8))
    f.update_layout(xaxis_title="range", yaxis_title="−f'(range)",
                    xaxis_range=[390, 720], template="simple_white",
                    title=f"Fig {d['fig_id']}：{d['title']}（m_0.95 ≈ {d['m_095']:.2f}）")
    return f

def fig_band(d):
    if d is None: return _missing("band")
    f = go.Figure()
    f.add_trace(go.Scatter(x=d["x"], y=d["band_lo"], mode="lines",
                           line=dict(width=0), showlegend=False))
    f.add_trace(go.Scatter(x=d["x"], y=d["band_hi"], mode="lines",
                           line=dict(width=0), fill="tonexty",
                           fillcolor="rgba(150,150,150,0.35)",
                           name="±2 SE"))
    f.add_trace(go.Scatter(x=d["scatter"]["x"], y=d["scatter"]["y"],
                           mode="markers",
                           marker=dict(size=3, symbol="circle-open", color="grey"),
                           name="data"))
    f.add_trace(go.Scatter(x=d["x"], y=d["fit"], mode="lines",
                           line=dict(width=1.6, color="black"), name="fit"))
    f.update_layout(xaxis_title="range", yaxis_title="logratio",
                    xaxis_range=d["xlim"], yaxis_range=d["ylim"],
                    template="simple_white",
                    title=f"Fig {d['fig_id']}：{d['title']}")
    return f

def fig_scatter_smooth(d):
    if d is None: return _missing("scatter_smooth")
    f = go.Figure()
    f.add_trace(go.Scatter(x=d["scatter"]["x"], y=d["scatter"]["y"],
                           mode="markers", marker=GREY_MARKER, name="data"))
    f.add_trace(go.Scatter(x=d["fit"]["x"], y=d["fit"]["y"], mode="lines",
                           line=dict(width=1.6, color="black"), name="smooth"))
    f.update_layout(xaxis_title="range", yaxis_title="logratio",
                    xaxis_range=d["xlim"], yaxis_range=d["ylim"],
                    template="simple_white",
                    title=f"Fig {d['fig_id']}：{d['title']}")
    return f

def fig_abs_resid(d):
    if d is None: return _missing("abs_resid")
    f = go.Figure()
    f.add_trace(go.Scatter(x=d["scatter"]["x"], y=d["scatter"]["y"],
                           mode="markers", marker=GREY_MARKER,
                           name="|resid|/sd"))
    f.add_trace(go.Scatter(x=d["smooth"]["x"], y=d["smooth"]["y"],
                           mode="lines", line=dict(width=1.8, color="black"),
                           name="smooth"))
    f.update_layout(xaxis_title="range",
                    yaxis_title="normalized absolute residuals",
                    xaxis_range=d["xlim"], yaxis_range=d["ylim"],
                    template="simple_white",
                    title=f"Fig {d['fig_id']}：{d['title']}")
    return f

def fig_bayes_6panel(d):
    if d is None: return _missing("bayes6")
    f = make_subplots(rows=2, cols=3,
                      subplot_titles=["(a) 5 realizations",
                                      "(b) mean & 95% CI",
                                      "(c) posterior SD",
                                      "(d) relative error 1",
                                      "(e) σ_ε density",
                                      "(f) σ_u density"],
                      vertical_spacing=0.18, horizontal_spacing=0.07)
    xg = d["x_grid"]
    sx, sy = d["scatter"]["x"], d["scatter"]["y"]
    grey = dict(size=2.5, symbol="circle-open", color="rgba(150,150,150,0.7)")
    f.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
                           marker=grey, showlegend=False), row=1, col=1)
    for r in d["panel_a"]["realizations"]:
        f.add_trace(go.Scatter(x=xg, y=r, mode="lines",
                               line=dict(width=0.9, color="black"),
                               showlegend=False), row=1, col=1)
    f.add_trace(go.Scatter(x=xg, y=d["panel_b"]["lo"], mode="lines",
                           line=dict(width=0), showlegend=False), row=1, col=2)
    f.add_trace(go.Scatter(x=xg, y=d["panel_b"]["hi"], mode="lines",
                           line=dict(width=0), fill="tonexty",
                           fillcolor="rgba(150,150,150,0.4)",
                           showlegend=False), row=1, col=2)
    f.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
                           marker=grey, showlegend=False), row=1, col=2)
    f.add_trace(go.Scatter(x=xg, y=d["panel_b"]["mean"], mode="lines",
                           line=dict(width=1.6, color="black"),
                           showlegend=False), row=1, col=2)
    f.add_trace(go.Scatter(x=xg, y=d["panel_c"]["sd"], mode="lines",
                           line=dict(width=1.6, color="black"),
                           showlegend=False), row=1, col=3)
    f.add_trace(go.Scatter(x=xg, y=d["panel_d"]["rel_err"], mode="lines",
                           line=dict(width=1.4, color="black"),
                           showlegend=False), row=2, col=1)
    f.add_hline(y=0, line=dict(color="grey", dash="dot"), row=2, col=1)
    f.add_trace(go.Scatter(x=d["panel_e"]["x"], y=d["panel_e"]["density"],
                           mode="lines", line=dict(width=1.4, color="black"),
                           showlegend=False), row=2, col=2)
    f.add_trace(go.Scatter(x=d["panel_f"]["x"], y=d["panel_f"]["density"],
                           mode="lines", line=dict(width=1.4, color="black"),
                           showlegend=False), row=2, col=3)
    f.update_xaxes(range=d["xlim"], row=1, col=1)
    f.update_xaxes(range=d["xlim"], row=1, col=2)
    f.update_xaxes(range=d["xlim"], row=1, col=3)
    f.update_xaxes(range=d["xlim"], row=2, col=1)
    f.update_layout(height=650, template="simple_white",
                    title=f"Fig {d['fig_id']}：{d['title']}")
    return f

def fig_bayes_chain(d):
    if d is None: return _missing("bayeschain")
    f = make_subplots(rows=1, cols=2,
                      subplot_titles=["(a) every 30th", "(b) 1501–1600"],
                      horizontal_spacing=0.10)
    it = np.array(d["iterations"])
    ch = np.array(d["chain"])
    mv = d["overall_mean"]
    f.add_trace(go.Scatter(x=it[::30], y=ch[::30], mode="markers",
                           marker=dict(size=4), showlegend=False),
                row=1, col=1)
    f.add_hline(y=mv, line=dict(color="black", width=1), row=1, col=1)
    f.add_trace(go.Scatter(x=it[1501:1601], y=ch[1501:1601],
                           mode="lines+markers", line=dict(width=0.8),
                           marker=dict(size=4), showlegend=False),
                row=1, col=2)
    f.add_hline(y=mv, line=dict(color="black", width=1), row=1, col=2)
    f.update_layout(height=380, template="simple_white",
                    title=f"Fig {d['fig_id']}：{d['title']}")
    return f

def _missing(tag):
    return go.Figure().update_layout(
        template="simple_white",
        title=f"缺少预计算 JSON（{tag}），请先运行对应脚本"
    )

# ---------- 章节页面 ----------
def _card(fig_id, figure, notes=""):
    return html.Div([
        html.H4(f"Fig {fig_id}"),
        dcc.Graph(figure=figure),
        html.Div(notes, style={"fontSize": "12px", "color": "grey"}),
    ], style={"marginBottom": "20px"})

def page_ch2():
    return html.Div([
        _card("2.25", fig_scatter(FIG["2.25"])),
        _card("2.26", fig_poly(FIG["2.26"]),
              FIG["2.26"]["notes"] if FIG["2.26"] else ""),
    ])

def page_ch3():
    return html.Div([
        _card("3.1",  fig_scatter(FIG["3.1"])),
        _card("3.6",  fig_ols_spline(FIG["3.6"])),
        _card("3.7",  fig_ols_spline(FIG["3.7"])),
        _card("3.8",  fig_ols_spline(FIG["3.8"])),
        _card("3.9",  fig_ols_spline(FIG["3.9"])),
        _card("3.10", fig_pspline(FIG["3.10"])),
        _card("3.11", fig_quad(FIG["3.11"])),
        _card("3.20", fig_panels(FIG["3.20"])),
        _card("3.21", fig_curve(FIG["3.21"])),
        _card("3.22", fig_curve(FIG["3.22"])),
        _card("3.25", fig_overlay(FIG["3.25"]),
              FIG["3.25"]["notes"] if FIG["3.25"] else ""),
    ])

def page_ch5():
    return html.Div([
        _card("5.1（同 3.11）", fig_quad(FIG["3.11"])),
        _card("5.2",  fig_overlay(FIG["5.2"]),
              FIG["5.2"]["notes"] if FIG["5.2"] else ""),
        _card("5.3",  fig_curve(FIG["5.3"])),
        _card("5.9",  fig_curve(FIG["5.9"])),
        _card("5.10", fig_overlay(FIG["5.10"])),
        _card("5.11", fig_curve(FIG["5.11"])),
        _card("5.12", fig_overlay(FIG["5.12"])),
        _card("5.13", fig_overlay(FIG["5.13"])),
    ])

def page_ch6():
    return html.Div([
        _card("6.13", fig_deriv_panels(FIG["6.13"]),
              FIG["6.13"]["notes"] if FIG["6.13"] else ""),
        _card("6.14", fig_deriv_sim(FIG["6.14"]),
              FIG["6.14"]["notes"] if FIG["6.14"] else ""),
    ])

def page_ch13():
    return html.Div([
        _card("13.13", fig_overlay(FIG["13.13"]),
              FIG["13.13"]["notes"] if FIG["13.13"] else ""),
    ])

def page_ch14():
    return html.Div([
        _card("14.1", fig_scatter_smooth(FIG["14.1"])),
        _card("14.2", fig_band(FIG["14.2"])),
        _card("14.3", fig_band(FIG["14.3"])),
        _card("14.4", fig_abs_resid(FIG["14.4"])),
        _card("14.5", fig_abs_resid(FIG["14.5"])),
    ])

def page_ch16():
    return html.Div([
        _card("16.1", fig_bayes_6panel(FIG["16.1"]),
              FIG["16.1"]["notes"] if FIG["16.1"] else ""),
        _card("16.2", fig_bayes_chain(FIG["16.2"]),
              FIG["16.2"]["notes"] if FIG["16.2"] else ""),
    ])

# ---------- 探索模式（沿用第四步） ----------
QUAD = FIG["3.11"]

def page_explore():
    if QUAD is None:
        return html.Div("缺少 fig_3_11.json，请先运行 scripts/04_fig_3_11.py")
    return html.Div([
        html.H4("探索模式：拖动 λ 滑条"),
        html.Div([
            html.Button("λ=0",    id="btn-0",    n_clicks=0),
            html.Button("λ=10",   id="btn-10",   n_clicks=0),
            html.Button("λ=30",   id="btn-30",   n_clicks=0),
            html.Button("λ=1000", id="btn-1000", n_clicks=0),
            html.Span(" 显示 knots ", style={"marginLeft": "20px"}),
            dcc.Checklist(id="show-knots",
                          options=[{"label": "", "value": "on"}],
                          value=["on"], inline=True),
        ], style={"marginBottom": "10px"}),
        dcc.Slider(id="lambda-slider",
                   min=-1, max=3, step=0.05,
                   value=float(np.log10(30)),
                   marks={-1: "0.1", 0: "1", 1: "10", 2: "100", 3: "1000"},
                   tooltip={"placement": "bottom"}),
        dcc.Graph(id="explore-graph"),
        html.Div(id="explore-info", style={"fontSize": "16px"}),
    ])

def find_closest_fit(lam):
    if QUAD is None:
        return None
    lams = np.array([g["lambda"] for g in QUAD["lambda_grid"]])
    logl = np.log10(np.maximum(lams, 1e-3))
    idx = int(np.argmin(np.abs(logl - np.log10(max(lam, 1e-3)))))
    return QUAD["lambda_grid"][idx]

def build_explore_fig(fit_x, fit_y, show_knots, label, df):
    f = go.Figure()
    f.add_trace(go.Scatter(x=QUAD["scatter"]["x"], y=QUAD["scatter"]["y"],
                           mode="markers", marker=GREY_MARKER, name="data"))
    f.add_trace(go.Scatter(x=fit_x, y=fit_y, mode="lines",
                           line=dict(color="black", width=2), name=label))
    if show_knots and "on" in show_knots:
        ymin, ymax = QUAD["ylim"]
        tick = (ymax - ymin) * 0.04
        for k in QUAD["knots"]:
            f.add_trace(go.Scatter(x=[k, k], y=[ymin, ymin + tick],
                                   mode="lines",
                                   line=dict(color="grey", width=1),
                                   showlegend=False))
    f.update_layout(xaxis_title="range", yaxis_title="logratio",
                    xaxis_range=QUAD["xlim"], yaxis_range=QUAD["ylim"],
                    template="simple_white",
                    title=f"{label}，df_fit = {df:.2f}")
    return f

# ---------- 全图索引 ----------
def page_index():
    if not MANIFEST:
        return html.Div("缺少 manifest.json，请先运行 scripts/10_finalize.py")
    rows = [html.Tr([html.Th(h) for h in
                     ["图号", "章节", "标题", "数据文件", "预计算 JSON"]])]
    for m in MANIFEST:
        rows.append(html.Tr([
            html.Td(m["fig_id"]),
            html.Td(m["chapter"]),
            html.Td(m["title"]),
            html.Td("Session2.7_lidar.txt"),
            html.Td(m["file"]),
        ]))
    return html.Div([
        html.H4(f"全图索引（共 {len(MANIFEST)} 条，含 Fig 5.1 复用 Fig 3.11）"),
        html.P("使用上方 Tab 切换章节查看。"),
        html.Table(rows, style={
            "borderCollapse": "collapse",
            "fontSize": "13px",
            "width": "100%",
        }),
    ])

# ---------- 主应用 ----------
app = dash.Dash(__name__)
app.title = "LIDAR 惩罚样条复现"

app.layout = html.Div([
    html.H2("LIDAR 惩罚样条全流程复现（Session2.7_lidar.txt，n=221）"),
    dcc.Tabs(id="tabs", value="ch3", children=[
        dcc.Tab(label="第 2 章", value="ch2"),
        dcc.Tab(label="第 3 章", value="ch3"),
        dcc.Tab(label="第 5 章", value="ch5"),
        dcc.Tab(label="第 6 章", value="ch6"),
        dcc.Tab(label="第 13 章", value="ch13"),
        dcc.Tab(label="第 14 章", value="ch14"),
        dcc.Tab(label="第 16 章", value="ch16"),
        dcc.Tab(label="探索模式", value="explore"),
        dcc.Tab(label="全图索引", value="index"),
    ]),
    html.Div(id="tab-content", style={"padding": "20px"}),
], style={"maxWidth": "1200px", "margin": "0 auto",
          "fontFamily": "system-ui, -apple-system, sans-serif"})

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(value):
    return {
        "ch2": page_ch2,
        "ch3": page_ch3,
        "ch5": page_ch5,
        "ch6": page_ch6,
        "ch13": page_ch13,
        "ch14": page_ch14,
        "ch16": page_ch16,
        "explore": page_explore,
        "index": page_index,
    }[value]()

@app.callback(
    Output("explore-graph", "figure"),
    Output("explore-info", "children"),
    Output("lambda-slider", "value"),
    Input("lambda-slider", "value"),
    Input("btn-0", "n_clicks"),
    Input("btn-10", "n_clicks"),
    Input("btn-30", "n_clicks"),
    Input("btn-1000", "n_clicks"),
    Input("show-knots", "value"),
)
def update_explore(slider_log, b0, b10, b30, b1000, show_knots):
    if QUAD is None:
        return _missing("quad"), "缺少 fig_3_11.json", slider_log

    trigger = ctx.triggered_id
    if trigger == "btn-0":
        lam, slider_log = 0.0, -1.0
    elif trigger == "btn-10":
        lam, slider_log = 10.0, 1.0
    elif trigger == "btn-30":
        lam = 30.0
        slider_log = float(np.log10(30))
    elif trigger == "btn-1000":
        lam, slider_log = 1000.0, 3.0
    else:
        lam = 10 ** slider_log

    if lam == 0:
        p = QUAD["panels"][0]
        fig = build_explore_fig(p["fit_x"], p["fit_y"],
                                show_knots, "λ=0 (OLS)", p["df_fit"])
        info = f"当前：λ=0 (OLS)，df_fit = {p['df_fit']:.2f}"
    else:
        g = find_closest_fit(lam)
        fig = build_explore_fig(QUAD["x_grid"], g["fit_y"],
                                show_knots,
                                f"λ≈{g['lambda']:.1f}", g["df_fit"])
        info = f"当前：λ≈{g['lambda']:.1f}，df_fit = {g['df_fit']:.2f}"
    return fig, info, slider_log

server = app.server

if __name__ == "__main__":
    app.run(debug=True)
