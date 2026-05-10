"""Simple Dash app to visualize benchmark CSVs interactively.

Run: python benchmarks/dashboard_app.py --csv benchmarks/results.csv
"""
import argparse
from pathlib import Path

import math
import pandas as pd
import numpy as np

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    dash = None


def build_app(df: pd.DataFrame):
    app = dash.Dash(__name__)
    algorithms = sorted(df["algorithm"].unique())
    all_backends = sorted(df["backend"].unique())
    # two-column layout: left column for Algorithms, right column for Backends
    left_col = html.Div([
        html.Label("Algorithms"),
        dcc.Dropdown(
            id="algo",
            options=[{"label": "All", "value": "__all__"}] + [{"label": a, "value": a} for a in algorithms],
            value=["__all__"],
            multi=True,
            style={"width": "100%", "display": "block"},
        ),
        html.Div([
            html.Label("X Scale", style={"marginRight": "8px", "marginLeft": "2px"}),
            dcc.RadioItems(
                id="x_scale",
                options=[{"label": "Linear", "value": "linear"}, {"label": "Log", "value": "log"}],
                value="log",
                labelStyle={"display": "inline-block", "marginRight": "10px"},
                style={"marginTop": "6px"},
            ),
        ], style={"marginTop": "6px"}),
    ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "overflow": "hidden"})

    right_col = html.Div([
        html.Label("Backends"),
        dcc.Dropdown(id="backend", options=[{"label": b, "value": b} for b in all_backends], value=all_backends, multi=True, style={"width": "100%", "display": "block"}),
        html.Div([
            html.Label("Y Scale", style={"marginRight": "8px"}),
            dcc.RadioItems(
                id="y_scale",
                options=[{"label": "Linear", "value": "linear"}, {"label": "Log", "value": "log"}],
                value="log",
                labelStyle={"display": "inline-block", "marginRight": "10px"},
                style={"marginTop": "6px"},
            ),
        ], style={"marginTop": "6px"}),
    ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "overflow": "hidden", "float": "right"})

    # X/Y scale radios moved inside left_col and right_col

    # Plot type UI removed; plotting defaults to line with bands

    # appearance control removed; dashboard defaults to light mode

    # Build layout: header, selectors, tabs (Timing | Speedup)
    app.layout = html.Div([
        html.Div([
            html.H3("Pythonic-Algorithms Benchmarks", id="heading", style={"marginBottom": "6px"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),

        dcc.Tabs(id="tabs", value="timing", children=[
            dcc.Tab(label="Timing", value="timing", children=[
                html.Div([left_col, right_col], style={"display": "flex", "justifyContent": "space-between", "gap": "12px", "marginTop": "8px"}),
                html.Div([dcc.Graph(id="graph", style={"width": "100%"})], style={"marginTop": "12px"}),
            ]),
            dcc.Tab(label="CPU vs GPU Speedup", value="speedup", children=[
                html.Div([
                    html.P("Speedup = cpu_mean / gpu_mean per algorithm and input size. Values > 1 mean GPU is faster.",
                           style={"marginTop": "12px", "color": "#555"}),
                    dcc.Graph(id="speedup-graph", style={"width": "100%"}),
                ], style={"marginTop": "8px"}),
            ]),
        ]),

    ], id="page-container", className="light", style={"backgroundColor": "#ffffff", "color": "#000000", "padding": "12px"})

    @app.callback(
        [
            Output("graph", "figure"),
            Output("page-container", "style"),
            Output("page-container", "className"),
            Output("heading", "style"),
            Output("algo", "style"),
            Output("backend", "style"),
        ],
        [
            Input("algo", "value"),
            Input("backend", "value"),
            Input("x_scale", "value"),
            Input("y_scale", "value"),
        ],
    )
    def update(algos, backends, x_scale, y_scale):
        # normalize algorithm selection: support the special '__all__' token
        if not algos:
            algos = algorithms
        elif isinstance(algos, (list, tuple)) and "__all__" in algos:
            algos = algorithms
        if not backends:
            backends = all_backends

        # Appearance control removed; default to light theme
        template = "plotly"
        container_style = {"backgroundColor": "#ffffff", "color": "#000000", "padding": "12px"}
        heading_style = {"color": "#000000", "marginBottom": "6px"}
        container_class = "light"

        # dropdown styles (light theme only)
        dropdown_style = {"width": "100%", "display": "block", "backgroundColor": "#ffffff", "color": "#000000", "border": "1px solid #ccc"}

        d = df[df["algorithm"].isin(algos) & df["backend"].isin(backends)].copy()

        def mkout(fig):
            return fig, container_style, container_class, heading_style, dropdown_style, dropdown_style

        if d.empty:
            fig = px.line(title="No data for selection")
            fig.update_layout(template=template)
            return mkout(fig)

        d["n"] = pd.to_numeric(d["n"], errors="coerce")
        d["mean"] = pd.to_numeric(d["mean"], errors="coerce")
        d["min"] = pd.to_numeric(d.get("min", pd.Series()), errors="coerce")
        d["max"] = pd.to_numeric(d.get("max", pd.Series()), errors="coerce")

        # Plot type is fixed to line with bands
        plot_type = "line_bands"

        # Helper: convert hex color to rgba string with alpha
        def hex_to_rgba(hex_color, alpha=0.15):
            hex_color = hex_color.lstrip("#")
            if len(hex_color) == 3:
                hex_color = "".join([c * 2 for c in hex_color])
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        # LINE + BANDS: mean line with min/max filled band
        if plot_type == "line_bands":
            fig = go.Figure()
            # color and dash cycles
            colors = px.colors.qualitative.Plotly
            dash_styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
            algs = sorted(d["algorithm"].unique())
            alg_color = {alg: colors[i % len(colors)] for i, alg in enumerate(algs)}
            backends_used = sorted(d["backend"].unique())
            backend_dash = {b: dash_styles[i % len(dash_styles)] for i, b in enumerate(backends_used)}

            for alg in algs:
                for b in backends_used:
                    grp = d[(d["algorithm"] == alg) & (d["backend"] == b)].sort_values("n")
                    if grp.empty:
                        continue
                    x = grp["n"].tolist()
                    y_mean = grp["mean"].tolist()
                    y_min = grp["min"].tolist() if "min" in grp.columns else [None] * len(x)
                    y_max = grp["max"].tolist() if "max" in grp.columns else [None] * len(x)
                    color = alg_color[alg]
                    # add filled band using a closed polygon
                    if any([v is not None and not np.isnan(v) for v in y_min]) and any([v is not None and not np.isnan(v) for v in y_max]):
                        xs = x + x[::-1]
                        ys = y_max + y_min[::-1]
                        fig.add_trace(
                            go.Scatter(
                                x=xs,
                                y=ys,
                                fill="toself",
                                fillcolor=hex_to_rgba(color, 0.12),
                                line=dict(color="rgba(255,255,255,0)"),
                                hoverinfo="skip",
                                showlegend=False,
                            )
                        )
                    # add mean line
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y_mean,
                            mode="lines+markers",
                            name=f"{alg} ({b})",
                            line=dict(color=color, dash=backend_dash.get(b, "solid")),
                            marker=dict(size=6),
                        )
                    )

            fig.update_layout(
                height=800,
                width=1200,
                font=dict(size=14),
                margin=dict(l=70, r=30, t=70, b=140),
                legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center", yanchor="top"),
                template=template,
            )
            fig.update_xaxes(title_text="n", title_font=dict(size=16), tickfont=dict(size=12), type=("log" if x_scale == "log" else "linear"))
            fig.update_yaxes(title_text="mean time (s)", title_font=dict(size=16), tickfont=dict(size=12), type=("log" if y_scale == "log" else "linear"))
            return mkout(fig)

        # FACETS: small multiples per algorithm
        if plot_type == "facets":
            selected_algs = sorted(d["algorithm"].unique())
            if len(selected_algs) > 12:
                fig = px.line(title="Too many algorithms for facets. Select ≤12 algorithms.")
                fig.update_layout(template=template)
                return mkout(fig)
            fig = px.line(
                d,
                x="n",
                y="mean",
                color="backend",
                facet_col="algorithm",
                facet_col_wrap=4,
                markers=True,
            )
            rows = math.ceil(len(selected_algs) / 4) if selected_algs else 1
            fig.update_layout(height=max(400, rows * 250), width=1200, margin=dict(l=70, r=30, t=70, b=140), template=template)
            fig.for_each_xaxis(lambda ax: ax.update(type=("log" if x_scale == "log" else "linear")))
            fig.for_each_yaxis(lambda ax: ax.update(type=("log" if y_scale == "log" else "linear")))
            fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center", yanchor="top"))
            return mkout(fig)

        # BOX / VIOLIN: show distribution of mean times per algorithm
        if plot_type == "box_violin":
            # use violin with box + points for richer distribution
            fig = px.violin(d, x="algorithm", y="mean", color="backend", box=True, points="all")
            fig.update_layout(height=700, width=1200, margin=dict(l=70, r=30, t=70, b=140), template=template)
            fig.update_xaxes(title_text="algorithm", tickfont=dict(size=11))
            fig.update_yaxes(title_text="mean time (s)", type=("log" if y_scale == "log" else "linear"))
            fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center", yanchor="top"))
            return mkout(fig)

        # HEATMAP: algorithm x n (mean)
        if plot_type == "heatmap":
            pivot = d.pivot_table(index="algorithm", columns="n", values="mean", aggfunc="mean")
            if pivot.empty:
                fig = px.imshow([[0]], title="No data for heatmap")
                fig.update_layout(template=template)
                return mkout(fig)
            # sort rows by median performance for nicer ordering
            pivot = pivot.loc[pivot.median(axis=1).sort_values().index]
            fig = px.imshow(pivot, labels={"x": "n", "y": "algorithm", "color": "mean"}, aspect="auto", origin="lower")
            fig.update_layout(height=max(400, pivot.shape[0] * 24), width=1200, margin=dict(l=140, r=30, t=70, b=140), template=template)
            return mkout(fig)

        # Fallback to simple line
        fig = px.line(d, x="n", y="mean", color="algorithm", line_dash="backend", markers=True)
        fig.update_layout(height=800, width=1200, font=dict(size=14), margin=dict(l=70, r=30, t=70, b=140), template=template)
        fig.update_xaxes(title_text="n", title_font=dict(size=16), tickfont=dict(size=12), type=("log" if x_scale == "log" else "linear"))
        fig.update_yaxes(title_text="mean time (s)", title_font=dict(size=16), tickfont=dict(size=12), type=("log" if y_scale == "log" else "linear"))
        return mkout(fig)

    @app.callback(
        Output("speedup-graph", "figure"),
        Input("tabs", "value"),
    )
    def update_speedup(tab):
        if tab != "speedup":
            return go.Figure()
        d = df.copy()
        d["n"] = pd.to_numeric(d["n"], errors="coerce")
        d["mean"] = pd.to_numeric(d["mean"], errors="coerce")
        d = d.dropna(subset=["n", "mean"])

        def _canonical(name):
            for suffix in ("_numpy", "_cupy", "_numba", "_tiled"):
                if name.endswith(suffix):
                    return name[: -len(suffix)]
            return {"bfs_frontier": "bfs"}.get(name, name)

        d["_key"] = d["algorithm"].apply(_canonical)
        cpu = d[d["backend"] == "cpu"][["_key", "n", "mean"]].rename(columns={"mean": "cpu_mean"})
        gpu = d[d["backend"] == "gpu"][["_key", "n", "mean"]].rename(columns={"mean": "gpu_mean"})
        merged = pd.merge(cpu, gpu, on=["_key", "n"])
        if merged.empty:
            fig = go.Figure()
            fig.update_layout(title="No paired CPU/GPU data found", template="plotly")
            return fig
        merged["speedup"] = merged["cpu_mean"] / merged["gpu_mean"]
        merged = merged.sort_values(["_key", "n"])
        fig = px.line(
            merged,
            x="n",
            y="speedup",
            color="_key",
            markers=True,
            log_x=True,
            title="GPU Speedup over CPU (cpu_mean / gpu_mean)",
            labels={"speedup": "Speedup (×)", "n": "Input size n", "_key": "algorithm"},
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="break-even")
        fig.update_layout(
            height=700,
            width=1200,
            font=dict(size=14),
            margin=dict(l=70, r=30, t=70, b=80),
            legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center", yanchor="top"),
            template="plotly",
        )
        return fig

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="benchmarks/results.csv")
    args = parser.parse_args()
    p = Path(args.csv)
    if not p.exists():
        print("CSV not found:", p)
        return
    df = pd.read_csv(p)
    if dash is None:
        print("Dash is not installed. Install requirements (plotly,dash) to run the app.")
        return
    app = build_app(df)
    app.run(debug=True)


if __name__ == "__main__":
    main()
