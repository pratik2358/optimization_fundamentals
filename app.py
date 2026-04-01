import math
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.function_lib import BUILTIN_FUNCTIONS, compile_user_function, function_help_markdown
from src.optimizers import OPTIMIZER_SPECS, run_optimizer


st.set_page_config(
    page_title="Optimizer Explorer",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📉 Optimizer Explorer")
st.markdown(
    "Explore how different optimization algorithms move across convex and non-convex surfaces. "
    "Choose a built-in function or enter your own scalar function of `x` and `y`, then tune the optimizer and watch the descent path."
)


def make_surface_figure(
    f,
    trajectory: List[Dict[str, float]],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    grid_size: int,
    title: str,
) -> go.Figure:
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    zz = np.asarray(f(jnp.asarray(xx), jnp.asarray(yy)), dtype=float)

    traj_x = np.array([p["x"] for p in trajectory], dtype=float)
    traj_y = np.array([p["y"] for p in trajectory], dtype=float)
    traj_z = np.asarray(f(jnp.asarray(traj_x), jnp.asarray(traj_y)), dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            colorscale="Viridis",
            opacity=0.9,
            showscale=True,
            name="surface",
            colorbar=dict(title="f(x, y)"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=traj_x,
            y=traj_y,
            z=traj_z,
            mode="lines+markers",
            line=dict(color="red", width=6),
            marker=dict(size=4, color=np.arange(len(traj_x)), colorscale="Plasma"),
            name="descent path",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[traj_x[0]],
            y=[traj_y[0]],
            z=[traj_z[0]],
            mode="markers",
            marker=dict(size=8, color="orange", symbol="diamond"),
            name="start",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[traj_x[-1]],
            y=[traj_y[-1]],
            z=[traj_z[-1]],
            mode="markers",
            marker=dict(size=8, color="lime", symbol="circle"),
            name="end",
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x, y)",
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.95)),
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


@st.cache_data(show_spinner=False)
def compute_trajectory(
    optimizer_name: str,
    function_source: str,
    x0: float,
    y0: float,
    steps: int,
    params: Tuple[Tuple[str, float], ...],
):
    f = compile_user_function(function_source)
    result = run_optimizer(
        optimizer_name=optimizer_name,
        f=f,
        start=jnp.array([x0, y0], dtype=jnp.float32),
        steps=steps,
        params=dict(params),
    )
    return result


with st.sidebar:
    st.header("Function")
    use_custom = st.toggle("Use custom function", value=False)

    if not use_custom:
        selected_builtin = st.selectbox(
            "Choose a built-in function",
            list(BUILTIN_FUNCTIONS.keys()),
            index=0,
        )
        builtin_meta = BUILTIN_FUNCTIONS[selected_builtin]
        function_source = builtin_meta["expr"]
        default_x_range = builtin_meta["x_range"]
        default_y_range = builtin_meta["y_range"]
        st.code(f"f(x, y) = {function_source}", language="python")
        st.caption(builtin_meta["description"])
    else:
        function_source = st.text_input(
            "Enter f(x, y)",
            value="x**2 + y**2",
            help="Use a scalar expression involving x, y, and jax.numpy-style functions like sin, cos, exp, log, sqrt, tanh.",
        )
        default_x_range = (-4.0, 4.0)
        default_y_range = (-4.0, 4.0)
        with st.expander("Supported syntax"):
            st.markdown(function_help_markdown())

    st.header("Start point")
    x0 = st.number_input("x₀", value=2.0, step=0.1)
    y0 = st.number_input("y₀", value=2.0, step=0.1)

    st.header("Plot domain")
    x_min, x_max = st.slider("x range", -10.0, 10.0, default_x_range)
    y_min, y_max = st.slider("y range", -10.0, 10.0, default_y_range)
    grid_size = st.slider("mesh resolution", 30, 160, 80, step=10)

    st.header("Run")
    steps = st.slider("optimization steps", 5, 400, 80, step=5)


optimizer_tabs = st.tabs(list(OPTIMIZER_SPECS.keys()))

for tab, optimizer_name in zip(optimizer_tabs, OPTIMIZER_SPECS.keys()):
    with tab:
        spec = OPTIMIZER_SPECS[optimizer_name]
        st.subheader(spec["title"])
        st.caption(spec["description"])

        param_cols = st.columns(len(spec["controls"])) if spec["controls"] else []
        current_params = {}
        for col, control in zip(param_cols, spec["controls"]):
            with col:
                current_params[control["name"]] = st.slider(
                    control["label"],
                    min_value=control["min"],
                    max_value=control["max"],
                    value=control["value"],
                    step=control["step"],
                    key=f"{optimizer_name}_{control['name']}",
                )

        try:
            result = compute_trajectory(
                optimizer_name=optimizer_name,
                function_source=function_source,
                x0=float(x0),
                y0=float(y0),
                steps=int(steps),
                params=tuple(sorted((k, float(v)) for k, v in current_params.items())),
            )
            f = compile_user_function(function_source)
        except Exception as exc:
            st.error(f"Could not evaluate the function or optimizer: {exc}")
            continue

        left, right = st.columns([2.2, 1.0])
        with left:
            fig = make_surface_figure(
                f=f,
                trajectory=result["trajectory"],
                x_range=(x_min, x_max),
                y_range=(y_min, y_max),
                grid_size=grid_size,
                title=f"{optimizer_name} trajectory on f(x, y)",
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        with right:
            st.metric("Final x", f"{result['final_point'][0]:.4f}")
            st.metric("Final y", f"{result['final_point'][1]:.4f}")
            st.metric("Final loss", f"{result['losses'][-1]:.6f}")
            st.metric("Best loss", f"{min(result['losses']):.6f}")

            st.markdown("**Latest gradient**")
            st.code(
                f"dx = {result['gradients'][-1][0]:.6f}\ndy = {result['gradients'][-1][1]:.6f}",
                language="text",
            )

            st.markdown("**What this optimizer is doing**")
            st.write(spec["intuition"])

        st.markdown("### Loss across steps")
        losses = np.asarray(result["losses"], dtype=float)
        st.line_chart(losses)

        with st.expander("Show trajectory table"):
            st.dataframe(result["trajectory"], use_container_width=True)
