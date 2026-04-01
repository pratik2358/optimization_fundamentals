from typing import Callable, Dict, List

import jax
import jax.numpy as jnp
import numpy as np


def _to_record(step: int, point, loss, grad):
    return {
        "step": int(step),
        "x": float(point[0]),
        "y": float(point[1]),
        "loss": float(loss),
        "grad_x": float(grad[0]),
        "grad_y": float(grad[1]),
    }


OPTIMIZER_SPECS = {
    "Gradient Descent": {
        "title": "Gradient Descent",
        "description": "Takes a full step along the negative gradient at every iteration.",
        "intuition": "This is the clean baseline: always move in the steepest downhill direction using the current gradient.",
        "controls": [
            {"name": "lr", "label": "learning rate", "min": 0.001, "max": 1.0, "value": 0.05, "step": 0.001},
        ],
    },
    "Momentum": {
        "title": "SGD with Momentum",
        "description": "Accumulates a velocity vector to damp oscillations and build speed in consistent directions.",
        "intuition": "Momentum remembers the recent past, so it keeps moving through shallow valleys instead of reacting only to the latest gradient.",
        "controls": [
            {"name": "lr", "label": "learning rate", "min": 0.001, "max": 1.0, "value": 0.03, "step": 0.001},
            {"name": "beta", "label": "momentum β", "min": 0.0, "max": 0.999, "value": 0.9, "step": 0.001},
        ],
    },
    "RMSprop": {
        "title": "RMSprop",
        "description": "Rescales each coordinate by a running average of recent squared gradients.",
        "intuition": "Big-gradient directions get smaller effective steps, which reduces zig-zagging on steep axes.",
        "controls": [
            {"name": "lr", "label": "learning rate", "min": 0.0001, "max": 0.5, "value": 0.02, "step": 0.0001},
            {"name": "beta2", "label": "decay β₂", "min": 0.0, "max": 0.999, "value": 0.9, "step": 0.001},
            {"name": "eps", "label": "epsilon", "min": 1e-08, "max": 1e-03, "value": 1e-08, "step": 1e-08},
        ],
    },
    "Adam": {
        "title": "Adam",
        "description": "Combines momentum with adaptive per-coordinate scaling.",
        "intuition": "Adam smooths the gradient and rescales directions automatically, so it often feels stable and easy to tune.",
        "controls": [
            {"name": "lr", "label": "learning rate", "min": 0.0001, "max": 0.5, "value": 0.03, "step": 0.0001},
            {"name": "beta1", "label": "β₁", "min": 0.0, "max": 0.999, "value": 0.9, "step": 0.001},
            {"name": "beta2", "label": "β₂", "min": 0.0, "max": 0.9999, "value": 0.999, "step": 0.0001},
            {"name": "eps", "label": "epsilon", "min": 1e-08, "max": 1e-03, "value": 1e-08, "step": 1e-08},
        ],
    },
    "AdamW": {
        "title": "AdamW",
        "description": "Adam plus decoupled weight decay.",
        "intuition": "AdamW behaves like Adam, but also shrinks parameters directly, making regularization cleaner and easier to interpret.",
        "controls": [
            {"name": "lr", "label": "learning rate", "min": 0.0001, "max": 0.5, "value": 0.03, "step": 0.0001},
            {"name": "beta1", "label": "β₁", "min": 0.0, "max": 0.999, "value": 0.9, "step": 0.001},
            {"name": "beta2", "label": "β₂", "min": 0.0, "max": 0.9999, "value": 0.999, "step": 0.0001},
            {"name": "eps", "label": "epsilon", "min": 1e-08, "max": 1e-03, "value": 1e-08, "step": 1e-08},
            {"name": "weight_decay", "label": "weight decay", "min": 0.0, "max": 0.5, "value": 0.01, "step": 0.001},
        ],
    },
}


def run_optimizer(
    optimizer_name: str,
    f: Callable,
    start: jnp.ndarray,
    steps: int,
    params: Dict[str, float],
):
    value_and_grad = jax.value_and_grad(lambda p: f(p[0], p[1]))
    point = jnp.array(start, dtype=jnp.float32)

    trajectory: List[Dict] = []
    gradients: List[List[float]] = []
    losses: List[float] = []

    v = jnp.zeros_like(point)
    m = jnp.zeros_like(point)
    s = jnp.zeros_like(point)

    loss, grad = value_and_grad(point)
    trajectory.append(_to_record(0, point, loss, grad))
    gradients.append([float(grad[0]), float(grad[1])])
    losses.append(float(loss))

    for t in range(1, steps + 1):
        loss, grad = value_and_grad(point)
        lr = params.get("lr", 0.05)

        if optimizer_name == "Gradient Descent":
            point = point - lr * grad

        elif optimizer_name == "Momentum":
            beta = params.get("beta", 0.9)
            v = beta * v - lr * grad
            point = point + v

        elif optimizer_name == "RMSprop":
            beta2 = params.get("beta2", 0.9)
            eps = params.get("eps", 1e-8)
            s = beta2 * s + (1.0 - beta2) * (grad ** 2)
            point = point - lr * grad / (jnp.sqrt(s) + eps)

        elif optimizer_name == "Adam":
            beta1 = params.get("beta1", 0.9)
            beta2 = params.get("beta2", 0.999)
            eps = params.get("eps", 1e-8)
            m = beta1 * m + (1.0 - beta1) * grad
            s = beta2 * s + (1.0 - beta2) * (grad ** 2)
            m_hat = m / (1.0 - beta1 ** t)
            s_hat = s / (1.0 - beta2 ** t)
            point = point - lr * m_hat / (jnp.sqrt(s_hat) + eps)

        elif optimizer_name == "AdamW":
            beta1 = params.get("beta1", 0.9)
            beta2 = params.get("beta2", 0.999)
            eps = params.get("eps", 1e-8)
            weight_decay = params.get("weight_decay", 0.01)
            m = beta1 * m + (1.0 - beta1) * grad
            s = beta2 * s + (1.0 - beta2) * (grad ** 2)
            m_hat = m / (1.0 - beta1 ** t)
            s_hat = s / (1.0 - beta2 ** t)
            point = point - lr * m_hat / (jnp.sqrt(s_hat) + eps)
            point = point - lr * weight_decay * point

        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        new_loss, new_grad = value_and_grad(point)
        trajectory.append(_to_record(t, point, new_loss, new_grad))
        gradients.append([float(new_grad[0]), float(new_grad[1])])
        losses.append(float(new_loss))

    return {
        "trajectory": trajectory,
        "gradients": gradients,
        "losses": losses,
        "final_point": [float(point[0]), float(point[1])],
    }
