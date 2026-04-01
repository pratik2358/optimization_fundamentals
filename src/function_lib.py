import ast
from typing import Callable, Dict

import jax.numpy as jnp


BUILTIN_FUNCTIONS: Dict[str, Dict] = {
    "Quadratic bowl": {
        "expr": "x**2 + y**2",
        "description": "Simple convex bowl with one global minimum at (0, 0).",
        "x_range": (-4.0, 4.0),
        "y_range": (-4.0, 4.0),
    },
    "Anisotropic bowl": {
        "expr": "10*x**2 + y**2",
        "description": "Convex, but much steeper in x than y. Good for seeing zig-zagging and adaptive step sizes.",
        "x_range": (-3.0, 3.0),
        "y_range": (-4.0, 4.0),
    },
    "Rosenbrock": {
        "expr": "(1 - x)**2 + 100*(y - x**2)**2",
        "description": "Classic narrow curved valley. Convex locally around the minimum, non-convex globally.",
        "x_range": (-2.0, 2.0),
        "y_range": (-1.0, 3.0),
    },
    "Himmelblau": {
        "expr": "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
        "description": "Non-convex surface with multiple minima.",
        "x_range": (-6.0, 6.0),
        "y_range": (-6.0, 6.0),
    },
    "Saddle": {
        "expr": "x**2 - y**2",
        "description": "Classic saddle surface. Useful for seeing instability and escaping directions.",
        "x_range": (-4.0, 4.0),
        "y_range": (-4.0, 4.0),
    },
    "Wavy basin": {
        "expr": "0.2*(x**2 + y**2) + sin(2*x)*cos(2*y)",
        "description": "Non-convex surface with local ripples on top of a shallow bowl.",
        "x_range": (-5.0, 5.0),
        "y_range": (-5.0, 5.0),
    },
}

_ALLOWED_FUNCS = {
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tan": jnp.tan,
    "exp": jnp.exp,
    "log": jnp.log,
    "sqrt": jnp.sqrt,
    "abs": jnp.abs,
    "tanh": jnp.tanh,
    "sinh": jnp.sinh,
    "cosh": jnp.cosh,
    "maximum": jnp.maximum,
    "minimum": jnp.minimum,
    "where": jnp.where,
    "pi": jnp.pi,
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Constant,
    ast.Call,
    ast.Tuple,
    ast.List,
)


def _validate_ast(node: ast.AST) -> None:
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODES):
            raise ValueError(f"Unsupported syntax: {type(child).__name__}")
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name) or child.func.id not in _ALLOWED_FUNCS:
                raise ValueError("Only selected mathematical functions are allowed.")
        if isinstance(child, ast.Name) and child.id not in {"x", "y", *list(_ALLOWED_FUNCS.keys())}:
            raise ValueError(f"Unknown symbol: {child.id}")


def compile_user_function(expr: str) -> Callable:
    parsed = ast.parse(expr, mode="eval")
    _validate_ast(parsed)
    code = compile(parsed, "<user_function>", "eval")

    def f(x, y):
        env = {"x": x, "y": y, **_ALLOWED_FUNCS}
        return eval(code, {"__builtins__": {}}, env)

    test_val = f(jnp.array(0.1), jnp.array(-0.2))
    if jnp.ndim(test_val) != 0:
        raise ValueError("The function must return a scalar value.")
    return f


def function_help_markdown() -> str:
    return """
Allowed operators: `+`, `-`, `*`, `/`, `**`, `%`

Allowed symbols: `x`, `y`

Allowed functions/constants:
- `sin`, `cos`, `tan`
- `exp`, `log`, `sqrt`, `abs`, `tanh`, `sinh`, `cosh`
- `maximum`, `minimum`, `where`
- `pi`

Examples:
- `x**2 + y**2`
- `(1 - x)**2 + 100*(y - x**2)**2`
- `0.2*(x**2 + y**2) + sin(2*x)*cos(2*y)`
""".strip()
