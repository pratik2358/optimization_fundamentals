# Optimizer Explorer

A Streamlit demo for visualizing how common optimization algorithms move across 2D objective functions.

## Features

- Separate tabs for Gradient Descent, Momentum, RMSprop, Adam, and AdamW
- Built-in convex and non-convex test functions
- Safe custom scalar functions of `x` and `y`
- Tunable hyperparameters and starting point
- Interactive 3D color-coded surface with the descent path overlaid
- Loss curve and trajectory table

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit app and point it to `app.py`.
3. Make sure `requirements.txt` is in the repo root.

## Built-in functions

- Quadratic bowl
- Anisotropic bowl
- Rosenbrock
- Himmelblau
- Saddle
- Wavy basin

## Custom functions

You can enter expressions like:

```python
x**2 + y**2
(1 - x)**2 + 100*(y - x**2)**2
0.2*(x**2 + y**2) + sin(2*x)*cos(2*y)
```

Allowed functions include `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `tanh`, `sinh`, `cosh`, `maximum`, `minimum`, and `where`.
