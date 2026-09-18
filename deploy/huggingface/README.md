---
title: RoverDevKit
emoji: 🛰️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# RoverDevKit — hosted demo

Interactive tradespace explorer for conceptual design of lunar micro-rovers:
physics-based mission evaluator, calibrated surrogate predictions, parametric
sweeps, NSGA-II multi-objective optimization, and SHAP-style design
explanations.

- Source code: <https://github.com/Coordinated-Autonomous-Systems-Lab/roverdevkit>
- Paper preprint: <https://arxiv.org/abs/2606.21755>

This Space runs the single-container build from
[`webapp/Dockerfile`](https://github.com/Coordinated-Autonomous-Systems-Lab/roverdevkit/blob/main/webapp/Dockerfile):
one `uvicorn` process serves the FastAPI backend and the React single-page app
from the same origin on port 8000.

> This README (with its Spaces front matter) is generated for the hosted demo
> by `scripts/deploy_hf_space.sh` and is not the repository's main README.
