# PINN-PC-1D  
**Physics-Informed Neural Network for 1D Heat Transfer with Phase Change**

> An engineering-focused implementation of a Physics-Informed Neural Network (PINN) for modelling transient heat transfer with phase change, aimed at surrogate modelling and manufacturing quality prediction.

---

## Why This Project Exists

In manufacturing processes such as **casting, welding, and metal additive manufacturing**, engineers repeatedly solve heat-transfer problems for:
- parameter studies
- quality prediction
- optimisation loops
- digital twins

Classical solvers are accurate but **computationally expensive** when embedded in decision-making pipelines.

This project evaluates **PINNs as physics-aware surrogate models**, with explicit attention to **where they work, where they fail, and how they compare to classical approaches**.

---

## What This Repository Demonstrates

- End-to-end **physics → model → evaluation** workflow
- Explicit enforcement of governing equations via autograd
- Separation of **modeling, data, metrics, and post-processing**
- Engineering judgement around **trade-offs, stability, and failure modes**

This is **not a benchmark-chasing repo** — it is an engineering evaluation.

---

## Core Physics

We solve the transient 1D heat equation:

The governing physics is the 1D transient heat equation:

dT/dt = α · d²T/dx²

where:
- T(x,t) is temperature
- α is thermal diffusivity
with extensions relevant to **phase-change-driven solidification**.  
Physics constraints are enforced directly in the loss function.

---

## Repository Structure

```text
pinn-pc-1d/
├── pinn/                 # PINN architecture and loss definitions
├── Data/                 # Sampling and dataset generation
├── gpr/                  # Baseline surrogate (Gaussian Process)
├── niyama-calculator/    # Quality/solidification metrics
├── postprocessing/       # Visualisation and diagnostics
├── utils/                # Shared utilities
├── main.py               # Training entry point
