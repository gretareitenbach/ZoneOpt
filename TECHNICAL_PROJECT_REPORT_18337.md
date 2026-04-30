# ZoneOpt.jl: Agentic Optimization of Ice Hockey Zone Entry Strategies via Neural ODEs
**Technical Project Report for 18.337/6.7320: Parallel Computing and Scientific Machine Learning**

## 1. Executive Summary

**ZoneOpt.jl** is a custom hardware-accelerated Julia framework that blends Scientific Machine Learning (SciML) and high-performance parallel computing to model and optimize tactical ice hockey zone entries. Formulated as a Universal Differential Equation (UDE), the framework models the continuous state of the puck $u(t)$ utilizing physical priors (structural kinematics and dampening drag) augmented by a Neural Network that discovers unmodeled dynamics (player collisions, stick deflections, ice friction variation) directly from Big Data Cup tracking data.

Operating explicitly within the purview of the 18.337/6.7320 curriculum, this project highlights key Scientific Machine Learning architectures: memory-efficient backpropagation through the ODE solver using the Continuous Adjoint Method (`InterpolatingAdjoint`), optimized Vector-Jacobian Products (VJPs) via `Zygote.jl`, execution of data-parallel `EnsembleProblem` abstractions for batching, and highly efficient sensitivity analysis for initial-condition optimization using `ForwardDiff.jl`.

## 2. Motivation and Class Context

In traditional sports analytics, predictive models treat spatial transitions—like carrying or dumping a hockey puck across a blue line—as discrete classification problems using static Random Forests or basic deep learning mechanisms. However, a hockey play is fundamentally a continuous dynamical system governed by rigid body physics intersecting with human agency.

SciML offers the ideal paradigm for this system. By treating the puck's motion as a neural-augmented ordinary differential equation, we achieve a model that:
1. Interpolates time accurately irrespective of the tracking camera's 30 FPS sampling rate.
2. Injects known physics, forcing the neural network to only learn the *residual* interactions (which accelerates training convergence drastically).
3. Provides a continuous gradient landscape, allowing a tactical AI agent to iteratively optimize initial conditions (action vectors) via gradient ascent on an empirical reward function.

The computational demands of integrating these continuous paths forward and backward across thousands of game events necessitate strict optimization of parallel workloads and algorithmic differentiation—the core focus of this implementation.

## 3. Dataset & Data Engineering

Data is consumed from the Stathletes Big Data Cup (2026), consisting of discrete event logs and high-frequency (30 FPS) spatial tracking streams of the puck and 10 skaters. 

### Data Ingestion and Context Embedding
1. **Synchronization:** Discrete play-by-play timestamps map directly to the corresponding temporal index in the Cartesian tracking space.
2. **Trajectory Generation:** 5-second sequences of continuous tracking states $u(t)$ are partitioned as ground truths. 
3. **Context Vectorization:** Since a Neural ODE requires contextual awareness of external variables, we derive a static parameter vector $p$ representing the state of the ice at $t=0$. This 23-dimensional context vector structure is injected alongside the weights $\theta$:
   - One-hot representation of categorical tactical strategy.
   - Puck-relative $x,y$ coordinate configurations of all 5 attacking teammates and 5 defending opponents.

*(Note: Prior bugs regarding absolute vs. relative coordinates during Pass strategies were patched by properly recalibrating coordinate inversions during preprocessing).*

## 4. Universal Differential Equation (UDE) Formulation

Instead of an opaque deep neural network attempting to guess coordinates, the physics-informed state transitions follow a derivative modeled by a Universal Differential Equation:

$$ \frac{d\vec{u}}{dt} = f(\vec{u},p,t) + \text{NN}(\vec{u}, C, \theta) $$

Where:
- $\vec{u}(t) \in \mathbb{R}^6$ represents $[x, y, z, v_x, v_y, v_z]$.
- $f(\vec{u},p,t)$ is the hard-coded physics prior. We mathematically assert that $v_z$ experiences $-9.8 \, m/s^2$ acceleration (gravity) and the puck experiences a baseline kinematic drag. 
- $\text{NN}(\vec{u}, C, \theta)$ parameterized by neural weights $\theta$ and context $C$, functions as the nonlinear estimator to capture what basic equations miss (e.g. puck bouncing off the boards, an opponent intercepting the trajectory).

This equation is solved numerically via `OrdinaryDiffEq.jl` leveraging the `Tsit5()` (Tsitouras 5/4 Runge-Kutta) explicit solver, chosen for its excellent efficiency profile in non-stiff configurations.

## 5. Adjoint Sensitivity Analysis and Backpropagation

A cornerstone of this 18.337 project lies in how the network weights $\theta$ are optimized. Backpropagation through time (BPTT) for a standard RNN discrete sequence scales memory linearly with the number of time steps. In an ODE setting, tracking the tape across hundreds of intermediate solver steps is overwhelmingly inefficient.

Instead, **ZoneOpt.jl** invokes the continuous Adjoint Method via the `SciMLSensitivity.jl` library. Through Pontryagin’s Maximum Principle, an augmented state containing the adjoint vector $\vec{a}(t) = \frac{\partial L}{\partial \vec{u}(t)}$ is defined and solved *backwards* in time:

$$ \frac{d\vec{a}}{dt} = -\vec{a}(t)^\top \frac{\partial f_{UDE}}{\partial \vec{u}} $$

To obtain the gradient of the loss with respect to continuous parameters $\theta$:
$$ \frac{dL}{d\theta} = -\int_{t_1}^{t_0} \vec{a}(t)^\top \frac{\partial f_{UDE}}{\partial \theta} dt $$

### Algorithmic Differentiation Choices:
We specifically utilized `InterpolatingAdjoint(autojacvec=ZygoteVJP())`. This allows the solver to build a continuous spline of the forward pass, avoiding complete resolution of the forward ODE during the backward pass, balancing compute and memory perfectly ($O(1)$ memory backprop). Reverse-mode automatic differentiation (`Zygote.jl`) manages the Vector-Jacobian Products locally for the Neural Network terms seamlessly.

## 6. Parallelization and Computational Architecture

Training over high-frequency spatial tracking data demands strict parallel computing considerations:
- **Data Parallelism (`EnsembleProblem`):** The loss evaluation requires solving the UDE across multiple discrete puck entry scenarios sequentially. By wrapping the ODE within an `EnsembleProblem(prob, EnsembleThreads())` (or `EnsembleDistributed`), we farm the trajectory simulations mapping the batch loss across available Julia CPU threads. This perfectly aligns with multi-core processor usage efficiently circumventing Julia's global boundaries where memory allocations stay isolated to specific thread heaps.
- **Batched Losses:** The batch objective function compares the interpolated sequence $u(t)$ points to the empirical tracker using heavily weighted position metrics ($L_2$ position error $\times 1.0$) and secondary velocity metrics ($\times 0.25$).

Over the course of 210+ epochs (e.g. up to checkpoint `epoch_210_best.jls`), multi-threading successfully condensed raw computing times spanning thousands of parameter evaluations.

## 7. Agentic Optimization Layer 

With the SciML proxy environment established, an AI coach algorithm executes continuous tactical policy evaluations. Given a fixed defensive context vector $C_{def}$, the agent tests multiple strategies (different initial vectors $u_0$).

### Performance Duality: ForwardDiff vs. Zygote
1. When training millions of Neural net parameters ($\theta$), reverse-mode AD (`Zygote` + Adjoint) is mathematically optimal computationally.
2. During agentic optimization, the system seeks to find the best immediate trajectory. We optimize the initial physical velocity of the puck $v_{x, 0}, v_{y, 0}$ for a single scenario. Because the input dimension of $u_0$ is extraordinarily small (6 scalar values), reverse-mode AD presents unnecessary overhead. As explored in class, **Forward-Mode Differentiation (`ForwardDiff.jl`)** utilizing Dual Numbers is mathematically superior here. The model maps Jacobians pushing forward derivations of $u_0$ simultaneously with primitive physics floats to discover the exact vector producing the maximum custom reward surface (targeting depth $x \approx 50$, precision $y \approx 10$).

## 8. Results, Training Dynamics, and Evaluation

- **Physics-Informed Accuracy:** The inclusion of kinematic priors resulted in the Neural ODE demonstrating exceptionally realistic energy dissipation. Unrealistic paths (e.g., accelerating pucks without player contact) were inherently squashed by the physics block, rendering highly plausible continuous mappings.
- **Optimization Stability:** The training loss decreased significantly over the 210 recorded epochs without exploding gradients, a common issue in raw discrete networks attempting physical systems. 
- **Decision Validity Baseline:** Beyond pure coordinate RMSE, we gauged model utility using tactical validity. Applying the trained environment algorithmized reward to historically successful zone entries (instances generating a real-world shot), the simulated optimal path correlated remarkably with the recorded real-world tactical decision. The recommendation output far superseded naive randomized baselines (~33%).

## 9. Limitations and Future Work

Drawing from concepts elucidated in 18.337/6.7320, we acknowledge the following expansion zones:
- **Local Minima inside Reward Surfaces:** The optimization of local velocity vectors faces highly non-convex topography (due to defending player coordinate geometry). Incorporating simulated annealing or stochastic global optimization techniques before descending with `ForwardDiff` may yield stronger global minimum strategies.
- **Defensive Velocity Derivatives:** The contextual state relies purely on the $t=0$ absolute positions. Providing the ODE access to defensive velocity approximations (`x'` of the players) as secondary inputs will strengthen the neural net's prediction of intercepts.
- **GPU Deployment:** The workflow is primarily executing across standard compute frameworks. Given `DiffEqGPU.jl`, future iterations will aggressively map `EnsembleProblems` onto CUDA acceleration for hyper-scalar batching.

## 10. Conclusion

**ZoneOpt.jl** bridges the gap between mechanical data-fitting and physics-engine simulation. By engineering a framework around a Universal Differential Equation optimized with class-centric tools (`Zygote.jl`, `SciMLSensitivity.jl`, `ForwardDiff.jl`), the project successfully establishes a differential agent environment capable of resolving the complexities of sports analytics. Combining multi-threaded trajectory ensembles with adjoint-enhanced backward passes produces an AI system capable of rendering deterministic, optimized coaching decisions rooted directly in modeled reality.