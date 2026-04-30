# 18.337/6.7320 Project Proposal

**Author:** Greta Reitenbach (gretar@mit.edu)

**Project Title:** ZoneOpt.jl: Agentic Optimization of Ice Hockey Zone Entry Strategies via Neural ODEs

---

## 1. Project Summary

This project proposes the development of **ZoneOpt.jl**, a Julia-based framework that couples a Neural Ordinary Differential Equation (Neural ODE) with an agentic optimization layer to evaluate and recommend ice hockey zone entry strategies. Using high-frequency tracking data from the Stathletes Big Data Cup, the project will model the puck's movement as a physics-informed dynamical system. The framework will use an AI agent to iteratively simulate different scenarios. It will compare carried, dumped, and passed zone entries to identify the strategic choice that maximizes offensive success.

---

## 2. Motivation

In modern hockey analytics, the transition from the neutral zone to the offensive zone is a critical inflection point. While carrying the puck into the zone is statistically linked to higher shot volumes, the high risk of turnovers at the blue line often leads teams to dump the puck ("shoot" it into the far boards behind the net), conceding possession for a safer defensive posture and time for line changes. Classical statistical models often fail to capture the continuous, high-speed physics of these movements or the spatial influence of defensive formations.

By leveraging Scientific Machine Learning, we can move beyond discrete event analysis. A Neural ODE can learn the underlying physics of the ice, including puck friction, air drag, and player interference, directly from the detailed tracking data. This allows for the creation of a model where an AI coach can test strategies against a learned representation of reality.

---

## 3. Mathematical Foundation: Neural ODEs

Following the methodology established in the course, the puck's state `u(t) = [x,y,z,v_x,v_y,v_z]` will be modeled as a differential equation. Unlike a standard ODE with fixed physical constants, the derivative will be defined by a neural network:

$$\frac{du}{dt} = f(u,p,t) + \text{NN}(u,\theta)$$

Where:
- $f(u,p,t)$ represents known physical terms, such as gravity affecting the z-coordinate
- $\text{NN}(u,\theta)$ is a neural network that learns the unmodeled interactions of the puck with the ice and players

The model will be trained using the Adjoint Method to efficiently compute gradients with respect to the network parameters $\theta$.

---

## 4. Methodology & Workflow

### Phase 1: Preprocessing

The preprocessing pipeline will integrate two disparate datasets:

- **Event Filtering:** Extract all Zone Entry events from the Events.csv file, noting the Clock time, Period, and entry type (Carried, Dumped, or Played).
- **Tracking Synchronization:** For each identified entry, locate the corresponding Game Clock in the Tracking.csv files to extract the 3D coordinates (x,y,z) of the puck and the 2D coordinates (x,y) of the 10 skaters on the ice.
- **Sequence Generation:** Generate 5-second trajectory windows following each entry to serve as training labels for the SciML model.

### Phase 2: Training the Neural ODE

- **Forward Pass:** The system will be integrated using `OrdinaryDiffEq.jl` with the `Tsit5()` integrator.
- **Cost Function:** The loss will be defined as the L2 distance (Euclidean distance) between the predicted puck trajectory and the actual tracked coordinates from the Stathletes data.
- **Optimization:** Using the adjoint sensitivity analysis from the homework, the model will update the weights of the neural network to minimize the trajectory error across diverse entry types.

### Phase 3: Agentic Optimization

An AI Tactical Agent will be implemented to optimize the entry decision:

- **Scenario Simulation:** Given a defensive formation at the blue line, the agent will run the Neural ODE forward under three different initial conditions ($u_0$): a high-velocity dump, a player-controlled carry, and a pass.
- **Reward Maximization:** The agent will select the strategy that maximizes a success metric, defined as the probability of the puck reaching a high-danger shot location within the offensive zone.

---

## 5. Julia Ecosystem Integration

The project will leverage the following high-performance Julia packages:

- **DiffEqFlux.jl:** To implement the Neural ODE and the training loop.
- **ModelingToolkit.jl:** To provide a symbolic representation of the puck's dynamics.
- **ForwardDiff.jl:** To validate the gradients calculated by the adjoint method.
- **Makie.jl:** To create 3D visualizations of the rink, player positioning, and predicted puck probabilities for each entry strategy.

---

## 6. Evaluation and Accuracy

Accuracy will be measured using a two-tier approach:

- **Trajectory Accuracy:** Measured by the Root Mean Square Error (RMSE) between simulated and tracked puck positions over a 5-second horizon.
- **Decision Validity:** We will compare the agent's recommendations against historical successful entries (those resulting in a shot). A "correct" recommendation is one where the agent's preferred strategy aligns with the successful real-world outcome.

---

## 7. References

- Stathletes Big Data Cup 2026: Publicly available hockey tracking and event dataset.
- Rackauckas, C. et al. "Universal Differential Equations for Scientific Machine Learning." arXiv, 2020.
