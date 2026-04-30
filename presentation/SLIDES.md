---
marp: true
theme: default
---

# ZoneOpt.jl 
## Agentic Optimization of Ice Hockey Zone Entry Strategies via Neural ODEs
**By:** Greta Reitenbach

---

## 1. The Problem: Zone Entries
- **Context:** Transitioning into the offensive zone is a critical inflection point.
- **Carry vs. Dump:** Carrying the puck leads to more shots but risks turnovers. Dumping is safer but concedes possession.
- **Goal:** Build an AI system to evaluate these choices contextually and optimize the initial entry parameters.

---

## 2. Methodology: Neural ODEs
- **Data:** Stathletes Tracking & Event Data (30fps 3D puck, 2D skaters).
- **Physics-Informed Modeling:** Predict puck state $[x,y,z,v_x,v_y,v_z]$ forward 5 seconds.
- **Context:** Defensive spacing relative to the puck at the blue line.
- **Tools:** `DiffEqFlux.jl`, `OrdinaryDiffEq.jl`.

---

## 3. Strategy Evaluation
- The agent simulates trajectories for Carry, Dump, Pass, and Shoot using varying initial conditions $u_0$.
- **Reward Function:** 
  - Maximize offensive depth ($x \approx 50$).
  - Minimize distance to center ($y \approx 0$).
  - Control speed near the net.

---

## 4. Agentic Refinement & Sensitivity
- **Gradient Optimization:** Using `ForwardDiff` and `Zygote`.
- The agent refines initial entry velocity to maximize the future reward.
- Shows tactical awareness responding to changing defensive features.

---

## 5. Conclusion
- Neural ODEs prove to be a viable simulator for multi-agent spatial sports data.
- The framework supports programmatic coaching and tactical adjustments.
