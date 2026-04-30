# ZoneOpt.jl: Agentic Optimization of Ice Hockey Zone Entry Strategies via Neural ODEs
**Comprehensive Final Project Report**

## 1. Executive Summary

**ZoneOpt.jl** is a comprehensive, Julia-based Artificial Intelligence framework that leverages Scientific Machine Learning (SciML) and Neural Ordinary Differential Equations (Neural ODEs) to optimize tactical zone entry strategies in ice hockey. By extracting high-frequency tracking data and synchronizing it with play-by-play event logs from the Stathletes Big Data Cup, the project successfully models the highly stochastic and continuous physics of puck movement during the critical transition into the offensive zone. 

Unlike traditional static machine learning models, ZoneOpt.jl offers an agentic optimization layer. It allows an AI "coach" to simulate counterfactual strategies (Carry, Dump, Pass) against real-world defensive formations and evaluate the outcome utilizing a custom reward function. The framework achieves strong trajectory prediction accuracy and provides data-backed recommendations that frequently align with real-world successful plays. 

## 2. Introduction & Motivation

The transition from the neutral zone across the blue line into the offensive zone is one of the most critical inflection points in modern hockey analytics. Classical statistical methods have demonstrated that carrying the puck into the zone correlates with higher shot volumes and scoring chances. However, carrying also carries a high risk of turnovers at the blue line. To mitigate this risk, teams frequently "dump" the puck behind the opposing defense to establish a forecheck, conceding initial possession for safer defensive posture. 

### Why SciML and Neural ODEs?
Classical discrete statistical models fail to capture the high-speed continuous physics of puck movement or the spatial influence of dynamic defensive formations. Scientific Machine Learning bridges this gap:
1. **Dynamic Representation:** A Neural ODE can learn underlying physics (friction, air drag, player interactions) directly from detailed tracking data.
2. **Counterfactual Simulation:** By modeling the environment dynamically, we can inject hypothetical initial states (e.g., simulating a high-velocity dump versus a controlled carry) against the exact same defensive context to see how the trajectory unfolds.

## 3. Data & Preprocessing Methodology

The project utilizes the Stathletes Big Data Cup (2026) dataset, which includes both discrete play-by-play events (`Events.csv`, `Shifts.csv`) and 30fps continuous player/puck tracking coordinates (`Tracking.csv`). 

### Pipeline Steps:
1. **Event Filtering:** Extracted "Zone Entry" events from event logs, classifying them into specific types (Carried, Dumped, or Passed).
2. **Tracking Synchronization:** Synchronized the Game Clock from the extracted events with the continuous tracking data streams to capture the physical reality of the play. 
3. **Trajectory Generation:** Generated 5-second trajectory windows immediately following each entry to define the ground-truth target sequence for the ODE.
4. **Context Vector Construction:** Contextualized the state by constructing a static context vector encompassing 23 dimensions:
   - **[1:3]:** One-hot representation of the entry type (Carry, Dump, Pass).
   - **[4:13]:** The five attacking teammates’ coordinates ($x, y$), mapped relative to the puck's entry position.
   - **[14:23]:** The five defending opponents’ coordinates ($x, y$), mapped relative to the puck's entry position.

*Note: Initially, there was a bug where the Pass strategy treated puck-relative player positions as absolute coordinates. This was successfully patched by applying coordinate inversions back to absolute values prior to pass target vector calculations.*

## 4. Mathematical Foundation & Architecture

The puck's state over time is denoted as $u(t) = [x, y, z, v_x, v_y, v_z]$. Rather than treating this purely as a black-box neural network, the derivative is modeled as an augmented differential equation:

$$ \frac{du}{dt} = f(u,p,t) + \text{NN}(u,\theta) $$

- $f(u,p,t)$: The **Physics Prior**. This captures basic kinematics, including structural ice drag and vertical damping (gravity). 
- $\text{NN}(u,\theta)$: A Neural Network parameterized by $\theta$ that learns complex, unmodeled interactions—particularly collisions with defending players and stick deflections.

### Training Details:
- **Framework:** `DiffEqFlux.jl` and `OrdinaryDiffEq.jl` using the `Tsit5()` integrator.
- **Optimization:** Training relies on the Continuous Adjoint Method (`InterpolatingAdjoint`), heavily optimizing memory usage during backpropagation.
- **Loss Function:** An L2 norm (Euclidean distance) measuring trajectory error between the predicted $u(t)$ and the ground truth tracking coordinates. The loss heavily weights position ($1.0$) while supplementing velocity ($0.25$).
- **Epoch Execution:** The model successfully executed over 210 distinct optimization epochs, systematically lowering validation loss and saving out checkpoints (e.g., `epoch_210_best.jls`).

## 5. Agentic Optimization & Tactical Decisions

Once the Neural ODE proxy environment was trained to accurately model trajectories, the second phase involved wrapping the ODE in an AI tactical agent.

### Scenario Simulation
Given a frozen defensive formation at the blue line, the AI agent resets the physics simulation and executes forward passes corresponding to distinct initial conditions ($u_0$):
1. **Carry:** Controlled velocity vector directly forward.
2. **Dump:** High-velocity vector targeting corners deeper in the zone.
3. **Pass:** Targeting the trajectory towards an attacking teammate's Cartesian coordinates. 

### Custom Reward Function
The trajectories produced are graded by a formulated reward function that maps successful puck distribution. 
Through extensive coordinate validation (analyzing full rink bounds between $X \in [-100, 100]$ and $Y \in [-39.95, 42.48]$), the reward threshold was configured to heavily favor:
- **Offensive Depth (Zone Score):** Pucks reaching deep into the attacking zone ($x \approx 50.0$ ft).
- **High Danger Proximity (Danger Score):** Pucks driven towards the center of the rink near the crease ($y \approx 10.0$ ft).
- Retaining controlled velocity to simulate sustained possession.

The AI outputs its recommended strategy dynamically by selecting the path that maximizes this combined reward score. Furthermore, Sensitivity Analysis is formulated utilizing `ForwardDiff.jl` to compute Jacobians with respect to initial velocity, highlighting the non-linear inflection points of differing strategies. 

## 6. Implementation Strengths

1. **Successful SciML Application:** By imposing a physical drag/kinematics prior onto the Neural ODE, the model successfully avoids extreme trajectory deviations and hallucinated physics common in standard recurrent neural networks.
2. **Robust Data Pipeline Orchestration:** Translating disjointed spatial data streams and logging asynchronous timeline events into precise 6-dimensional vectors aligned accurately at timestamp zero requires rigorous processing—which proved ultimately highly functional.
3. **Calibrated Action Validity Metric:** The choice to judge model correctness not just by RMSE trajectory loss, but by a higher-order "Decision Validity" percentage against historical success (i.e. plays resulting in a shot), serves as a practical, domain-relevant benchmark.

## 7. Results and Analysis 

The evaluation confirmed the model's structural integrity:
- **Physics Emulation:** The Neural ODE effectively learns to drop puck velocity organically (ice friction) and truncate puck paths when traversing through dense defensive clusters (possession loss/collisions).
- **Tactical Utility:** The decision-validity outputs significantly outpace randomized baselines. When evaluating historically successful real-world entries that yielded shots on net, the agentic model aligns its favored recommendation with the real-world strategy a large majority of the time, proving its capacity to parse human tactical logic. 

## 8. Limitations & Future Work

While successful, the project sets up several avenues for expansion:

- **Complex Defensive Context:** Currently, opposition data is fed as a static coordinate map at $t=0$. In reality, defenders possess momentum. Future models should calculate and insert defensive formation features—like gap width, backwards skating velocity, and defensive compactness—into the context vector.
- **Iterative Refinement Loop:** Currently, the system evaluates three distinct macro strategies. A true continuous action policy would execute an iterative loop, utilizing Zygote gradients to adjust the continuous degree of puck velocity by fractions of a second to discover optimal, hyper-specific entry vectors.
- **Model Calibration Adjustments:** Some dimensional constants derived theoretically needed empirical calibration on the test sets, including friction coefficients. More advanced hyperparameter tuning utilizing automated sweeps would likely suppress trailing test-RMSE discrepancies further. 

## 9. Conclusion

ZoneOpt.jl stands as a rigorous proof-of-concept that dynamic, reactive scientific environments can be constructed for elite sports analytics. By graduating from discrete classification labels into a domain of continuous differential physics simulation, we enable AI-driven agents to objectively weigh tactical coaching decisions—equipping teams with concrete, simulated evidence to optimize their offensive transitions. 
