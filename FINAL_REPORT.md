# ZoneOpt.jl: Agentic Optimization of Ice Hockey Zone Entry Strategies via Neural ODEs

## 1. Introduction & Motivation
In modern hockey analytics, transitioning from the neutral zone to the offensive zone is a critical inflection point. While carrying the puck into the zone is statistically linked to higher shot volumes, the high risk of turnovers at the blue line often leads teams to "dump" the puck in. Traditional models struggle to capture the continuous, high-speed physics of these movements or the spatial influence of defensive formations. 

**ZoneOpt.jl** bridges this gap using Scientific Machine Learning (SciML). By coupling a Neural ODE with an agentic optimization layer, we predict puck trajectories under different strategies (carry, dump, pass, shoot) and evaluate which method optimal maximizes offensive output given defensive constraints.

## 2. Methodology

### 2.1 Preprocessing and Context Construction
Using the Stathletes Big Data Cup dataset, we parse event data alongside 30fps tracking coordinate streams. We define the context vector using the positions of both attacking and defending players relative to the puck at the moment of zone entry to feed into the model as situational context.

### 2.2 Neural ODE Architecture
We model the puck's state \(u(t) = [x, y, z, v_x, v_y, v_z]\) using a neural-augmented ordinary differential equation. We trained the model on 5-second puck movement trajectories using the Adjoint method with `DiffEqFlux.jl`. The loss function penalizes L2 trajectory error (puck Cartesian distances). 

### 2.3 Agentic Optimization & Decisions
Using our fully trained proxy environment, the agent simulates multiple counterfactual initial inputs (representing carried vs. dumped puck velocities) and runs the ODE forward. The **reward function** heavily weights offensive zone depth (\(x \approx 50\) ft) and proximity to the net center (\(y \approx 0\) ft) with controlled velocity. Further strategy refinement operates through Continuous Action Policy Gradient or basic trajectory optimization using `ForwardDiff` gradients of the reward surface.

## 3. Results & Evaluation 
The model demonstrates an ability to discern tactical outcomes. The learned physics prior captures realistic puck paths (e.g. friction dropping velocity) and collision with defensive formations resulting in loss of puck possession. 
When scored against historically successful entries ending in a shot, the model's recommended strategy matches real-world winning plays a significant portion of the time. 

## 4. Conclusion
ZoneOpt.jl successfully demonstrates that Neural ODEs can function as dynamic, reactive environments for sports analytics agents. Rather than classifying inputs in a static feature space, we simulate the complex, continuous physics of ice hockey, optimizing for tactical intelligence.
