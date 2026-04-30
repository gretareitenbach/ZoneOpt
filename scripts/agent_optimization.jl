using CSV, DataFrames, Flux, OrdinaryDiffEq, DiffEqFlux, SciMLSensitivity, Zygote, LinearAlgebra
include("../src/neural_ode_framework.jl")
using .ZoneOptNeuralODEFramework

function optimize_initial_state(model, seq; lr=0.01, steps=50)
    u0 = ZoneOptNeuralODEFramework.strategy_initial_state(seq, :carry)
    # We only optimize the initial velocity (v_x, v_y), which are u0[4:5]
    v0_opt = copy(u0[4:5])
    tspan = (seq.t[1], seq.t[end])
    flat_params, reconstruct = Flux.destructure(model)

    for i in 1:steps
        loss, grad = Zygote.withgradient(v0_opt) do v
            u0_new = copy(u0)
            u0_new[4:5] .= v
            rhs = (du, u, p, t) -> begin
                du .= reconstruct(p)(u, seq.context, t)
            end
            prob = ODEProblem(rhs, u0_new, tspan, flat_params)
            sol = solve(prob, Tsit5(); saveat = seq.t, abstol = 1e-6, reltol = 1e-6, sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()))
            pred = Array(sol)
            # maximize reward -> minimize negative reward
            return -ZoneOptNeuralODEFramework.strategy_reward(pred, seq)
        end
        v0_opt .-= lr .* grad[1]
    end
    return v0_opt
end

long_csv = joinpath("processed","neural_ode_sequences_long.csv")
index_csv = joinpath("processed","neural_ode_sequence_index.csv")
long_df, index_df = ZoneOptNeuralODEFramework.load_sequence_tables(long_csv, index_csv)
examples = ZoneOptNeuralODEFramework.build_sequence_examples(long_df, index_df)

train_set, val_set, test_set = ZoneOptNeuralODEFramework.split_sequences(examples; seed=42)
ctx_len = length(first(examples).context)
model = ZoneOptNeuralODEFramework.build_model(ctx_len, train_set; hidden_dim=64, depth=3)

checkpoint = "models/epoch_140.jls"
ZoneOptNeuralODEFramework.restore_model_checkpoint!(model, checkpoint)

println("Optimizing strategy for sequence 1...")
v0_opt = optimize_initial_state(model, test_set[1])
println("Optimized initial velocity: ", v0_opt)
