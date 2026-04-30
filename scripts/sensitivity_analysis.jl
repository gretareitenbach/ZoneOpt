using CSV, DataFrames, Flux, OrdinaryDiffEq, DiffEqFlux, SciMLSensitivity, ForwardDiff, LinearAlgebra
include("../src/neural_ode_framework.jl")
using .ZoneOptNeuralODEFramework

function sensitivity_analysis(model, seq)
    u0 = seq.u[:, 1]
    tspan = (seq.t[1], seq.t[end])
    flat_params, reconstruct = Flux.destructure(model)

    function forward_simulate(v)
        u0_new = copy(u0)
        u0_new[4:5] .= v
        rhs = (du, u, p, t) -> begin
            # need to make sure context is of the right type or handled properly! 
            # let's just do an in-place mutation but wait forward_simulate needs to be AD-friendly
            # for out of place
            du .= reconstruct(p)(u, seq.context, t)
        end
        prob = ODEProblem(rhs, u0_new, tspan, flat_params)
        sol = solve(prob, Tsit5(); saveat = [seq.t[end]], abstol = 1e-4, reltol = 1e-4) # wait, we need to return final state
        return sol.u[end]
    end

    # Jacobian of final state with respect to initial velocity u0[4:5]
    J = ForwardDiff.jacobian(forward_simulate, u0[4:5])
    return J
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

println("Running sensitivity analysis for sequence 1...")
J = sensitivity_analysis(model, test_set[1])
println("Jacobian of final state w.r.t initial velocity:")
display(J)
