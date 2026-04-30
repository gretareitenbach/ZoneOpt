using CSV, DataFrames, Plots, Statistics

repo_root = normpath(joinpath(dirname(@__FILE__), ".."))
eval_dir = joinpath(repo_root, "evaluation")
out_dir = joinpath(repo_root, "presentation", "figures")
mkpath(out_dir)

# RMSE by entry type
rmse_df = CSV.read(joinpath(eval_dir, "test_rmse_by_entry_type_quick.csv"), DataFrame)
barplot = bar(rmse_df.entry_type, rmse_df.rmse, legend=false, ylabel="RMSE (ft)", title="RMSE by Entry Type", color=:steelblue)
savefig(barplot, joinpath(out_dir, "rmse_by_entry_type.png"))

# Calibration heatmap
cal_df = CSV.read(joinpath(eval_dir, "physics_calibration_grid.csv"), DataFrame)
drags = sort(unique(cal_df.drag))
verts = sort(unique(cal_df.vert))
Z = [first(cal_df[(cal_df.drag .== d) .& (cal_df.vert .== v), :rmse]) for v in verts, d in drags]
heat = heatmap(drags, verts, Z, xlabel="drag", ylabel="vertical_damping", title="Physics Prior Calibration RMSE", color=:viridis)
savefig(heat, joinpath(out_dir, "calibration_heatmap.png"))

# Summary metrics as a small annotated figure
summary_df = CSV.read(joinpath(eval_dir, "quick_evaluation_summary.csv"), DataFrame)
txt = join([string(row.metric, ": ", round(row.value, digits=4)) for row in eachrow(summary_df)], "\n")
plt = plot(legend=false, framestyle=:box, xticks=false, yticks=false)
annotate!(0.1, 0.9, text("Evaluation Summary", 14, :black))
annotate!(0.1, 0.6, text(txt, 10, :black))
savefig(plt, joinpath(out_dir, "evaluation_summary.png"))

# Copy example sequence plots into presentation folder
for f in readdir(joinpath(eval_dir, "figures"))
    src = joinpath(eval_dir, "figures", f)
    dst = joinpath(out_dir, f)
    cp(src, dst; force=true)
end

println("Report figures generated in: ", out_dir)
