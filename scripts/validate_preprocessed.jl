using CSV, DataFrames, Statistics

function summarize_df(path)
    println("\nReading: ", path)
    df = CSV.read(path, DataFrame)
    println("Rows: ", size(df, 1), "  Columns: ", size(df, 2))

    println("Column types:")
    for (n, col) in zip(names(df), eachcol(df))
        println("  ", n, ": ", eltype(col))
    end

    # Missing / NaN counts for numeric columns
    total_rows = size(df,1)
    println("Missing/NaN counts:")
    for n in names(df)
        col = df[!, n]
        miss = count(ismissing, col)
        nan = 0
        if eltype(col) <: AbstractFloat
            nan = count(x -> !ismissing(x) && isnan(x), col)
        end
        if miss>0 || nan>0
            println("  ", n, ": missing=", miss, ", NaN=", nan)
        end
    end

    # Print example rows
    nshow = min(3, total_rows)
    if nshow>0
        println("First $(nshow) rows:")
        show(first(df, nshow))
        println()
    end
end

function main()
    base = normpath(joinpath(dirname(@__FILE__), ".."))
    p1 = joinpath(base, "processed", "neural_ode_sequence_index.csv")
    p2 = joinpath(base, "processed", "neural_ode_sequences_long.csv")

    for p in (p1, p2)
        if isfile(p)
            summarize_df(p)
        else
            println("File not found: ", p)
        end
    end
end

main()
