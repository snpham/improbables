using Random
using QuadGK
using Distributions
using KernelDensity
using PlotlyJS
using StatsBase
using Roots
using LinearAlgebra
using SpecialFunctions


function random_dice_roll7(n::Int)
    """
    Conduct a Monte Carlo experiment to estimate the probability of getting 
    a sum equal to seven when rolling two fair dice.

    Parameters:
    - n: Int, the number of trials to perform.

    Returns:
    Float64, the estimated probability of the sum being 7.
    """
    return sum(rand(1:6, n) .+ rand(1:6, n) .== 7) / n
end


function probability_of_dice_sum(dice_sum::Int)
    """
    Computes the exact probability of obtaining a specific sum when rolling 
    two fair dice.

    Parameters:
    - dice_sum: Int, the target sum for which to compute the probability.

    Returns:
    Float64, the exact probability of achieving the target sum.
    """
    total_outcomes = 6 * 6
    target_outcomes = [(i, j) for i in 1:6 for j in 1:6 if i + j == dice_sum]
    return length(target_outcomes) / total_outcomes
end


function cheby()
    # Define the Chebyshev PDF function
    chebyshev_pdf(x) = 1 / (π * sqrt(1 - x^2))

    # Define the Chebyshev CDF function
    chebyshev_cdf(x) = 0.5 + (1/π) * asin(x)

    # Define the inverse CDF function for Chebyshev distribution
    inverse_chebyshev_cdf(u) = sin(π * (u - 0.5))

    # Generate a range of x values for plotting the exact PDF and CDF
    x = range(-0.99, stop=0.99, length=1000)

    # Compute the exact PDF and CDF values for the x range
    pdf_vals = chebyshev_pdf.(x)
    cdf_vals = chebyshev_cdf.(x)

    # Generate random samples using the inverse CDF method
    samples = rand(10000)
    chebyshev_samples = inverse_chebyshev_cdf.(samples)

    # Estimate the empirical PDF using kernel density estimation
    kde_estimation = kde(chebyshev_samples)

    # Estimate the empirical CDF
    ecdf_estimation = ecdf(chebyshev_samples)

    # set up subplots
    traces = make_subplots(rows=1, cols=2,
        subplot_titles=["Probability Density Function" "Cumulative Distribution Function"])

    # plot the exact and empirical density functions
    add_trace!(traces, scatter(x=x, y=pdf_vals, name="Exact Density"), row=1, col=1)
    add_trace!(traces, scatter(x=kde_estimation.x, y=kde_estimation.density, 
        name="Empirical Density"))

    # plot the exact and empirical CDF
    add_trace!(traces, scatter(x=x, y=cdf_vals, name="Exact Distribution"), row=1, col=2)
    add_trace!(traces, scatter(x=x, y=ecdf_estimation.(x), name="Empirical Distribution"), row=1, col=2)
    traces.plot.layout["width"] = 900
    traces.plot.layout["height"] = 400

    # save figure
    relayout!(traces, title_text="PDF and CDF of the Chebyshev PDF")
    savefig(traces, "outputs/cheby_pdf_cdf.png", height = 400, width=900)
    display(traces)
end


if abspath(PROGRAM_FILE) == @__FILE__

    # Set of trials
    n_set = [50, 100, 1000, 10000, 100000]
    foreach(n -> println("$n samples: $(random_dice_roll7(n))"), n_set)

    cheby()

end