using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean
using BenchmarkTools: @btime
using Statistics

##############
# Instructions
##############
#=

This starter code is here to show examples of how to use the HW3 code that you
can copy and paste into your homework code if you wish. It is not meant to be a
fill-in-the blank skeleton code, so the structure of your final submission may
differ from this considerably.

Please make sure to update DMUStudent to gain access to the HW3 module.

=#

############
# Question 2
############

m = HW3.DenseGridWorld(seed=3)

As = actions(m)
# isterminal(m, s)
γ = discount(m)
S_type = statetype(m)
A_type = actiontype(m)


function rollout(mdp, policy_function, s0, max_steps=100)
    # fill this in with code from the assignment document
    r_total = 0.0 
    t=0 
    s = s0

    while !isterminal(mdp, s) && t < max_steps
        a = policy_function(mdp, s)
        s, r = @gen(:sp,:r)(mdp, s, a) 
        r_total += discount(m)^t*r 
        t += 1
    end

    return r_total
end

function random_policy(m, s)
    # put a smarter heuristic policy here
    return rand(actions(m))
end

function heuristic_policy(m, s)
    # find the nearest target from (20, 40, 60)
    utb, ltb = 60, 20
    target_x = max(ltb, min(((s[1] + 10) ÷ 20) * 20, utb))
    target_y = max(ltb, min(((s[2] + 10) ÷ 20) * 20, utb))

    if s[1] != target_x
        # moves L/R first
        return s[1] < target_x ? :right : :left
        # then U/D
    elseif s[2] != target_y
        return s[2] < target_y ? :up : :down
    else
        return rand(actions(m))
    end
end

# This code runs monte carlo simulations: you can calculate the mean and standard error from the results
results = [rollout(m, random_policy, rand(initialstate(m))) for _ in 1:500]
# @show mean_reward = mean(results)
# @show SEM = std(results) / sqrt(length(results))

# This code runs monte carlo simulations: you can calculate the mean and standard error from the results
results = [rollout(m, heuristic_policy, rand(initialstate(m))) for _ in 1:500]
@show mean_results = mean(results)
@show SEM = std(results) / sqrt(length(results))


