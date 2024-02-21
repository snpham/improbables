using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean
using BenchmarkTools: @btime, @benchmark
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

function rollout_q2(mdp, policy_function, s0, max_steps=100)
    # fill this in with code from the assignment document
    r_total = 0.0 
    t = 0 
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

m = HW3.DenseGridWorld(seed=3)
# This code runs monte carlo simulations: you can calculate the mean and standard error from the results
results = [rollout_q2(m, random_policy, rand(initialstate(m))) for _ in 1:500]
@show mean_reward = mean(results)
@show SEM = std(results) / sqrt(length(results))

m = HW3.DenseGridWorld(seed=3)
# This code runs monte carlo simulations: you can calculate the mean and standard error from the results
results = [rollout_q2(m, heuristic_policy, rand(initialstate(m))) for _ in 1:500]
@show mean_results = mean(results)
@show SEM = std(results) / sqrt(length(results))







############
# Question 3
############

function select_action(mdp, s, d, π₀, Q, N, T, t)
    simulate!(mdp, s, d, π₀, Q, N, T, t)
    return explore(mdp, s, Q, N)
end

function simulate!(mdp, s, d, π₀, Q, N, T, t)
    if d == 0 || isterminal(mdp, s)
        return 0
    end

    if s ∉ T
        for a in actions(mdp)
            N[(s, a)] = 0
            Q[(s, a)] = 0.0
        end
        push!(T, s)
        return rollout_loop(mdp, s, d, π₀)
    end

    a = explore(mdp, s, Q, N)
    sp, r = @gen(:sp,:r)(mdp, s, a)

    t_key = (s, a, sp)
    t[t_key] = get(t, t_key, 0) + 1

    q = r + discount(m) * simulate!(mdp, sp, d - 1, π₀, Q, N, T, t)
    N[(s, a)] += 1
    Q[(s, a)] += (q - Q[(s, a)]) / N[(s, a)]
    return q
end

function rollout_recursive(mdp, s, d, π₀)
    if d == 0 || isterminal(mdp, s)
        return 0
    end
    a = π₀(mdp, s)
    sp, r = @gen(:sp,:r)(mdp, s, a) 
    return r + discount(m) * rollout_recursive(mdp, sp, d - 1, π₀)
end

function rollout_loop(mdp, s, d, π₀)
    r_total = 0.0
    discount_factor = 1.0
    current_state = s
    
    for current_depth in 1:d
        if isterminal(mdp, current_state)
            break
        end
        
        a = π₀(mdp, current_state)
        sp, r = @gen(:sp, :r)(mdp, current_state, a)
        
        r_total += discount_factor * r
        discount_factor *= discount(mdp)
        current_state = sp
    end
    
    return r_total
end

function explore(mdp, s, Q, N)
    c = sqrt(2)
    As = actions(mdp, s)
    
    if all(a -> (s, a) ∉ keys(N), As)
        return rand(As)
    end

    # total visits
    Ns = sum([N[(s, a)] for a in As if (s, a) in keys(N)])

    ucb_values = [(a, ((s, a) in keys(N) && N[(s, a)] > 0) ? 
        Q[(s, a)] / N[(s, a)] + c * sqrt(2 * log(Ns) / N[(s, a)]) : Inf) for a in As]

    return maximum(ucb_values)[1]
end

# running q3
m = DenseGridWorld(seed=4)
n = Dict{Tuple{statetype(m), actiontype(m)}, Int}()
q = Dict{Tuple{statetype(m), actiontype(m)}, Float64}()
T = Set{statetype(m)}()
t = Dict{Tuple{statetype(m), actiontype(m), statetype(m)}, Int}()

# Example usage:
s0 = SA[19, 19]
max_depth = 7
max_iters = 7
π₀ = heuristic_policy
    for i in 1:max_iters
        select_action(m, s0, max_depth, π₀, q, n, T, t)
    end
inchrome(visualize_tree(q, n, t, s0))

