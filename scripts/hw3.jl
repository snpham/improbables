using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean
using BenchmarkTools: @btime, @benchmark
using Statistics
using StaticArrays


############################################################
# Question 2
############################################################

function rollout_q2(mdp, policy_function, s, max_steps=100)
    r_total = 0.0 
    t = 0 
    while !isterminal(mdp, s) && t < max_steps
        a = policy_function(mdp, s)
        s, r = @gen(:sp,:r)(mdp, s, a) 
        r_total += discount(m)^t*r 
        t += 1
    end
    return r_total
end

function random_policy(m, s)
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
println("# Q2a) Monte Carlo Evaluation - Random Policy")
results = [rollout_q2(m, random_policy, rand(initialstate(m))) for _ in 1:400]
@show mean_discounted_reward_estimate = mean(results)
@show standard_error_mean = std(results) / sqrt(length(results))
println("# Q2b) Monte Carlo Evaluation - Heuristic Policy")
results = [rollout_q2(m, heuristic_policy, rand(initialstate(m))) for _ in 1:400]
@show mean_discounted_reward_estimate = mean(results)
@show standard_error_mean = std(results) / sqrt(length(results))


############################################################
# Question 3
############################################################

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

    t[(s, a, sp)] = get(t, (s, a, sp), 0) + 1

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
    mdp_discount = discount(mdp)
    for depth in 1:d
        if isterminal(mdp, s)
            break
        end
        
        a = π₀(mdp, s)
        s, r = @gen(:sp, :r)(mdp, s, a)
        r_total += discount_factor * r
        discount_factor *= mdp_discount
    end
    return r_total
end

function explore(mdp, s, Q, N)
    c = sqrt(2)
    A = actions(mdp, s)
    bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)
    # total visits
    Ns = sum(N[(s,a)] for a in A)
    return argmax(a -> Q[(s,a)] + c * bonus(N[(s,a)], Ns), A)
end

# running q3
m = DenseGridWorld(seed=4)
n = Dict{Tuple{statetype(m), actiontype(m)}, Int}()
q = Dict{Tuple{statetype(m), actiontype(m)}, Float64}()
T = Set{statetype(m)}()
t = Dict{Tuple{statetype(m), actiontype(m), statetype(m)}, Int}()
s = SA[19, 19]
max_depth = 50
max_iters = 7
π₀ = heuristic_policy
for i in 1:max_iters
    select_action(m, s, max_depth, π₀, q, n, T, t)
end
inchrome(visualize_tree(q, n, t, s))


############################################################
# Question 4
############################################################

# A starting point for the MCTS select_action function which can be used for Questions 4 and 5
function select_action_q4(mdp, s)

    start = time_ns()
    n = Dict{Tuple{statetype(m), actiontype(m)}, Int}()
    q = Dict{Tuple{statetype(m), actiontype(m)}, Float64}()
    T = Set{statetype(m)}()
    t = Dict{Tuple{statetype(m), actiontype(m), statetype(m)}, Int}()

    max_depth = 50
    π₀ = heuristic_policy

    # for _ in 1:1000
    while time_ns() < start + 40_000_000 # you can replace the above line with this if you want to limit this loop to run within 40ms
        simulate!(mdp, s, max_depth, π₀, q, n, T, t)
    end

    # select a good action based on q and/or n
    return explore(mdp, s, q, n)
end

m4 = DenseGridWorld(seed=4)
@btime select_action_q4(m, SA[35,35]) # you can use this to see how much time your function takes to run. A good time is 10-20ms.


function evaluate_mcts_planner(mdp, num_simulations=100, steps_per_simulation=100)
    total_rewards = Float64[]

    for _ in 1:num_simulations
        s = SA[35,35]
        r_total = 0.0
        
        for _ in 1:steps_per_simulation
            if isterminal(mdp, s)
                break
            end
            
            a = select_action_q4(mdp, s)
            s, r = @gen(:sp, :r)(mdp, s, a)
            r_total += r 
        end
        
        push!(total_rewards, r_total)
    end
    
    mean_reward = mean(total_rewards)
    sem = std(total_rewards) / sqrt(length(total_rewards))  # Standard Error of the Mean
    
    println("# Q4) Planning with MCTS")
    println("mean_discounted_reward_estimate: $mean_reward")
    println("SEM: $sem")
end

m4 = DenseGridWorld(seed=4)
evaluate_mcts_planner(m4)




# ############
# # Question 5
# ############
m4 = DenseGridWorld(seed=4)
HW3.evaluate(select_action_q4, "son.pham-2@colorado.edu", time=true)

# # If you want to see roughly what's in the evaluate function (with the timing code removed), check sanitized_evaluate.jl
