using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean
using BenchmarkTools: @btime, @benchmark
using Statistics
using StaticArrays


# ############
# # Question 5
# ############

function heuristic_policy_q5(m, s)
    # find the nearest target from (20, 40, 60)
    utb, ltb = 100, 20
    # target_x = max(ltb, min(((s[1] + 10) ÷ 20) * 20, utb))
    # target_y = max(ltb, min(((s[2] + 10) ÷ 20) * 20, utb))
    target_x = round(s[1] / 20) * 20
    target_y = round(s[2] / 20) * 20
    target_x = clamp(target_x, 20, 100)
    target_y = clamp(target_y, 20, 100)
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

HW3.evaluate(heuristic_policy_q5, "son.pham-2@colorado.edu")

# The provided heuristic policy was the policy chosen for question 2 with the minor
# modification of adding the "clamp" function instead of manually computing the bounds.
# For this policy, no tuning parameters or rollouts were available due to this being
# a heuristic policy without a MCTS. The policy computes the reward cells, then determines
# what direction to move to get closer to that reward cell. If it has reached the 
# cell, then it randomly exits and searches for the cell again.