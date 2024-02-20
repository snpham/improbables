
using DMUStudent.HW2
using POMDPs: states, actions
using POMDPTools: ordered_states
using PlutoUI
using Profile

    
##############
# Instructions
##############
#=

This starter code is here to show examples of how to use the HW2 code that you
can copy and paste into your homework code if you wish. It is not meant to be a
fill-in-the blank skeleton code, so the structure of your final submission may
differ from this considerably.

=#

############
# Question 3
############
m = grid_world
begin
	@show m.size
	@show m.rewards
	@show m.terminate_from
	@show m.tprob
	@show m.discount
end

begin
	# Return the actions that can be taken from state s.
	@show actions(grid_world)
	
	# Create a dictionary mapping actions to transition matrices for MDP m.
	T = transition_matrices(grid_world)

	# probability of transitioning between states with indices 1 and 2 when taking action :left
	@show T[:left][1, 2] 
	
	# dictionary for reward vectors for each action
	R = reward_vectors(grid_world)
	
	# the reward for taking action :right in the state with index 1
	@show R[:right][1] 

end

function value_iteration(m, γ)

	# setup
	T = transition_matrices(m, sparse=true)
	R = reward_vectors(m)
    V = rand(length(states(m)))
	@show A = collect(actions(m))
	@show n = length(states(m))
	@show γ = γ
	@show θ = 1e-6
	v = rand(length(states(m)))

	while maximum(abs, V-v) > 0.000
		v[:] = V
		V[:] = max.((R[a] + γ*T[a]*V for a in keys(R))...)
	end

    return V
end


V = value_iteration(m, 0.95)
# You can use the following commented code to display the value. If you are in an environment with multimedia capability (e.g. Jupyter, Pluto, VSCode, Juno), you can display the environment with the following commented code. From the REPL, you can use the ElectronDisplay package.
# render(m, color=V)


############
# Question 4
############
begin
	using Pkg
	using DMUStudent.HW2
	using POMDPs: states, actions
	using POMDPTools: ordered_states
	using PlutoUI
	using POMDPs: states, stateindex, convert_s
	using SparseArrays
end

# You can create an mdp object representing the problem with the following:
m = UnresponsiveACASMDP(2)

# transition_matrices and reward_vectors work the same as for grid_world, however this problem is much larger, so you will have to exploit the structure of the problem. In particular, you may find the docstring of transition_matrices helpful:
display(@doc(transition_matrices))

V = value_iteration(m, 0.99)

# HW2.evaluate(V, "son.pham-2@colorado.edu")

# ########
# # Extras
# ########

# # The comments below are not needed for the homework, but may be helpful for interpreting the problems or getting a high score on the leaderboard.

# # Both UnresponsiveACASMDP and grid_world implement the POMDPs.jl interface. You can find complete documentation here: https://juliapomdp.github.io/POMDPs.jl/stable/api/#Model-Functions

# # To convert from physical states to indices in the transition function, use the stateindex function
# # IMPORTANT NOTE: YOU ONLY NEED TO USE STATE INDICES FOR THIS ASSIGNMENT, using the states may help you make faster specialized code for the ACAS problem, but it is not required
# using POMDPs: states, stateindex

# s = first(states(m))
# @show si = stateindex(m, s)

# # To convert from a state index to a physical state in the ACAS MDP, use convert_s:
# using POMDPs: convert_s

# @show s = convert_s(ACASState, si, m)

# # To visualize a state in the ACAS MDP, use
# render(m, (s=s,))
