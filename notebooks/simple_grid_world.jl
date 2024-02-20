### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 2c00ae36-c624-11ee-3d63-d1be178683d2
begin
	using Pkg
	Pkg.add(url="https://github.com/zsunberg/DMUStudent.jl");
	using DMUStudent.HW2
	using POMDPs: states, actions
	using POMDPTools: ordered_states
	using PlutoUI
end

# ╔═╡ f5f81b7c-810d-4591-a105-6b5047f4ed52
# size           :: Tuple{Int64, Int64}
# rewards        :: Dict{StaticArraysCore.SVector{2, Int64}, Float64}
# terminate_from :: Set{StaticArraysCore.SVector{2, Int64}}
# tprob          :: Float64
# discount       :: Float64
m = grid_world

# ╔═╡ 5c701aac-d23c-4232-baec-dacce54f78df
begin
	@show m.size
	@show m.rewards
	@show m.terminate_from
	@show m.tprob
	@show m.discount
end

# ╔═╡ 87361b93-4171-49eb-ae58-1b6533ccc93d
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

	@show states(m)
	@show actions(m)

end

# ╔═╡ e1030317-54a9-46e0-8b09-a502888ff757
function value_iteration_original(m)

	# setup
	T = transition_matrices(m)
	R = reward_vectors(m)
    V = rand(length(states(m)))
	@show A = collect(actions(m))
	@show n = length(states(m))
	@show γ = discount(m)
	@show θ = 1e-6

	while true
		# initialize delta for this iteration
	    δ = 0.0
		
		# iterate over all states
        for s in 1:n
			# store the current value of the state
            v = V[s]  
            u_sp_list = []
			
			# iterate over all actions
            for a in A
                u_sp = 0.0
				
				# sum over all possible next states
                for sp in 1:n
                    # calculate value function for s->sp
					# T[a][s,sp] = transition probability from s->sp after action a
                    # R[a][s] = reward for action a in state s
                    u_sp += T[a][s,sp] * (R[a][s] + γ*V[sp])
                end
					
                # collect the total for each action
                push!(u_sp_list, u_sp)
            end

            # update the value function with the max value found for all actions
            V[s] = maximum(u_sp_list)

            # update δ
            δ = max(δ, abs(v - V[s]))
        end

        # check for convergence
        if δ < θ
            break
        end
    end

					
    return V
end

# ╔═╡ 36816262-aa38-42e4-9134-64276c8a11ec
function value_iteration(m)

	# setup
	T = transition_matrices(m, sparse=true)
	R = reward_vectors(m)
    V = rand(length(states(m)))
	@show A = collect(actions(m))
	@show n = length(states(m))
	@show γ = m.discount
	@show θ = 1e-6
	v = rand(length(states(m)))

	while maximum(abs, V-v) > 0.000
		v[:] = V
		V[:] = max.((R[a] + γ*T[a]*V for a in keys(R))...)
	end

    return V
end

# ╔═╡ ec5e606c-4b5c-437b-8fba-4a1f74e197fb
V = value_iteration(m)

# ╔═╡ 0c24d5a0-b726-4d8d-91cf-73923ad8d98d
render(m)

# ╔═╡ d0b49f1f-bf30-4b58-af59-dadadedddf1a
render(m, color=V)

# ╔═╡ 5a9c73c0-078e-4373-9d8d-7a14faea4765


# ╔═╡ a3669afc-61bc-4478-a9d4-13e99cbbab60


# ╔═╡ 4b59970f-cc2b-4cc5-8aa9-1f5345bbc848


# ╔═╡ a0def3d0-1b1c-4a59-b082-df9aa9cdddca


# ╔═╡ c24343df-9507-4e7d-8d36-856fac6931a5


# ╔═╡ 083ce55e-307c-4230-9e0d-26395684ca63


# ╔═╡ Cell order:
# ╠═2c00ae36-c624-11ee-3d63-d1be178683d2
# ╠═f5f81b7c-810d-4591-a105-6b5047f4ed52
# ╠═5c701aac-d23c-4232-baec-dacce54f78df
# ╠═87361b93-4171-49eb-ae58-1b6533ccc93d
# ╟─e1030317-54a9-46e0-8b09-a502888ff757
# ╠═36816262-aa38-42e4-9134-64276c8a11ec
# ╠═ec5e606c-4b5c-437b-8fba-4a1f74e197fb
# ╠═0c24d5a0-b726-4d8d-91cf-73923ad8d98d
# ╠═d0b49f1f-bf30-4b58-af59-dadadedddf1a
# ╠═5a9c73c0-078e-4373-9d8d-7a14faea4765
# ╠═a3669afc-61bc-4478-a9d4-13e99cbbab60
# ╠═4b59970f-cc2b-4cc5-8aa9-1f5345bbc848
# ╠═a0def3d0-1b1c-4a59-b082-df9aa9cdddca
# ╠═c24343df-9507-4e7d-8d36-856fac6931a5
# ╠═083ce55e-307c-4230-9e0d-26395684ca63
