using DMUStudent.HW1

function get_distributions(joint_dist)

    # a: what is the probability of the outcome A=0,B=0,C=0
    sum_prob = round(sum(joint_dist[:, end]), digits=6)
    p_000 = 1 - sum_prob
    joint_dist = vcat(joint_dist, [0 0 0 p_000])

    # b: what is the marginal distribution of A?
    unique_vals = unique(joint_dist[:,1])
    marginal_A = zeros(length(unique_vals), 2)
    for (i, val) in enumerate(unique_vals)
        marginal_A[i, 1] = val
        marginal_A[i, 2] = sum(joint_dist[joint_dist[:,1] .== val, end])
    end

    # c: what is the conditional distribution of A given B=0 and C=1
    # P(A|B,C) = P(A, B=0, C=1)/P(B=0, C=1)
    # P(A=0|B=0,C=1) = P(A=0, B=0, C=1)/P(B=0, C=1)
    row_A0 = joint_dist[:,1] .== 0 .&& joint_dist[:,2] .== 0 .&& joint_dist[:,3] .== 1
    row_A1 = joint_dist[:,1] .== 1 .&& joint_dist[:,2] .== 0 .&& joint_dist[:,3] .== 1
    cond_dist_A0 = joint_dist[row_A0, end] / (joint_dist[row_A0, end]+joint_dist[row_A1, end])
    return marginal_A, cond_dist_A0
end


function get_maxv(a::Array{<:Number,2}, bs)
    ab = [a * b for b in bs]
    maxv = [maximum(getindex.(ab, i)) for i in 1:size(a, 1)]
    return maxv
end

# A B C  P(A,B,C)
joint_dist = [0 0 1    0.15;
0 1 0    0.05;
0 1 1    0.01;
1 0 0    0.14;
1 0 1    0.18;
1 1 0    0.29;
1 1 1    0.06]
marginal_A, cond_dist_A0 = get_distributions(joint_dist)

maxv = get_maxv(a, bs)

HW1.evaluate(q5, "son.pham-2@colorado.edu")
