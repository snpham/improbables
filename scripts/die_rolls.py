import numpy as np

gamma = 0.95
T_roll = np.array([[0, 0, 1], 
                   [0, 0, 1], 
                   [0, 0, 1]])
T_reset = np.array([[1, 0, 0], 
                    [1, 0, 0], 
                    [1, 0, 0]])
R_reset = np.array([0, 2, -1])
V = np.zeros(3)

for _ in range(1000):
    V_new = np.zeros(3)
    for s in range(3):
        V_roll = np.sum(T_roll[s] * (0 + gamma * V))
        V_reset = R_reset[s] + gamma * V[0]
        V_new[s] = max(V_roll, V_reset)
    
    if np.max(np.abs(V - V_new)) < 1e-4:
        break
    V = V_new

print(V)
