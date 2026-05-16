"""
author：CTBUXuBJ
date：2025-1-5
"""
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpMinimize, GLPK, PULP_CBC_CMD
import numpy as np
import pulp
import pandas
import matplotlib
# print(pulp.__version__)
# print(pandas.__version__)
# print(matplotlib.__version__)

m = 6
# Variable Definitions
wprob = LpProblem(name="WeightOptimization", sense=LpMinimize)
w = [LpVariable(name=f"w_{j}", lowBound=0, upBound=1) for j in range(m)]
MMSV = LpVariable(name="MMSV", lowBound=0)
# print(w)
# Constraint Definitions
for j in range(m):

    wprob += w[0] - 5 * w[1] <= MMSV
    wprob += -w[0] + 5 * w[1] <= MMSV

    wprob += w[0] - 3 * w[2] <= MMSV
    wprob += -w[0] + 3 * w[2] <= MMSV

    wprob += w[0] - 2 * w[3] <= MMSV
    wprob += -w[0] + 2 * w[3] <= MMSV

    wprob += w[0] - 4 * w[4] <= MMSV
    wprob += -w[0] + 4 * w[4] <= MMSV

    wprob += w[0] - 2 * w[5] <= MMSV
    wprob += -w[0] + 2 * w[5] <= MMSV

    wprob += w[0] - 3 * w[1] <= MMSV
    wprob += -w[0] + 3 * w[1] <= MMSV

    wprob += w[2] - 2 * w[1] <= MMSV
    wprob += -w[2] + 2 * w[1] <= MMSV

    wprob += w[3] - 4 * w[1] <= MMSV
    wprob += -w[3] + 4 * w[1] <= MMSV

    wprob += w[4] - 3 * w[1] <= MMSV
    wprob += -w[4] + 3 * w[1] <= MMSV

    wprob += w[5] - 2 * w[1] <= MMSV
    wprob += -w[5] + 2 * w[1] <= MMSV

wprob += lpSum(w) == 1  # Sum of weights equal to 1

# Objective Function
wprob += MMSV

# Solving mixed integer linear programming model
wprob.solve(pulp.GLPK_CMD())

# Displaying the results
print("Status:", wprob.status)
print("Final Weights:")
for v in w:
    print(f"{v.name}: {v.value()}")
print(MMSV.value())
