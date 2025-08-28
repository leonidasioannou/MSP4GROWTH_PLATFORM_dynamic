import gurobipy as gp

print("Gurobi version:", gp.gurobi.version())

env = gp.Env(empty=True)
env.setParam('WLSAccessID', 'dc02d694-311e-42d6-a955-d94ff28d385a')
env.setParam('WLSSecret', '716a7c85-6942-4238-b278-cc4bc656497e')
env.setParam('LicenseID', 2509155)
env.setParam('ThreadLimit', 12)
env.setParam('Threads', 12)
env.start()

model = gp.Model(env=env)
x = model.addVar(name="x")
model.setObjective(x, gp.GRB.MAXIMIZE)
model.optimize()

print("Optimization completed with status:", model.Status)
