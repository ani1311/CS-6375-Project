import TD0_cartpole as td0cp
import TD0_gridworld as td0gw
import monte_carlo_cartpole as mccp
import monte_carlo_gridworld as mcgw
import TDN_gridworld as tdngw
import TDN_cartpole as tdncp

print("Running all experiments")

#  print("Running monte carlo on gridworld")
#  mcgw.train()
#
#  print("Running monte carlo on cartpole")
#  mccp.train()

print("Running TD0 on gridworld")
td0gw.train()

print("Running TD0 on cartpole")
td0cp.train()

print("Running TDN on gridworld")
tdngw.train()

print("Running TDN on cartpole")
tdncp.train()
