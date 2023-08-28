from utils import *
import numpy as np
from plot_utils import constraint_violation_box_plot

chain_params = get_chain_params()

IDs = ["nominal", "fastzoRO", "zoRO", "robust"]
Seeds = range(1,30)
n_mass = 3

dist_dict = {id:[] for id in IDs}

# load results
for id in IDs:
    for seed in Seeds:
        chain_params["seed"] = seed
        chain_params["n_mass"] = n_mass
        results = load_results_from_json(id, chain_params)
        dist_dict[id] = dist_dict[id] + [min(results["wall_dist"])]

# plot
constraint_violation_box_plot(dist_dict, n_mass)

