from utils import *
import numpy as np
from plot_utils import timings_plot_vary_mass, num_nlp_iters_plot

chain_params = get_chain_params()

IDs = ["nominal", "fastzoRO-fixedK", "fastzoRO-riccatiFixedQuad", "fastzoRO-riccatiHessianV1", "fastzoRO-riccatiHessianV2"]
Seeds = range(1,5)
N_masses = range(3,7)

# mass_dict = {nm: [] for nm in N_masses}
timings = {id:dict() for id in IDs}
num_nlp_iters = {id:dict() for id in IDs}

# load results
for id in IDs:
    for n_mass in N_masses:
        timings[id][n_mass] = []
        for seed in Seeds:
            chain_params["seed"] = seed
            chain_params["n_mass"] = n_mass
            results = load_results_from_json(id, chain_params)
            total_timing = np.array(results["timings"]) + np.array(results["timings_P"])
            timings[id][n_mass] = timings[id][n_mass] + list(total_timing)
            num_nlp_iters[id][n_mass] = results["num_nlp_iter"]

# plot
timings_plot_vary_mass(timings, N_masses)
num_nlp_iters_plot(num_nlp_iters, N_masses)
