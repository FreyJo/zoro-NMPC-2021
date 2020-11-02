from utils import *
import numpy as np
from plot_utils import timings_plot

chain_params = get_chain_params()

IDs = ["nominal", "zoRO", "robust"]
Seeds = range(1,6)

for n_mass in range(3,8):

    timings = {id:[] for id in IDs}

    # load results
    for id in IDs:
        for seed in Seeds:
            chain_params["seed"] = seed
            chain_params["n_mass"] = n_mass
            results = load_results_from_json(id, chain_params)
            total_timing = np.array(results["timings"]) + np.array(results["timings_P"])
            timings[id] = timings[id] + list(total_timing)

    timings_plot(timings, n_mass)
