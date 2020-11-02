from run_nominal_control import run_nominal_control
from run_robust_control import run_robust_control
from run_tailored_robust_control import run_tailored_robust_control

from utils import *

chain_params = get_chain_params()

for n_mass in range(3, 8):
    for seed in range(1, 11):
        # adjust parameters wrt experiment
        chain_params["seed"] = seed
        chain_params["n_mass"] = n_mass

        # run all versions
        run_nominal_control(chain_params)
        run_robust_control(chain_params)
        run_tailored_robust_control(chain_params)


for n_mass in [5]:
    for seed in range(10, 51):
        # adjust parameters wrt experiment
        chain_params["seed"] = seed
        chain_params["n_mass"] = n_mass

        # run all versions
        run_nominal_control(chain_params)
        run_robust_control(chain_params)
        run_tailored_robust_control(chain_params)
