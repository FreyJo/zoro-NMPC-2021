from run_nominal_control import run_nominal_control
from run_robust_control import run_robust_control
from run_tailored_robust_control import run_tailored_robust_control
from run_fastzoro_robust_control import run_fastzoro_robust_control

from utils import get_chain_params

chain_params = get_chain_params()

for n_mass in range(3, 7):
    for seed in range(1, 5):
        # adjust parameters wrt experiment
        chain_params["seed"] = seed
        chain_params["n_mass"] = n_mass
        chain_params["save_results"] = True

        # run all versions
        run_nominal_control(chain_params)
        run_fastzoro_robust_control(chain_params, zoro_riccati=-1)
        run_fastzoro_robust_control(chain_params, zoro_riccati=0)
        run_fastzoro_robust_control(chain_params, zoro_riccati=1)
        run_fastzoro_robust_control(chain_params, zoro_riccati=2)
        # run_robust_control(chain_params)
        # run_tailored_robust_control(chain_params)


# for n_mass in [5]:
#     for seed in range(10, 51):
#         # adjust parameters wrt experiment
#         chain_params["seed"] = seed
#         chain_params["n_mass"] = n_mass

#         # run all versions
#         run_nominal_control(chain_params)
#         run_robust_control(chain_params)
#         run_tailored_robust_control(chain_params)
