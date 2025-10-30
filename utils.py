import os
import json
import numpy as np
import casadi as ca
from export_chain_mass_model import export_chain_mass_model


def get_chain_params():
    params = dict()

    params["n_mass"] = 5
    params["Ts"] = 0.2
    params["Tsim"] = 5
    params["N"] = 40
    params["u_init"] = np.array([-1, 1, 1])
    params["with_wall"] = True
    params["yPosWall"] = -0.05 # Dimitris: - 0.1;
    params["m"] = 0.033 # mass of the balls
    params["D"] = 1.0 # spring constant
    params["L"] = 0.033 # rest length of spring
    params["perturb_scale"] = 1e-2

    params["save_results"] = True
    params["show_plots"] = False
    params["nlp_iter"] = 50
    params["seed"] = 50
    params["nlp_tol"] = 1e-6

    return params



def compute_steady_state(n_mass, m, D, L, xPosFirstMass, xEndRef):

    model = export_chain_mass_model(n_mass, m, D, L)
    nx = model.x.shape[0]
    M = int((nx/3 -1)/2)

    # initial guess for state
    pos0_x = np.linspace(xPosFirstMass[0], xEndRef[0], n_mass)
    x0 = np.zeros((nx, 1))
    x0[:3*(M+1):3] = pos0_x[1:].reshape((M+1,1))

    # decision variables
    w = [model.x, model.xdot, model.u]
    # initial guess
    w0 = ca.vertcat(*[x0, np.zeros(model.xdot.shape), np.zeros(model.u.shape)])

    # constraints
    g = []
    g += [model.f_impl_expr]                        # steady state
    g += [model.x[3*M:3*(M+1)]  - xEndRef]          # fix position of last mass
    g += [model.u]                                  # don't actuate controlled mass

    # misuse IPOPT as nonlinear equation solver
    nlp = {'x': ca.vertcat(*w), 'f': 0, 'g': ca.vertcat(*g)}

    solver = ca.nlpsol('solver', 'ipopt', nlp)
    sol = solver(x0=w0,lbg=0,ubg=0)

    wrest = sol['x'].full()
    xrest = wrest[:nx]

    return xrest


def sampleFromEllipsoid(w, Z):
    """
    draws uniform(?) sample from ellipsoid with center w and variability matrix Z
    """

    n = w.shape[0]                  # dimension
    lam, v = np.linalg.eig(Z)

    # sample in hypersphere
    r = np.random.rand()**(1/n)     # radial position of sample
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    x *= r
    # project to ellipsoid
    y = v @ (np.sqrt(lam) * x) + w

    return y


def sym_mat2vec(mat):
    nx = mat.shape[0]

    if isinstance(mat, np.ndarray):
        vec = np.zeros((int((nx+1)*nx/2),))
    else:
        vec = ca.SX.zeros(int((nx+1)*nx/2))

    start_mat = 0
    for i in range(nx):
        end_mat = start_mat + (nx - i)
        vec[start_mat:end_mat] = mat[i:,i]
        start_mat += (nx-i)

    return vec


def vec2sym_mat(vec, nx):
    # nx = (vec.shape[0])

    if isinstance(vec, np.ndarray):
        mat = np.zeros((nx,nx))
    else:
        mat = ca.SX.zeros(nx,nx)

    start_mat = 0
    for i in range(nx):
        end_mat = start_mat + (nx - i)
        aux = vec[start_mat:end_mat]
        mat[i,i:] = aux.T
        mat[i:,i] = aux
        start_mat += (nx-i)

    return mat


def is_pos_def(mat):
    try:
        np.linalg.cholesky(mat)
        return 1
    except np.linalg.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in err.args[0]:
            return 0
        else:
            raise


def P_propagation(P, A, B, W):
    #  P_i+1 = A P A^T +  B*W*B^T
    return A @ P @ A.T + B @ W @ B.T


def np_array_to_list(np_array):
    if isinstance(np_array, (np.ndarray)):
        return np_array.tolist()
    elif isinstance(np_array, (SX)):
        return DM(np_array).full()
    elif isinstance(np_array, (DM)):
        return np_array.full()
    else:
        raise(Exception(
            "Cannot convert to list type {}".format(type(np_array))
        ))


def save_results_as_json(result_dict, json_file):

    with open(json_file, 'w') as f:
        json.dump(result_dict, f, default=np_array_to_list, indent=4, sort_keys=True)

    return


def save_closed_loop_results_as_json(ID, timings, timings_P, wall_dist, num_nlp_iter, step_nlp_iter, chain_params):

    result_dict = dict()

    result_dict["timings"] = timings
    result_dict["wall_dist"] = wall_dist
    result_dict["timings_P"] = timings_P
    result_dict["chain_params"] = chain_params
    result_dict["num_nlp_iter"] = num_nlp_iter
    result_dict["step_nlp_iter"] = step_nlp_iter

    if not os.path.isdir('results'):
        os.makedirs('results')

    json_file = os.path.join('results', ID + '_nm_' + str(chain_params["n_mass"]) + \
         '_seed_' + str(chain_params["seed"]) + '_iter_' + str(chain_params["nlp_iter"]) + '.json')

    save_results_as_json(result_dict, json_file)

    return



def load_results_from_json(ID, chain_params):

    if not os.path.isdir('results'):
        os.makedirs('results')

    json_file = os.path.join('results', ID + '_nm_' + str(chain_params["n_mass"]) + \
         '_seed_' + str(chain_params["seed"]) + '_iter_' + str(chain_params["nlp_iter"]) + '.json')

    with open(json_file, 'r') as f:
        results = json.load(f)

    return results


def get_input_state_trajectory(x:np.ndarray, u:np.ndarray, solver):
    N = u.shape[0]
    for idx_stage in range(N):
        x[idx_stage, :] = solver.get(idx_stage, "x")
        u[idx_stage, :] = solver.get(idx_stage, "u")
    x[N, :] = solver.get(N, "x")
    return