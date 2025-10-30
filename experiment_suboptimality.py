import sys
import numpy as np
import scipy.linalg

from time import process_time

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

from export_chain_mass_model import export_chain_mass_model
from export_disturbed_chain_mass_model import export_disturbed_chain_mass_model
from export_chain_mass_integrator import export_chain_mass_integrator
from export_augmented_chain_mass_model import export_augmented_chain_mass_model

from plot_utils import *
from utils import *
import matplotlib.pyplot as plt

def ZO_robust(chain_params):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # chain parameters
    n_mass = chain_params["n_mass"]
    M = chain_params["n_mass"] - 2 # number of intermediate masses
    Ts = chain_params["Ts"]
    Tsim = chain_params["Tsim"]
    N = chain_params["N"]
    u_init = chain_params["u_init"]
    with_wall = chain_params["with_wall"]
    yPosWall = chain_params["yPosWall"]
    m = chain_params["m"]
    D = chain_params["D"]
    L = chain_params["L"]
    perturb_scale = chain_params["perturb_scale"]

    nlp_iter = chain_params["nlp_iter"]
    nlp_tol = chain_params["nlp_tol"]
    save_results = chain_params["save_results"]
    show_plots = chain_params["show_plots"]
    seed = chain_params["seed"]

    np.random.seed(seed)

    nparam = 3*M

    # experiment adjustments
    # no closed loop
    Tsim = Ts

    W = perturb_scale * np.eye(nparam)

    # export model
    model = export_disturbed_chain_mass_model(n_mass, m, D, L)
    model.name = model.name + "_robust"

    # set model
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx
    Tf = N * Ts

    # initial state
    xPosFirstMass = np.zeros((3,1))
    xEndRef = np.zeros((3,1))
    xEndRef[0] = L * (M+1) * 6
    pos0_x = np.linspace(xPosFirstMass[0], xEndRef[0], n_mass)

    xrest = compute_steady_state(n_mass, m, D, L, xPosFirstMass, xEndRef)

    x0 = xrest

    # set dimensions
    ocp.solver_options.N_horizon = N

    # set cost module
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    Q = 2*np.diagflat( np.ones((nx, 1)) )
    q_diag = np.ones((nx,1))
    strong_penalty = M+1
    q_diag[3*M] = strong_penalty
    q_diag[3*M+1] = strong_penalty
    q_diag[3*M+2] = strong_penalty
    Q = 2*np.diagflat( q_diag )

    R = 2*np.diagflat( 1e-2 * np.ones((nu, 1)) )

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx:nx+nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    # import pdb; pdb.set_trace()
    yref = np.vstack((xrest, np.zeros((nu,1)))).flatten()
    ocp.cost.yref = yref
    ocp.cost.yref_e = xrest.flatten()

    # set constraints
    umax = 1*np.ones((nu,))

    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.lbu = -umax
    ocp.constraints.ubu = umax
    ocp.constraints.x0 = x0.reshape((nx,))
    ocp.constraints.idxbu = np.array(range(nu))

    # disturbances
    nparam = 3*M
    ocp.parameter_values = np.zeros((nparam,))

    # wall constraint
    if with_wall:
        nbx = M + 1
        Jbx = np.zeros((nbx,nx))
        for i in range(nbx):
            Jbx[i, 3*i+1] = 1.0

        ocp.constraints.Jbx = Jbx
        ocp.constraints.lbx = yPosWall * np.ones((nbx,))
        ocp.constraints.ubx = 1e9 * np.ones((nbx,))

        # slacks
        # ocp.constraints.Jsbx = np.eye(nbx)
        # L2_pen = 1e3
        # L1_pen = 1
        # ocp.cost.Zl = L2_pen * np.ones((nbx,))
        # ocp.cost.Zu = L2_pen * np.ones((nbx,))
        # ocp.cost.zl = L1_pen * np.ones((nbx,))
        # ocp.cost.zu = L1_pen * np.ones((nbx,))


    # solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP, SQP_RTI

    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_num_steps = 2
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.tol = nlp_tol
    ocp.solver_options.qp_tol = nlp_tol
    ocp.solver_options.nlp_solver_max_iter = nlp_iter
    # ocp.solver_options.qp_solver_iter_max = 500
    # ocp.solver_options.qp_solver_max_iter = 100

    # set prediction horizon
    ocp.solver_options.tf = Tf


    # acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
    acados_integrator = export_chain_mass_integrator(n_mass, m, D, L)

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')

    #%% get initial state from xrest
    xcurrent = x0.reshape((nx,))
    for i in range(5):
        acados_integrator.set("x", xcurrent)
        acados_integrator.set("u", u_init)

        status = acados_integrator.solve()
        if status != 0:
            raise Exception('acados integrator returned status {}. Exiting.'.format(status))

        # update state
        xcurrent = acados_integrator.get("x")

    #%% actual simulation
    N_sim = int(np.floor(Tsim/Ts))
    simX = np.ndarray((N_sim+1, nx))
    simU = np.ndarray((N_sim, nu))
    wall_dist = np.zeros((N_sim,))

    simX[0,:] = xcurrent

    P0_mat = 1e-3 * np.eye(nx)
    P_mat_list = [None] * (N+1)
    P_mat_list[0] = P0_mat

    nbx = M + 1
    lbx = np.zeros((nbx,))

    # solve ocp
    acados_ocp_solver.set(0, "lbx", xcurrent)
    acados_ocp_solver.set(0, "ubx", xcurrent)

    if ocp.solver_options.nlp_solver_type == 'SQP_RTI':
        # SQP loop:
        for i_sqp in range(ocp.solver_options.nlp_solver_max_iter):

            # preparation rti_phase
            acados_ocp_solver.options_set('rti_phase', 1)
            status = acados_ocp_solver.solve()

            # hardcode B for discrete time disturbance
            B = np.vstack(( np.zeros((nx - nparam, nparam)), np.eye(nparam)))
            for stage in range(N):
                # get A matrices
                A = acados_ocp_solver.get_from_qp_in(stage, "A")
                # propagate Ps

                P_mat_list[stage+1] = P_propagation(P_mat_list[stage], A, B, W*Ts)

                # compute backoff using P
                for j in range(nbx):
                    P = P_mat_list[stage]
                    lbx[j] = yPosWall + np.sqrt(P[3*j+1,3*j+1])

                # set bounds with backoff (nabla h available since h linear)
                if stage > 0:
                    acados_ocp_solver.constraints_set(stage, "lbx", lbx)

            # - h <-> wall constraint

            # feedback rti_phase
            acados_ocp_solver.options_set('rti_phase', 2)
            status = acados_ocp_solver.solve()

            # check on residuals and terminate loop.
            # acados_ocp_solver.print_statistics() # encapsulates: stat = acados_ocp_solver.get_stats("statistics")
            residuals = acados_ocp_solver.get_residuals()
            print("residuals after ", i_sqp, "SQP_RTI iterations:\n", residuals)

            if status != 0:
                raise Exception('acados acados_ocp_solver returned status {} in time step {}. Exiting.'.format(status, i))

            if max(residuals) < nlp_tol:
                break

    else:
        status = acados_ocp_solver.solve()
        # acados_ocp_solver.print_statistics() # encapsulates: stat = acados_ocp_solver.get_stats("statistics")
        if status != 0:
            raise Exception('acados acados_ocp_solver returned status {} in time step {}. Exiting.'.format(status, i))

    simU[0,:] = acados_ocp_solver.get(0, "u")

    # only LQR
    # simU[i,:] = K @ (xcurrent - xrest.flatten())

    print("control at time", 0, ":", simU[0,:])

    i = 0
    # simulate system
    acados_integrator.set("x", xcurrent)
    acados_integrator.set("u", simU[0,:])

    pertubation = sampleFromEllipsoid(np.zeros((nparam,)), W)
    # acados_integrator.set("p", pertubation)

    status = acados_integrator.solve()
    if status != 0:
        raise Exception('acados integrator returned status {}. Exiting.'.format(status))

    # update state
    xcurrent = acados_integrator.get("x") + Ts * np.hstack((np.zeros(((M+1)*3,)), pertubation))
    simX[i+1,:] = xcurrent

    # xOcpPredict = acados_ocp_solver.get(1, "x")
    # print("model mismatch = ", str(np.max(xOcpPredict - xcurrent)))
    yPos = xcurrent[range(1,3*M+2,3)]
    wall_dist[i] = np.min(yPos - yPosWall)
    print("time i = ", str(i), " dist2wall ", str(wall_dist[i]))

    # get cost function value
    cost = acados_ocp_solver.get_cost()
    print("cost function value of solution = ", cost)

    # compute cost ourselves - without soft constraint contribution
    cost2 = 0
    for j in range(N):
        xocp = acados_ocp_solver.get(j, "x")
        uocp = acados_ocp_solver.get(j, "u")

        dx = xocp[:nx] - xrest.flatten()
        cost2 += dx.T @ Q @ dx + uocp.T @ R @ uocp

    xocp = acados_ocp_solver.get(N, "x")
    dx = xocp[:nx] - xrest.flatten()
    cost2 += dx.T @ Q @ dx

    return cost, cost2


def naive_robust(chain_params):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # chain parameters
    n_mass = chain_params["n_mass"]
    M = chain_params["n_mass"] - 2 # number of intermediate masses
    Ts = chain_params["Ts"]
    Tsim = chain_params["Tsim"]
    N = chain_params["N"]
    u_init = chain_params["u_init"]
    with_wall = chain_params["with_wall"]
    yPosWall = chain_params["yPosWall"]
    m = chain_params["m"]
    D = chain_params["D"]
    L = chain_params["L"]
    perturb_scale = chain_params["perturb_scale"]

    nlp_iter = chain_params["nlp_iter"]
    nlp_tol = chain_params["nlp_tol"]
    save_results = chain_params["save_results"]
    show_plots = chain_params["show_plots"]
    seed = chain_params["seed"]

    np.random.seed(seed)

    nparam = 3*M
    W = perturb_scale * np.eye(nparam)

    # export model
    model = export_augmented_chain_mass_model(n_mass, m, D, L, Ts, W)

    # set model
    ocp.model = model

    nx_orig = M * 3 + (M+1)*3
    nx_aug = model.x.shape[0]

    nu = model.u.size()[0]
    ny = nx_orig + nu
    ny_e = nx_orig
    Tf = N * Ts

    # initial state
    xPosFirstMass = np.zeros((3,1))
    xEndRef = np.zeros((3,1))
    xEndRef[0] = L * (M+1) * 6
    pos0_x = np.linspace(xPosFirstMass[0], xEndRef[0], n_mass)

    xrest = compute_steady_state(n_mass, m, D, L, xPosFirstMass, xEndRef)

    x0 = xrest

    P0_mat = 1e-3 * np.eye(nx_orig)
    P0_vec = sym_mat2vec(P0_mat)

    # set dimensions
    ocp.solver_options.N_horizon = N

    # set cost module
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    Q = 2*np.diagflat( np.ones((nx_orig, 1)) )
    q_diag = np.ones((nx_orig,1))
    strong_penalty = M+1
    q_diag[3*M] = strong_penalty
    q_diag[3*M+1] = strong_penalty
    q_diag[3*M+2] = strong_penalty
    Q = 2*np.diagflat( q_diag )

    R = 2*np.diagflat( 1e-2 * np.ones((nu, 1)) )

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q

    ocp.cost.Vx = np.zeros((ny, nx_aug))
    ocp.cost.Vx[:nx_orig,:nx_orig] = np.eye(nx_orig)

    Vu = np.zeros((ny, nu))
    Vu[nx_orig:nx_orig+nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx_aug))
    Vx_e[:nx_orig, :nx_orig] = np.eye(nx_orig)
    ocp.cost.Vx_e = Vx_e


    yref = np.vstack((xrest, np.zeros((nu,1)))).flatten()
    ocp.cost.yref = yref
    ocp.cost.yref_e = xrest.flatten()

    # set constraints
    umax = 1*np.ones((nu,))

    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.lbu = -umax
    ocp.constraints.ubu = umax
    x0_aug = np.hstack((x0.flatten(), P0_vec))
    ocp.constraints.x0 = x0_aug.reshape((nx_aug,))
    ocp.constraints.idxbu = np.array(range(nu))

    # disturbances

    # wall constraint
    if with_wall:
        # slacks
        L2_pen = 1e3
        L1_pen = 1

        nh = M + 1

        ocp.constraints.lh = yPosWall * np.ones((nh,))
        ocp.constraints.uh = 1e9 * np.ones((nh,))
        # ocp.constraints.Jsh = np.eye(nh)
        h_expr = ca.SX.zeros(nh,1)
        for j in range(nh):
            P_mat = vec2sym_mat( model.x[nx_orig:], nx_orig )
            # Note: lower bound, therefore need to substract the backoff term
            h_expr[j] = model.x[3*j+1] - ca.sqrt(P_mat[3*j+1,3*j+1])
        ocp.model.con_h_expr = h_expr

        # ocp.cost.Zl = L2_pen * np.ones((nh,))
        # ocp.cost.Zu = L2_pen * np.ones((nh,))
        # ocp.cost.zl = L1_pen * np.ones((nh,))
        # ocp.cost.zu = L1_pen * np.ones((nh,))



    # solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES' # FULL_CONDENSING_QPOASES

    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI
    ocp.solver_options.nlp_solver_max_iter = nlp_iter

    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_num_steps = 2
    ocp.solver_options.qp_solver_cond_N = N # N TODO
    ocp.solver_options.tol = nlp_tol
    ocp.solver_options.qp_tol = nlp_tol
    ocp.solver_options.print_level = 0

    # set prediction horizon
    ocp.solver_options.tf = Tf
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
    print("Generated AcadosOcpSolver successfully")
    # acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
    acados_integrator = export_chain_mass_integrator(n_mass, m, D, L)

    #%% get initial state from xrest
    xcurrent = x0.reshape((nx_orig,))
    for i in range(5):
        acados_integrator.set("x", xcurrent)
        acados_integrator.set("u", u_init)

        status = acados_integrator.solve()
        if status != 0:
            raise Exception('acados integrator returned status {}. Exiting.'.format(status))

        # update state
        xcurrent = acados_integrator.get("x")

    #%% actual simulation
    N_sim = int(np.floor(Tsim/Ts))
    simX = np.ndarray((N_sim+1, nx_orig))
    simU = np.ndarray((N_sim, nu))
    wall_dist = np.zeros((N_sim,))
    simX[0,:] = xcurrent

    timings = np.zeros((N_sim,))
    Pposdef = np.zeros((N_sim, N))

    xcurrent_aug = np.hstack((xcurrent.flatten(), P0_vec))

    # no closed loop
    i = 0
    # solve ocp
    acados_ocp_solver.set(0, "lbx", xcurrent_aug)
    acados_ocp_solver.set(0, "ubx", xcurrent_aug)

    status = acados_ocp_solver.solve()
    acados_ocp_solver.print_statistics()

    if status != 0:
        raise Exception('acados acados_ocp_solver returned status {} in time step {}. Exiting.'.format(status, i))

    simU[i,:] = acados_ocp_solver.get(0, "u")

    # simulate system
    acados_integrator.set("x", xcurrent)
    acados_integrator.set("u", simU[i,:])

    pertubation = sampleFromEllipsoid(np.zeros((nparam,)), W)
    acados_integrator.set("p", pertubation)

    status = acados_integrator.solve()
    if status != 0:
        raise Exception('acados integrator returned status {}. Exiting.'.format(status))

    # update state
    xcurrent = acados_integrator.get("x")
    simX[i+1,:] = xcurrent
    xcurrent_aug = np.hstack((xcurrent.flatten(), P0_vec))

    # get P covariances
    for j in range(N):
        xocp = acados_ocp_solver.get(j, "x")
        P = vec2sym_mat(xocp[nx_orig:], nx_orig)
        Pposdef[i,j] = is_pos_def(P)

    timings[i] = acados_ocp_solver.get_stats("time_tot")

    yPos = xcurrent[range(1,3*M+1,3)]
    wall_dist[i] = np.min(yPos - yPosWall)
    print("time i = ", str(i), " dist2wall ", str(wall_dist[i]))


    # get cost function value
    cost = acados_ocp_solver.get_cost()
    print("cost function value of solution = ", cost)

    # compute cost ourselves - without soft constraint contribution
    cost2 = 0
    for j in range(N):
        xocp = acados_ocp_solver.get(j, "x")
        uocp = acados_ocp_solver.get(j, "u")

        dx = xocp[:nx_orig] - xrest.flatten()
        cost2 += dx.T @ Q @ dx + uocp.T @ R @ uocp

    xocp = acados_ocp_solver.get(N, "x")
    dx = xocp[:nx_orig] - xrest.flatten()
    cost2 += dx.T @ Q @ dx

    return cost, cost2


# MAIN
chain_params = get_chain_params()


# min_ps = -3
# max_ps = 0
# n_pert = 30

min_ps = -6
max_ps = -1
n_pert = 30

PS = np.logspace(min_ps, max_ps, n_pert)
i_pert = 0

cost_ZO = np.zeros((n_pert,1))
cost_exact = np.zeros((n_pert,1))
cost2_ZO = np.zeros((n_pert,1))
cost2_exact = np.zeros((n_pert,1))
chain_params["nlp_tol"] = 1e-6

for perturb_scale in PS:
    chain_params["perturb_scale"] = perturb_scale
    cost_ZO[i_pert], cost2_ZO[i_pert] = ZO_robust(chain_params)
    # cost, cost2 = ZO_robust(chain_params)
    # import pdb; pdb.set_trace()

    cost_exact[i_pert], cost2_exact[i_pert] = naive_robust(chain_params)
    i_pert += 1


results = dict()
results["pertubation_scaling"] = PS
results["cost_ZO"] = cost_ZO
results["cost_exact"] = cost_exact
results["cost2_ZO"] = cost2_ZO
results["cost2_exact"] = cost2_exact


# json_file = "results/suboptimality" + str(n_pert) + ".json"
json_file = "results/suboptimality" + str(min_ps) + "_" + str(max_ps) + "_" + str(n_pert) + ".json"
json_file = "results/suboptimality_high_acc" + str(min_ps) + "_" + str(max_ps) + "_" + str(n_pert) + ".json"

save_results_as_json(results, json_file)

plot_suboptimility(PS, cost_exact, cost_ZO, "acados cost")
