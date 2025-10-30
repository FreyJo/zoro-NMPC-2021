#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

import numpy as np
import scipy.linalg
import casadi as ca

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

from export_chain_mass_model import export_chain_mass_model
from export_augmented_chain_mass_model import export_augmented_chain_mass_model
from export_chain_mass_integrator import export_chain_mass_integrator

from plot_utils import *
from utils import *
import matplotlib.pyplot as plt


def run_robust_control(chain_params):
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

        # # nominal
        # nbx = M + 1
        # Jbx = np.zeros((nbx, nx_orig))
        # for i in range(nbx):
        #     Jbx[i, 3*i+1] = 1.0

        # ocp.constraints.Jbx = Jbx
        # ocp.constraints.lbx = yPosWall * np.ones((nbx,))
        # ocp.constraints.ubx = 1e9 * np.ones((nbx,))
        # ocp.constraints.Jsbx = np.eye(nbx)

        # ocp.cost.Zl = L2_pen * np.ones((nbx,))
        # ocp.cost.Zu = L2_pen * np.ones((nbx,))
        # ocp.cost.zl = L1_pen * np.ones((nbx,))
        # ocp.cost.zu = L1_pen * np.ones((nbx,))

        nh = M + 1

        ocp.constraints.lh = yPosWall * np.ones((nh,))
        ocp.constraints.uh = 1e9 * np.ones((nh,))
        ocp.constraints.Jsh = np.eye(nh)
        h_expr = ca.SX.zeros(nh,1)
        for j in range(nh):
            P_mat = vec2sym_mat( model.x[nx_orig:], nx_orig )
            # Note: lower bound, therefore need to substract the backoff term
            h_expr[j] = model.x[3*j+1] - ca.sqrt(P_mat[3*j+1,3*j+1])
        ocp.model.con_h_expr = h_expr

        ocp.cost.Zl = L2_pen * np.ones((nh,))
        ocp.cost.Zu = L2_pen * np.ones((nh,))
        ocp.cost.zl = L1_pen * np.ones((nh,))
        ocp.cost.zu = L1_pen * np.ones((nh,))



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
    num_nlp_iter = np.zeros((N_sim,))
    step_nlp_iter = np.zeros((N_sim))

    xcurrent_aug = np.hstack((xcurrent.flatten(), P0_vec))

    # closed loop
    for i in range(N_sim):
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

    print("dist2wall (minimum over simulation) ", str(np.min(wall_dist)))
    print("average time OCP: ", str(np.average(timings)))

    #%% plot results
    if show_plots:
        plot_chain_control_traj(simU)
        plot_chain_position_traj(simX, yPosWall=yPosWall)
        plot_chain_velocity_traj(simX)
        # plot_chain_position(simX[-1,:], xPosFirstMass)
        animate_chain_position(simX, xPosFirstMass, yPosWall=yPosWall)

        plt.show()

    #%% save results
    if save_results:
        ID = "robust"
        timings_Pprop = np.zeros((N_sim,))
        save_closed_loop_results_as_json(ID, timings, timings_Pprop, wall_dist, num_nlp_iter, step_nlp_iter, chain_params)
