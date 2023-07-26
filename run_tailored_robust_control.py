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

import sys
import os
from casadi.casadi import sqrt
import numpy as np
import scipy.linalg

from time import process_time

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

from export_chain_mass_model import export_chain_mass_model
from export_disturbed_chain_mass_model import export_disturbed_chain_mass_model
from export_chain_mass_integrator import export_chain_mass_integrator

from plot_utils import *
from utils import *
import matplotlib.pyplot as plt

def run_tailored_robust_control(chain_params):
    ID = "zoRO"

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
    ocp.dims.N = N

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
        ocp.constraints.Jsbx = np.eye(nbx)
        L2_pen = 1e3
        L1_pen = 1
        ocp.cost.Zl = L2_pen * np.ones((nbx,))
        ocp.cost.Zu = L2_pen * np.ones((nbx,))
        ocp.cost.zl = L1_pen * np.ones((nbx,))
        ocp.cost.zu = L1_pen * np.ones((nbx,))


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
    # ocp.solver_options.initialize_t_slacks = 1
    # ocp.solver_options.qp_solver_max_iter = 100

    # set prediction horizon
    ocp.solver_options.tf = Tf

    ocp.code_export_directory = "c_generated_code" + "_" + ID

    # acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
    acados_integrator = export_chain_mass_integrator(n_mass, m, D, L)
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
    # AcadosOcpSolver.generate(ocp, json_file='acados_ocp_' + model.name + '.json')
    # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    # acados_ocp_solver = AcadosOcpSolver.create_cython_solver('acados_ocp_' + model.name + '.json')


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

    timings = np.zeros((N_sim,))
    timings_Pprop = np.zeros((N_sim,))

    simX[0,:] = xcurrent

    P0_mat = 1e-3 * np.eye(nx)
    P_mat_list = [None] * (N+1)
    P_mat_list[0] = P0_mat

    nbx = M + 1
    lbx = np.zeros((nbx,))

    # closed loop
    for i in range(N_sim):

        # solve ocp
        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)

        if ocp.solver_options.nlp_solver_type == 'SQP_RTI':
            # SQP loop:
            for i_sqp in range(ocp.solver_options.nlp_solver_max_iter):

                # preparation rti_phase
                acados_ocp_solver.options_set('rti_phase', 1)
                status = acados_ocp_solver.solve()
                timings[i] += acados_ocp_solver.get_stats("time_tot")[0]

                # hardcode B for discrete time disturbance
                B = np.vstack(( np.zeros((nx - nparam, nparam)), np.eye(nparam)))
                for stage in range(N):
                    # get A matrices
                    A = acados_ocp_solver.get_from_qp_in(stage, "A")
                    # propagate Ps

                    P_mat_old = P_mat_list[stage+1]

                    t1 = process_time()
                    P_mat_list[stage+1] = P_propagation(P_mat_list[stage], A, B, W*Ts)
                    timings_Pprop[i] += process_time() - t1

                    if isinstance(P_mat_old, type(None)):
                        # i == 0
                        dP_bar = np.zeros((nx,nx))
                    else:
                        dP_bar = P_mat_list[stage+1] - P_mat_old

                    # compute backoff using P
                    for j in range(nbx):
                        P = P_mat_list[stage]
                        ibx = 3*j+1
                        lbx[j] = yPosWall + np.sqrt(P[ibx, ibx]) + .5 * dP_bar[ibx, ibx] / np.sqrt(P[ibx, ibx])

                    # set bounds with backoff (nabla h available since h linear)
                    if stage > 0:
                        acados_ocp_solver.constraints_set(stage, "lbx", lbx)

                # - h <-> wall constraint

                # feedback rti_phase
                acados_ocp_solver.options_set('rti_phase', 2)
                status = acados_ocp_solver.solve()
                timings[i] += acados_ocp_solver.get_stats("time_tot")[0]

                # check on residuals and terminate loop.
                # acados_ocp_solver.print_statistics() # encapsulates: stat = acados_ocp_solver.get_stats("statistics")
                residuals = acados_ocp_solver.get_residuals()
                # print("residuals after ", i_sqp, "SQP_RTI iterations:\n", residuals)

                if status != 0:
                    raise Exception('acados acados_ocp_solver returned status {} in time step {}. Exiting.'.format(status, i))

                if max(residuals) < nlp_tol:
                    break

        else:
            status = acados_ocp_solver.solve()
            # acados_ocp_solver.print_statistics() # encapsulates: stat = acados_ocp_solver.get_stats("statistics")
            if status != 0:
                raise Exception('acados acados_ocp_solver returned status {} in time step {}. Exiting.'.format(status, i))

        simU[i,:] = acados_ocp_solver.get(0, "u")

        # only LQR
        # simU[i,:] = K @ (xcurrent - xrest.flatten())

        print("control at time", i, ":", simU[i,:])

        # simulate system
        acados_integrator.set("x", xcurrent)
        acados_integrator.set("u", simU[i,:])

        pertubation = sampleFromEllipsoid(np.zeros((nparam,)), W)
        # acados_integrator.set("p", pertubation)

        status = acados_integrator.solve()
        if status != 0:
            raise Exception('acados integrator returned status {}. Exiting.'.format(status))

        # update state
        # import pdb; pdb.set_trace()
        xcurrent = acados_integrator.get("x") + Ts * np.hstack((np.zeros(((M+1)*3,)), pertubation))
        simX[i+1,:] = xcurrent

        # xOcpPredict = acados_ocp_solver.get(1, "x")
        # print("model mismatch = ", str(np.max(xOcpPredict - xcurrent)))
        yPos = xcurrent[range(1,3*M+2,3)]
        wall_dist[i] = np.min(yPos - yPosWall)
        print("time i = ", str(i), " dist2wall ", str(wall_dist[i]))

    print("dist2wall (minimum over simulation) ", str(np.min(wall_dist)))

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
        save_closed_loop_results_as_json(ID, timings, timings_Pprop, wall_dist, chain_params)
