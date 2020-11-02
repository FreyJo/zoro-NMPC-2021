from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, norm_2, Function, transpose, jacobian

import numpy as np

from export_disturbed_chain_mass_model import export_disturbed_chain_mass_model

from utils import *

def export_augmented_chain_mass_model(n_mass, m, D, L, h, W):

    model_name = 'chain_mass_aug_ds_' + str(n_mass) + '_ps_' + str(W[0][0]).replace('-','_').replace('.','o')
    og = export_disturbed_chain_mass_model(n_mass, m, D, L)
    nx = og.x.size()[0]

    aug_model = AcadosModel()

    P = SX.sym('P_vec', int((nx+1)*nx/2))
    Pdot = SX.sym('Pdot_vec', int((nx+1)*nx/2))

    x_aug = vertcat(og.x, P)

    A_fun = Function('A_fun', [og.x, og.u, og.p], [jacobian(og.f_expl_expr, og.x)])
    A = A_fun(og.x, og.u, 0)

    # NOTE: need disturbances og.p here, but we eliminate the disturbances from the model later.
    B_fun = Function('B_fun', [og.x, og.u, og.p], [jacobian(og.f_expl_expr, og.p)])
    B = B_fun(og.x, og.u, 0)

    P_mat = vec2sym_mat(P, nx)
    Pdot_mat = A @ P_mat + P_mat @ A.T + B @ W @ B.T
    Pdot_expr = sym_mat2vec(Pdot_mat)
    # set up discrete dynamics
    f_expl = Function('f_expl_chain', [og.x, og.u, og.p], [og.f_expl_expr])
    nominal_f_expl_expr = f_expl(og.x, og.u, 0)
    # k1 = f_expl(og.x, og.u, 0)
    # k2 = f_expl(og.x + h * k1/2, og.u, 0)
    # k3 = f_expl(og.x + h * k2/2, og.u, 0)
    # k4 = f_expl(og.x + h * k3, og.u, 0)
    # xplus = og.x + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    # A_fun = Function('A_fun', [og.x, og.u, og.p], [jacobian(xplus, og.x)])
    # A = A_fun(og.x, og.u, 0)

    # # NOTE: need disturbances og.p here, but we eliminate the disturbances from the model later.
    # B_fun = Function('B_fun', [og.x, og.u, og.p], [jacobian(xplus, og.p)])
    # B = B_fun(og.x, og.u, 0)

    # P_mat = vec2sym_mat(P, nx)
    # P_mat = SX.zeros(nx,nx)
    # start_P = 0
    # for i in range(nx):
    #     end_P = start_P + (nx - i)
    #     P_mat[i,i:] = transpose(P[start_P:end_P])
    #     P_mat[i:,i] = P[start_P:end_P]
    #     start_P += (nx-i)

    # Pplus_mat = A @ P_mat @ transpose(A) + B @ W @ transpose(B)

    # Pplus = SX.zeros(int((nx+1)*nx/2), 1)

    # start_P = 0
    # for i in range(nx):
    #     end_P = start_P + (nx - i)
    #     # P_mat[i,i:] = transpose(P[start_P:end_P])
    #     # P_mat[i:,i] = P[start_P:end_P]
    #     Pplus[start_P:end_P] = Pplus_mat[i:,i]
    #     start_P += (nx-i)

    # # euler trick
    # xplus_aug = vertcat(xplus, Pplus)
    # f_cont = (xplus_aug - x_aug) / h


    # fill model
    xdot = vertcat(og.xdot, Pdot)
    f_expl_expr = vertcat(nominal_f_expl_expr, Pdot_expr)
    aug_model.f_expl_expr = f_expl_expr
    aug_model.f_impl_expr = f_expl_expr - xdot

    aug_model.xdot = xdot
    aug_model.u = og.u
    aug_model.x = x_aug
    aug_model.name = model_name

    return aug_model

