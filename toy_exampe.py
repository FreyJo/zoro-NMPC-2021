import casadi as ca
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

# UPDATE_FIG = True
UPDATE_FIG = False

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
font = {'family':'serif'}
plt.rc('font',**font)

maxiter = 500
TOL = 1e-8

# dims
nx = 2
nw = 2

# uncertainty values
sigma_v = np.linspace(1e-6,np.sqrt(5)*1.5,30)

# variables
x = ca.SX.sym('x', nx, 1)
w = ca.SX.sym('x', nw, 1)
P = ca.SX.sym('P', nw**2, 1)
sigma = ca.SX.sym('sigma', 1, 1)

# constants
c = np.array([[0.0], [0.0]])
M = np.eye(2)

# objective
f = 1/2*ca.mtimes(ca.mtimes(x.T, M), x)
# for plotting
f_fun = ca.Function('f_fun', [x[0],x[1]], [f]) 

A = ca.SX.zeros(nw**2, nw**2)
A[0,0] = 1 + 0.6*ca.sin(x[0])
A[1,1] = 1 + 0.6*ca.cos(x[1]) 
A[2,2] = 1 + 0.6*ca.sin(x[1]**2) 
A[3,3] = 1 
A[2,1] = 0.1 + ca.sin(x[1]) 

b = np.array([1,0,0,1]) 

# constraints
h = -ca.sqrt((x[0] + w[0] -1)**2 + (x[1] + w[1] -1)**2) + ca.sqrt(10) 

h_w = ca.substitute(ca.jacobian(h, w), w, 0).T
# for plotting
h_fun = ca.Function('h_fun', [x[0],x[1]], [ca.substitute(h,w,0)]) 

P_ = P.reshape((nw,nw))
h_hat = ca.substitute(h,w,0) + ca.sqrt(ca.mtimes(ca.mtimes(h_w.T, P_), h_w))

g = ca.vertcat(ca.mtimes(A,P) - sigma**2*b, h_hat)

ubg = np.zeros((nw**2+1,1))
lbg = np.zeros((nw**2+1,1))
lbg[-1] = -np.inf
x0 = ca.vertcat(10*np.ones((nx,1)), 0.1*np.ones((nw**2,1))) 

# create nlp solver
nlp = {'x':ca.vertcat(x,P), 'f':f, 'g':g, 'p':sigma}
nlp_solver = ca.nlpsol("solver", "ipopt", nlp)

# solve
sol_x_v = []
for i in range(len(sigma_v)):
    sol = nlp_solver(x0=x0, lbg=lbg, ubg=ubg, p = sigma_v[i])
    sol_x_v.append(sol['x'])
    x0 = sol['x']
    status = nlp_solver.stats()['success']
    if status != True:
        raise Exception('Solver failed at iteration {}'.format(i))

# define linearization points
x_bar = ca.SX.sym('x_bar', nx, 1)
P_bar = ca.SX.sym('P_bar', nw**2, 1)

# define QP opt. variables
dx = ca.SX.sym('dx', nx, 1)

A_fun = ca.Function('A_bar', [x],[A])
b_fun = ca.Function('b_bar', [x],[b])
h_hat_fun = ca.Function('h_hat_bar', [x,P], [h_hat])

# define QP 
h_hat_ = ca.substitute(h_hat, x, x_bar)
h_hat_ = ca.substitute(h_hat_, P, P_bar)

f_x = ca.substitute(ca.jacobian(f,x), x, x_bar)

h_hat_x = ca.substitute(ca.jacobian(h_hat,x), x, x_bar)
h_hat_x = ca.substitute(h_hat_x, P, P_bar)
h_hat_P = ca.substitute(ca.jacobian(h_hat, P), x, x_bar)
h_hat_P = ca.substitute(h_hat_P, P, P_bar)
h_hat_P_fun = ca.Function('h_hat_P_fun', [x,P], [ca.jacobian(h_hat,P)])

f_qp = ca.mtimes(f_x, dx) + 1/2*ca.mtimes(ca.mtimes(dx.T, M),dx) 
g_qp = ca.mtimes(h_hat_x, dx) 

# create QP solver
pars = ca.vertcat(sigma, x_bar, P_bar)

qp = {'x':ca.vertcat(dx), 'f':f_qp, 'g':g_qp, 'p':pars}
qp_solver = ca.nlpsol("solver", "ipopt", qp)
lbg = [-ca.inf]
ubg = [0]

# solve zoRO problem
sol_x_zoro_v = []
x_bar = np.zeros((nx,1))
P_bar = np.ones((nw**2,1))
for i in range(len(sigma_v)):
    # SQP
    for j in range(maxiter):
        # eliminate P
        A_bar = A_fun(x_bar).full()
        b_bar = b_fun(x_bar).full()

        P_plus = np.linalg.solve(A_bar, (sigma_v[i]**2)*b_bar) 

        # update rhs
        h_hat_bar = h_hat_fun(x_bar, P_bar)
        h_hat_P_bar = h_hat_P_fun(x_bar, P_bar).full()
        ubg[0] = (-h_hat_bar - ca.mtimes(h_hat_P_bar, P_plus - P_bar)).full() 

        p = ca.vertcat(sigma_v[i], x_bar, P_bar).full()
        sol = qp_solver(x0=x_bar,lbg=lbg, ubg=ubg, p=p)
        status = nlp_solver.stats()['success']

        if status != True:
            raise Exception('Solver failed at iteration {},{j}'.format(i,j))

        dx = sol['x']
        
        # step
        dP = (P_plus - P_bar)
        x_bar = x_bar + dx
        P_bar = P_bar + dP 

        if np.linalg.norm(np.concatenate
                ([dx, dP])) < TOL:
            print('Solution found!')
            sol_x_zoro_v.append(ca.vertcat(x_bar,P_bar))
            break 

        if j >= maxiter-1:
            raise Exception('max. number of iterations reached at'\
                ' iteration i={} j={}!'.format(i,j))

# plot 
fig, axs = plt.subplots(1, figsize=(4,3))
err = []

def plot_ellipse(P, c, axs, color, style,alpha):
    c.shape = (nx,1)
    npoints = 50
    theta = np.linspace(0, 2*np.pi)
    P_half = sqrtm(P)
    w = np.zeros((2,npoints))
    for i in range(npoints):
        v = np.array([np.cos(theta[i]), np.sin(theta[i])])
        v.shape = (2,1)
        w[:,i] = (c + np.matmul(P_half,v)).reshape(2,)
    axs.plot(w[0,:],w[1,:], color, alpha=alpha, linestyle=style, label='_no_legend_')

def plot_constraint(axs, xlim, ylim):
    delta = 0.025
    x1 = np.arange(xlim[0], xlim[1], delta)
    x2 = np.arange(ylim[0], ylim[1], delta)
    X1, X2 = np.meshgrid(x1, x2)
    Z = h_fun(X1,X2) 
    axs.contourf(X1,X2,Z,[0,10],color=['black'],alpha=0.3, label='_no_legend_')
    axs.contour(X1,X2,Z,[0],color=['black'], label='_no_legend_')

def plot_objective(axs, xlim, ylim):
    delta = 0.025
    x1 = np.arange(xlim[0], xlim[1], delta)
    x2 = np.arange(ylim[0], ylim[1], delta)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f_fun(X1,X2) 
    axs.contour(X1,X2,Z,alpha=0.5, label='_no_legend_')

sol_x_p = np.zeros((2, len(sigma_v)))
sol_x_zoro_p = np.zeros((2, len(sigma_v)))
c1 = 'black'
c2 = 'black'
s1 = 'dashed'
s2 = 'solid'
alpha1 = 0.5
alpha2 = 1.0

for i in range(len(sigma_v)):
    sol_x = sol_x_v[i].full()[0:2]
    sol_x.shape = (nx,)
    sol_x_p[:,i] = sol_x 
    sol_x_zoro = sol_x_zoro_v[i].full()[0:2]
    sol_x_zoro.shape = (nx,)
    sol_x_zoro_p[:,i] = sol_x_zoro

    if i%3 == 0:
        P = sol_x_v[i].full()[nx:].reshape((nw, nw))
        plot_ellipse(P, np.array([sol_x[0], sol_x[1]]), axs, c1, s1,alpha=alpha1)
        axs.plot(sol_x_p[0,i], sol_x_p[1,i],'black', marker='o', markersize=3,alpha=alpha1, label='_no_legend_')

        P = sol_x_zoro_v[i].full()[nx:].reshape((nw, nw))
        plot_ellipse(P, np.array([sol_x_zoro[0], sol_x_zoro[1]]), axs, c2, s2,alpha=alpha2)
        axs.plot(sol_x_zoro_p[0,i], sol_x_zoro_p[1,i],c2, linestyle=s2, marker = 'o', markersize = 3,alpha=alpha2, label='_no_legend_')

    err.append(np.linalg.norm(sol_x - sol_x_zoro))
   
axs.plot(sol_x_p[0,:], sol_x_p[1,:], c1, linestyle=s1,alpha=alpha1)
axs.plot(sol_x_zoro_p[0,:], sol_x_zoro_p[1,:], c2, linestyle=s2,alpha=alpha2)

axs.grid()
plot_constraint(axs, axs.get_xlim(), axs.get_ylim())
plot_objective(axs, axs.get_xlim(), axs.get_ylim())
axs.set_xlabel(r"$y_1$")
axs.set_ylabel(r"$y_2$")
axs.legend([r"exact", r"zoRO"],loc=2)

plt.figure()
plt.loglog(sigma_v, err)
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$\|\tilde{z}(\sigma) - \bar{z}(\sigma) \|$")
plt.grid()

# fit poynomials
p3 = np.polyfit(sigma_v, err,3)
p3_line = sigma_v**3*p3[3] +sigma_v**2*p3[2] +sigma_v**1*p3[1] +sigma_v**0*p3[0]  
p2 = np.polyfit(sigma_v, err,2)
p2_line = sigma_v**2*p2[2] +sigma_v**1*p2[1] +sigma_v**0*p2[0]  
p1 = np.polyfit(sigma_v, err,1)
p1_line = sigma_v**1*p1[1] +sigma_v**0*p1[0]  

if UPDATE_FIG:
    fig.savefig('illustrative_example.pdf', dpi=300, bbox_inches="tight")

plt.show()
