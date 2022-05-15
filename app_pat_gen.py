import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt

def huber_fitting_pat(n=10,m=100):
    # Generate problem 1 data
    sp.random.seed(1)
    # n = 10
    # m = 100
    Ad = sparse.random(m, n, density=0.5, format='csc')
    x_true = np.random.randn(n) / np.sqrt(n)
    ind95 = (np.random.rand(m) < 0.95).astype(float)
    b = Ad.dot(x_true) + np.multiply(0.5*np.random.randn(m), ind95) \
        + np.multiply(10.*np.random.rand(m), 1. - ind95)

    # OSQP data
    Im = sparse.eye(m)
    P = sparse.block_diag([sparse.csc_matrix((n, n)), 2*Im,
                        sparse.csc_matrix((2*m, 2*m))],
                        format='csc')
    q = np.append(np.zeros(m+n), 2*np.ones(2*m))
    A = sparse.bmat([[Ad,   -Im,   -Im,   Im],
                    [None,  None,  Im,   None],
                    [None,  None,  None, Im]], format='csc')
    l = np.hstack([b, np.zeros(2*m)])
    u = np.hstack([b, np.inf*np.ones(2*m)])
    return A, P

def lasso_pat(n=12,m=1000):
    # Generate problem 2 data
    sp.random.seed(1)
    # n = 10
    # n = 12
    # m = 1000
    Ad = sparse.random(m, n, density=0.5)
    x_true = np.multiply((np.random.rand(n) > 0.8).astype(float),
                        np.random.randn(n)) / np.sqrt(n)
    b = Ad.dot(x_true) + 0.5*np.random.randn(m)
    gammas = np.linspace(1, 10, 11)

    # Auxiliary data
    In = sparse.eye(n)
    Im = sparse.eye(m)
    On = sparse.csc_matrix((n, n))
    Onm = sparse.csc_matrix((n, m))

    # OSQP data
    P = sparse.block_diag([On, sparse.eye(m), On], format='csc')
    q = np.zeros(2*n + m)
    A = sparse.vstack([sparse.hstack([Ad, -Im, Onm.T]),
                    sparse.hstack([In, Onm, -In]),
                    sparse.hstack([In, Onm, In])], format='csc')
    l = np.hstack([b, -np.inf * np.ones(n), np.zeros(n)])
    u = np.hstack([b, np.zeros(n), np.inf * np.ones(n)])
    return A, P

def mpc_pat():
    # Discrete time model of a quadcopter
  Ad = sparse.csc_matrix([
    [1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.    ],
    [0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.    ],
    [0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.    ],
    [0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.    ],
    [0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.    ],
    [0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992],
    [0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.    ],
    [0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.    ],
    [0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.    ],
    [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.    ],
    [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.    ],
    [0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846]
  ])
  Bd = sparse.csc_matrix([
    [0.,      -0.0726,  0.,     0.0726],
    [-0.0726,  0.,      0.0726, 0.    ],
    [-0.0152,  0.0152, -0.0152, 0.0152],
    [-0.,     -0.0006, -0.,     0.0006],
    [0.0006,   0.,     -0.0006, 0.0000],
    [0.0106,   0.0106,  0.0106, 0.0106],
    [0,       -1.4512,  0.,     1.4512],
    [-1.4512,  0.,      1.4512, 0.    ],
    [-0.3049,  0.3049, -0.3049, 0.3049],
    [-0.,     -0.0236,  0.,     0.0236],
    [0.0236,   0.,     -0.0236, 0.    ],
    [0.2107,   0.2107,  0.2107, 0.2107]])
  [nx, nu] = Bd.shape

  # Constraints
  u0 = 10.5916
  umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
  umax = np.array([13., 13., 13., 13.]) - u0
  xmin = np.array([-np.pi/6,-np.pi/6,-np.inf,-np.inf,-np.inf,-1.,
                  -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])
  xmax = np.array([ np.pi/6, np.pi/6, np.inf, np.inf, np.inf, np.inf,
                    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

  # Objective function
  Q = sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
  QN = Q
  R = 0.1*sparse.eye(4)

  # Initial and reference states
  x0 = np.zeros(12)
  xr = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

  # Prediction horizon
  N = 10

  # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
  # - quadratic objective
  P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                        sparse.kron(sparse.eye(N), R)], format='csc')
  # - linear objective
  q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
                np.zeros(N*nu)])
  # - linear dynamics
  Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
  Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
  Aeq = sparse.hstack([Ax, Bu])
  leq = np.hstack([-x0, np.zeros(N*nx)])
  ueq = leq
  # - input and state constraints
  Aineq = sparse.eye((N+1)*nx + N*nu)
  lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
  uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
  # - OSQP constraints
  A = sparse.vstack([Aeq, Aineq], format='csc')
  l = np.hstack([leq, lineq])
  u = np.hstack([ueq, uineq])

  return A, P

def prof_pat():
    sp.random.seed(1)
    n = 100
    k = 10
    F = sparse.random(n, k, density=0.7, format='csc')
    D = sparse.diags(np.random.rand(n) * np.sqrt(k), format='csc')
    mu = np.random.randn(n)
    gamma = 1

    # OSQP data
    P = sparse.block_diag([D, sparse.eye(k)], format='csc')
    q = np.hstack([-mu / (2*gamma), np.zeros(k)])
    A = sparse.vstack([
            sparse.hstack([F.T, -sparse.eye(k)]),
            sparse.hstack([sparse.csc_matrix(np.ones((1, n))), sparse.csc_matrix((1, k))]),
            sparse.hstack((sparse.eye(n), sparse.csc_matrix((n, k))))
        ], format='csc')
    l = np.hstack([np.zeros(k), 1., np.zeros(n)])
    u = np.hstack([np.zeros(k), 1., np.ones(n)])
    return A, P

def svm_pat(n=10, m=1000):
    # Generate problem data
    sp.random.seed(1)
    # n = 10
    # m = 1000
    N = int(m / 2)
    gamma = 1.0
    b = np.hstack([np.ones(N), -np.ones(N)])
    A_upp = sparse.random(N, n, density=0.5)
    A_low = sparse.random(N, n, density=0.5)
    Ad = sparse.vstack([
            A_upp / np.sqrt(n) + (A_upp != 0.).astype(float) / n,
            A_low / np.sqrt(n) - (A_low != 0.).astype(float) / n
        ], format='csc')

    # OSQP data
    Im = sparse.eye(m)
    P = sparse.block_diag([sparse.eye(n), sparse.csc_matrix((m, m))], format='csc')
    q = np.hstack([np.zeros(n), gamma*np.ones(m)])
    A = sparse.vstack([
            sparse.hstack([sparse.diags(b).dot(Ad), -Im]),
            sparse.hstack([sparse.csc_matrix((m, n)), Im])
        ], format='csc')
    l = np.hstack([-np.inf*np.ones(m), np.zeros(m)])
    u = np.hstack([-np.ones(m), np.inf*np.ones(m)])
    return A, P

def least_square_pat(n=20, m=30):
    sp.random.seed(1)
    # m = 30
    # n = 20
    Ad = sparse.random(m, n, density=0.7, format='csc')
    b = np.random.randn(m)

    # OSQP data
    P = sparse.block_diag([sparse.csc_matrix((n, n)), sparse.eye(m)], format='csc')
    q = np.zeros(n+m)
    A = sparse.vstack([
            sparse.hstack([Ad, -sparse.eye(m)]),
            sparse.hstack([sparse.eye(n), sparse.csc_matrix((n, m))])], format='csc')
    l = np.hstack([b, np.zeros(n)])
    u = np.hstack([b, np.ones(n)])
    return A, P