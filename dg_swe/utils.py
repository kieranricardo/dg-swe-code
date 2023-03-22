import numpy as np
from matplotlib import pyplot as plt
import torch


def gll(N, iterative=False):
    """
    Returns GLL (Gauss Lobato Legendre module with collocation points and
    weights)
    """
    # Initialization of integration weights and collocation points
    # [xi, weights] =  gll(N)
    # Values taken from Diploma Thesis Bernhard Schuberth
    if N == 1:
        xi = [-1, 1]
        weights = [0.5, 0.5]
    elif N == 2:
        xi = [-1.0, 0.0, 1.0]
        weights = [0.33333333, 1.33333333, 0.33333333]
    elif N == 3:
        xi = [-1.0, -0.447213595499957, 0.447213595499957, 1.0]
        weights = [0.1666666667, 0.833333333, 0.833333333, 0.1666666666]
    elif N == 4:
        xi = [-1.0, -0.6546536707079772, 0.0, 0.6546536707079772, 1.0]
        weights = [0.1, 0.544444444, 0.711111111, 0.544444444, 0.1]
    elif N == 5:
        xi = [-1.0, -0.7650553239294647, -0.285231516480645, 0.285231516480645,
              0.7650553239294647, 1.0]
        weights = [0.0666666666666667,  0.3784749562978470,
                   0.5548583770354862, 0.5548583770354862, 0.3784749562978470,
                   0.0666666666666667]
    elif N == 6:
        xi = [-1.0, -0.8302238962785670, -0.4688487934707142, 0.0,
              0.4688487934707142, 0.8302238962785670, 1.0]
        weights = [0.0476190476190476, 0.2768260473615659, 0.4317453812098627,
                   0.4876190476190476, 0.4317453812098627, 0.2768260473615659,
                   0.0476190476190476]
    elif N == 7:
        xi = [-1.0, -0.8717401485096066, -0.5917001814331423,
              -0.2092992179024789, 0.2092992179024789, 0.5917001814331423,
              0.8717401485096066, 1.0]
        weights = [0.0357142857142857, 0.2107042271435061, 0.3411226924835044,
                   0.4124587946587038, 0.4124587946587038, 0.3411226924835044,
                   0.2107042271435061, 0.0357142857142857]
    elif N == 8:
        xi = [-1.0, -0.8997579954114602, -0.6771862795107377,
              -0.3631174638261782, 0.0, 0.3631174638261782,
              0.6771862795107377, 0.8997579954114602, 1.0]
        weights = [0.0277777777777778, 0.1654953615608055, 0.2745387125001617,
                   0.3464285109730463, 0.3715192743764172, 0.3464285109730463,
                   0.2745387125001617, 0.1654953615608055, 0.0277777777777778]
    elif N == 9:
        xi = [-1.0, -0.9195339081664589, -0.7387738651055050,
              -0.4779249498104445, -0.1652789576663870, 0.1652789576663870,
              0.4779249498104445, 0.7387738651055050, 0.9195339081664589, 1.0]
        weights = [0.0222222222222222, 0.1333059908510701, 0.2248893420631264,
                   0.2920426836796838, 0.3275397611838976, 0.3275397611838976,
                   0.2920426836796838, 0.2248893420631264, 0.1333059908510701,
                   0.0222222222222222]
    elif N == 10:
        xi = [-1.0, -0.9340014304080592, -0.7844834736631444,
              -0.5652353269962050, -0.2957581355869394, 0.0,
              0.2957581355869394, 0.5652353269962050, 0.7844834736631444,
              0.9340014304080592, 1.0]
        weights = [0.0181818181818182, 0.1096122732669949, 0.1871698817803052,
                   0.2480481042640284, 0.2868791247790080, 0.3002175954556907,
                   0.2868791247790080, 0.2480481042640284, 0.1871698817803052,
                   0.1096122732669949, 0.0181818181818182]
    elif N == 11:
        xi = [-1.0, -0.9448992722228822, -0.8192793216440067,
              -0.6328761530318606, -0.3995309409653489, -0.1365529328549276,
              0.1365529328549276, 0.3995309409653489, 0.6328761530318606,
              0.8192793216440067, 0.9448992722228822, 1.0]
        weights = [0.0151515151515152, 0.0916845174131962, 0.1579747055643701,
                   0.2125084177610211, 0.2512756031992013, 0.2714052409106962,
                   0.2714052409106962, 0.2512756031992013, 0.2125084177610211,
                   0.1579747055643701, 0.0916845174131962, 0.0151515151515152]
    elif N == 12:
        xi = [-1.0, -0.9533098466421639, -0.8463475646518723,
              -0.6861884690817575, -0.4829098210913362, -0.2492869301062400,
              0.0, 0.2492869301062400, 0.4829098210913362,
              0.6861884690817575, 0.8463475646518723, 0.9533098466421639,
              1.0]
        weights = [0.0128205128205128, 0.0778016867468189, 0.1349819266896083,
                   0.1836468652035501, 0.2207677935661101, 0.2440157903066763,
                   0.2519308493334467, 0.2440157903066763, 0.2207677935661101,
                   0.1836468652035501, 0.1349819266896083, 0.0778016867468189,
                   0.0128205128205128]
    else:
        xi, weights = gLLNodesAndWeights(N + 1)

    if iterative:
        xi, weights = gLLNodesAndWeights(N + 1)
    return np.array(xi), np.array(weights)


def lagrange(N, i, x, xi):
    """
    Function to calculate  Lagrange polynomial for order N and polynomial
    i[0, N] at location x.
    """

    # [xi, weights] = gll(N)
    fac = 1
    for j in range(-1, N):
        if j != i:
            fac = fac * ((x - xi[j + 1]) / (xi[i + 1] - xi[j + 1]))
    return fac


def lagrange1st(N, xi):
    """
    # Calculation of 1st derivatives of Lagrange polynomials
    # at GLL collocation points
    # out = legendre1st(N)
    # out is a matrix with columns -> GLL nodes
    #                        rows -> order
    """
    out = np.zeros([N+1, N+1])

    # [xi, w] = gll(N)

    # initialize dij matrix (see Funaro 1993 or Diploma thesis Bernhard
    # Schuberth)
    d = np.zeros([N + 1, N + 1])

    for i in range(-1, N):
        for j in range(-1, N):
            if i != j:
                d[i + 1, j + 1] = legendre(N, xi[i + 1]) / \
                    legendre(N, xi[j + 1]) * 1.0 / (xi[i + 1] - xi[j + 1])
            if i == -1:
                if j == -1:
                    d[i + 1, j + 1] = -1.0 / 4.0 * N * (N + 1)
            if i == N-1:
                if j == N-1:
                    d[i + 1, j + 1] = 1.0 / 4.0 * N * (N + 1)

    # Calculate matrix with 1st derivatives of Lagrange polynomials
    for n in range(-1, N):
        for i in range(-1, N):
            sum = 0
            for j in range(-1, N):
                sum = sum + d[i + 1, j + 1] * lagrange(N, n, xi[j + 1], xi)

            out[n + 1, i + 1] = sum
    return(out)


def legendre(N, x):
    """
    Returns the value of Legendre Polynomial P_N(x) at position x[-1, 1].
    """
    P = np.zeros(2 * N)

    if N == 0:
        P[0] = 1
    elif N == 1:
        P[1] = x
    else:
        P[0] = 1
        P[1] = x
    for i in range(2, N + 1):
        P[i] = (1.0 / float(i)) * ((2 * i - 1) * x * P[i - 1] - (i - 1) *
                                   P[i - 2])

    return(P[N])



# Ciardelli, C., Bozdağ, E., Peter, D., and van der Lee, S., 2021.
# #SphGLLTools: A toolbox for visualization of large seismic model files
# #based on 3D spectral-element meshes. Computer & Geosciences.
# #https://doi.org/10.1016/j.cageo.2021.105007.


def gLLNodesAndWeights(n, epsilon=1e-15):
    """
    Computes the GLL nodes and weights
    """
    if n < 2:

        print('Error: n must be larger than 1')

    else:

        x = np.empty(n)
        w = np.empty(n)

        x[0] = -1;
        x[n - 1] = 1
        w[0] = w[0] = 2.0 / ((n * (n - 1)));
        w[n - 1] = w[0];

        n_2 = n // 2

        for i in range(1, n_2):

            xi = (1 - (3 * (n - 2)) / (8 * (n - 1) ** 3)) * \
                 np.cos((4 * i + 1) * np.pi / (4 * (n - 1) + 1))

            error = 1.0

            while error > epsilon:
                y = dLgP(n - 1, xi)
                y1 = d2LgP(n - 1, xi)
                y2 = d3LgP(n - 1, xi)

                dx = 2 * y * y1 / (2 * y1 ** 2 - y * y2)

                xi -= dx
                error = abs(dx)

            x[i] = -xi
            x[n - i - 1] = xi

            w[i] = 2 / (n * (n - 1) * lgP(n - 1, x[i]) ** 2)
            w[n - i - 1] = w[i]

        if n % 2 != 0:
            x[n_2] = 0;
            w[n_2] = 2.0 / ((n * (n - 1)) * lgP(n - 1, np.array(x[n_2])) ** 2)

    return x, w


def dLgP (n, xi):
  """
  Evaluates the first derivative of P_{n}(xi)
  """
  return n * (lgP (n - 1, xi) - xi * lgP (n, xi))\
           / (1 - xi ** 2)

def d2LgP (n, xi):
  """
  Evaluates the second derivative of P_{n}(xi)
  """
  return (2 * xi * dLgP (n, xi) - n * (n + 1)\
                                    * lgP (n, xi)) / (1 - xi ** 2)

def d3LgP (n, xi):
  """
  Evaluates the third derivative of P_{n}(xi)
  """
  return (4 * xi * d2LgP (n, xi)\
                 - (n * (n + 1) - 2) * dLgP (n, xi)) / (1 - xi ** 2)


def lgP(n, xi):
    """
    Evaluates P_{n}(xi) using an iterative algorithm
    """
    if n == 0:

        return np.ones(xi.size)

    elif n == 1:

        return xi

    else:

        fP = np.ones(xi.size);
        sP = xi.copy();
        nP = np.empty(xi.size)

        for i in range(2, n + 1):
            nP = ((2 * i - 1) * xi * sP - (i - 1) * fP) / i

            fP = sP;
            sP = nP

        return nP


def analyze_conserved_quantities(solver, tend, label, dt=None):

    entropy = [solver.integrate(solver.entropy())]
    mass = [solver.integrate(solver.h)]
    enstrophy = [solver.integrate(solver.enstrophy())]
    vorticity = [solver.integrate(solver.vorticity())]

    # q = solver.q()
    #
    # grad_q_norm = [solver.grad_q_norm()]
    # div_F_norm = [solver.div_F_norm()]

    times = [0]

    while solver.time <= tend:
        solver.time_step(dt=dt, order=3)
        entropy.append(solver.integrate(solver.entropy()))
        mass.append(solver.integrate(solver.h))
        enstrophy.append(solver.integrate(solver.enstrophy()))
        vorticity.append(solver.integrate(solver.vorticity()))

        # grad_q_norm.append(solver.grad_q_norm())
        # div_F_norm.append(solver.div_F_norm())

        times.append(solver.time)

        # q = solver.continuous_q()
        # q2 = q ** 2
        # dq2dx = torch.einsum('fgcd,abcd->fgab', q2, self.grad[..., 1])
        # dq2dy = torch.einsum('fgcd,abcd->fgab', q2, self.grad[..., 0])
        # int_1 = 2 * solver.integrate(dqdx * q * (u * h) + dqdy * q * (v * h))
        # int_2 = solver.integrate(dq2dx * (u * h) + dq2dy * (v * h))
        # chain_rule_error = (int_1 - int_2)

    times = np.array(times)
    tunit = ''
    if times.max() >= 3600 * 24:
        times /= 3600 * 24
        tunit += ' (days)'

    entropy = np.array(entropy)
    mass = np.array(mass)
    enstrophy = np.array(enstrophy)
    vorticity = np.array(vorticity)

    plt.figure(1, figsize=(7, 7))

    # fig, axs = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
    plt.suptitle("Conservation errors")

    ax = plt.subplot(2, 2, 1)
    # ax = axs[0][0]
    ax.set_ylabel("Energy error (normalized)")
    # ax.set_xlabel("Time" + tunit)
    ax.set_xticks([], [])
    ax.plot(times, (entropy - entropy[0]) / entropy[0], label=label)
    ax.set_yscale('symlog', linthresh=1e-15)
    ax.grid(True, which='both')

    ax = plt.subplot(2, 2, 2)
    # ax = axs[0][1]
    ax.set_ylabel("Mass error (normalized)")
    # ax.set_label("Time" + tunit)
    ax.set_xticks([], [])
    ax.plot(times, (mass - mass[0]) / mass[0], label=label)
    ax.set_yscale('symlog', linthresh=1e-16)
    ax.grid(True, which='both')

    ax = plt.subplot(2, 2, 3)
    # ax = axs[1][0]
    ax.set_ylabel("Enstrophy error (normalized)")
    ax.set_xlabel("Time" + tunit)
    ax.plot(times, (enstrophy - enstrophy[0]) / enstrophy[0], label=label)
    ax.set_yscale('symlog', linthresh=1e-15)
    ax.grid(True, which='both')

    ax = plt.subplot(2, 2, 4)
    # ax = axs[1][1]
    plt.ylabel("Vorticity error (normalized)")
    plt.xlabel("Time" + tunit)
    plt.plot(times, (vorticity - vorticity[0]) / vorticity[0], label=label)
    ax.set_yscale('symlog', linthresh=1e-16)
    ax.grid(True, which='both')

    plt.legend()
    plt.tight_layout()


def cross_product(vec1, vec2):
    out = []
    out.append(vec1[1] * vec2[2] - vec1[2] * vec2[1])
    out.append(vec1[2] * vec2[0] - vec1[0] * vec2[2])
    out.append(vec1[0] * vec2[1] - vec1[1] * vec2[0])
    return out


def dot_product(vec1, vec2):
    out = sum(a * b for a, b in zip(vec1, vec2))
    return out


def norm_L2(vec):
    out = torch.sqrt(sum(a ** 2 for a in vec))
    return out


class Interpolate:

    def __init__(self, p1, p2):

        # goes from polynomial order p1
        # to polynomial order p2

        [xis_1d, _] = gll(p2, iterative=True)
        [etas_1d, _] = gll(p2, iterative=True)
        xis, etas = np.meshgrid(xis_1d, etas_1d)

        [gll_xs, _] = gll(p1, iterative=True)

        self.transform = np.zeros((p1 + 1, p1 + 1, p2 + 1, p2 + 1))

        for i, y_ in enumerate(gll_xs):
            for j, x_ in enumerate(gll_xs):
                    y_data = np.zeros_like(gll_xs)
                    y_data[i] = 1.0
                    y_poly = lagrange(gll_xs, y_data)

                    x_data = np.zeros_like(gll_xs)
                    x_data[j] = 1.0
                    x_poly = lagrange(gll_xs, x_data)

                    self.transform[i, j] = x_poly(xis) * y_poly(etas)

        self.tnsr = torch.from_numpy(self.transform)
    def interpolate(self, data):

        return np.einsum('abcd,cdef->abef', data, self.transform)

    def torch_interpolate(self, data):

        return torch.einsum('abcd,cdef->abef', data, self.tnsr)