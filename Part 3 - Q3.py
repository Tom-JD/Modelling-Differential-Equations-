#---------------------------------------------------------------------------------------------------------------------#
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
from numpy.linalg import norm
from math import log

def newton (F, DF, x0, eps, K):
    """Uses Newton's method to solve equations of the form F(x) = 0 to within
    a tolerance of eps and max number of iterations K"""
    x = x0
    error = 10000
    count = 0
    
    while(error >= eps and count < K):
        jacobian = DF(x)
        f = F(x)
        try:
            new_x = x - solve(jacobian, f)
        except:
            print("jacobian: ")
            print(jacobian)
        error = norm(new_x - x)
        if(error < eps):
            break
        x = new_x
        count += 1
    
    return x, count

def get_k_explicit(j, f, t, y, h, alpha, beta, gamma, computed_k):
    """Computes the value of the k function at {t, y} if it is explicit"""
    t_new = t + alpha[j] * h

    s = 0

    for l in range(j):
        s += beta[j][l] * computed_k[l]
    
    y_new = y + h * s

    new_k = f(t_new, y_new)
    computed_k.append(new_k)

    return new_k

def get_k_implicit(j, f, Df, t, y, h, alpha, beta, gamma, computed_k):
    """Compute the value of the k function at {t, y} if it is implicit"""
    t_new = t + alpha[j] * h

    constant = 0

    for l in range(j):
        constant += beta[j][l] * computed_k[l]
    
    func_to_solve = lambda k_j : k_j -  f(t_new, y + h * (constant + beta[j][j] * k_j))
    deriv = lambda k_j : np.identity(k_j.shape[0]) - beta[j][j] * h * Df(t_new, y + h * (constant + beta[j][j] * k_j))

    init = computed_k[j - 1] if j > 0 else y
    k_j, count = newton(func_to_solve, deriv, init, 1e-10, 1000)

    assert count < 1000, "Newton failed with count: " + str(count)

    computed_k.append(k_j)
    return k_j



def RK_phi(f, Df, t, y, h, alpha, beta, gamma):
    """Computes the phi function at {t, y} for the RK method determined by the 
    alpha, beta, and gamma matrices"""

    computed_k = []
    for j in range(len(alpha)):
        if beta[j][j] == 0:
            get_k_explicit(j, f, t, y, h, alpha, beta, gamma, computed_k)
        else:
            get_k_implicit(j, f, Df, t, y, h, alpha, beta, gamma, computed_k)

    y_new = 0

    for k in range(len(computed_k)):
        y_new += gamma[k] * computed_k[k]

    return y_new


def rungeKutta(f, Df, t0, y0, h, alpha, beta, gamma):
    """Calculates 1 step of the RK method given by 
    the alpha, beta, and gamma matrices"""
    phi = RK_phi(f, Df, t0, y0, h, alpha, beta, gamma)
    return y0 + h * phi

#---------------------------------------------------------------------------------------------------------------------#

q = 1
def y_prime(t, X):
    """Computes the ODE system given in question 2.1
    the first parameter represents time"""
    return np.array([-q * X[0] - q * X[1], q * X[0] - q * X[1]])


def y_pprime(t, X):
    return np.array([[-q, -q], [q, -q]])


def evolve(method, f, Df, t0, y0, h, num_steps, alpha=None, beta=None, gamma=None):
    """Evolve method"""

    t = [t0]
    y = [y0]
    
    count = 0

    while(count < num_steps):
        if alpha is not None:
            phi = method(f, Df, t[-1], y[-1], h, alpha, beta, gamma)
        else:
            phi = method(f, Df, t[-1], y[-1], h)

        t.append(t[-1] + h)
        y.append(y[-1] + h * phi)
        count += 1
    
    return (t, y)


def main():
    """Main function"""

    # heun_a = [0, 1]
    # heun_b = [[0, 0],
    #     [1, 0]]
    # heun_g = [0.5, 0.5]
    # heun_bound = 1.5437 / q
    # heun_h = 0.95 * heun_bound
    # t, heun_y = evolve(RK_phi, y_prime, y_pprime, 0, np.array([1, 1]), heun_h, 10, alpha=heun_a, beta=heun_b, gamma=heun_g)

    fe_a = [0]
    fe_b = [[0]]
    fe_g = [1]
    fe_bound = 1.414213562/q
    fe_h = 1.05 * fe_bound
    t, fe_y = evolve(RK_phi, y_prime, y_pprime, 0, np.array([1, 1]), fe_h, 10, alpha=fe_a, beta=fe_b, gamma=fe_g)


    fe_y = np.array(fe_y)


    plt.figure("FE")
    plt.plot(t, fe_y[:, 0], label="y1")
    plt.plot(t, fe_y[:, 1], label="y2")
    plt.title("A graph of y_1 and y_2 versus time for the Forward-Euler method")
    plt.xlabel("Time t")
    plt.ylabel("Values of the functions")
    plt.legend()


    # heun_y = np.array(heun_y)

    # plt.figure("Heun")
    # plt.plot(t, heun_y[:, 0], label="y1")
    # plt.plot(t, heun_y[:, 1], label="y2")
    # plt.title("A graph of y_1 and y_2 versus time for Heun's method")
    # plt.xlabel("Time t")
    # plt.ylabel("Values of the functions")
    # plt.legend()
    
    
    plt.show()


main()