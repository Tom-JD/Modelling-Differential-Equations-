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
        new_x = x - solve(jacobian, f)
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

def y_prime(_, y):
    """Computes the ODE system given in question 2.1
    the first parameter represents time"""
    return (1.5 - y) ** 2

def y_pprime(_, y):
        return -2 * (1.5 - y)

def exact(t):
    return (1 + 0.75 * t) / (1 + 0.5 * t)


def evolve(method, f, Df, t0, y0, T, N, alpha=None, beta=None, gamma=None):
    """Evolve method"""

    h = T/N
    t = t0
    y = y0
    
    count = 0

    while(count < N):
        if alpha is not None:
            phi = method(f, Df, t, y, h, alpha, beta, gamma)
        else:
            phi = method(f, Df, t, y, h)

        t = t + h
        y = y + h * phi
        count += 1
    
    error = np.linalg.norm(y - exact(T))

    return (y, error) 

def get_EOC_matrix_and_errors(method, f, DF, t0, y0, T, alpha=None, beta=None, gamma=None):
    """Computes the matrix fed into 'computeEocs' for the method 'method_func'"""
    N0 = 20

    h_err = []
    errors = []

    for _ in range(11):
        _, cur_error = evolve(method, f, DF, t0, y0, T, N0, alpha, beta, gamma)
        h_err.append([T/N0, cur_error])
        errors.append(cur_error)
        N0 *= 2

    return h_err, errors

def computeEocs(herr):
    """Computes the EOCs of a 2 x m matrix of the form [[h1, e1], ..., [hm, em]]"""
    ans = []
    for i in range(1, len(herr)):
        curEoc = log(herr[i][1]/herr[i - 1][1])/log(herr[i][0]/herr[i - 1][0])
        ans.append(curEoc)
    return ans

def main():
    """Main function"""

    heun_a = [0, 1]
    heun_b = [[0, 0],
        [1, 0]]
    heun_g = [0.5, 0.5]

    CN_a = [0, 1]
    CN_b = [[0, 0],
            [0.5, 0.5]]
    CN_g = [0.5, 0.5]

    DIRK_a = [1/3, 1]
    DIRK_b = [[1/3, 0], [1, 0]]
    DIRK_g = [0.75, 0.25]



    heun_eoc_m, heun_errs = get_EOC_matrix_and_errors(RK_phi, y_prime, y_pprime, 0, np.array([1]), 10, alpha=heun_a, beta=heun_b, gamma=heun_g)
    heun_eocs = computeEocs(heun_eoc_m)

    CN_eoc_m, CN_errs = get_EOC_matrix_and_errors(RK_phi, y_prime, y_pprime, 0, np.array([1]), 10, alpha=CN_a, beta=CN_b, gamma=CN_g)
    CN_eocs = computeEocs(CN_eoc_m)

    DIRK_eoc_m, DIRK_errs = get_EOC_matrix_and_errors(RK_phi, y_prime, y_pprime, 0, np.array([1]), 10, alpha=DIRK_a, beta=DIRK_b, gamma=DIRK_g)
    DIRK_eocs = computeEocs(DIRK_eoc_m)


    _ = plt.figure(1)
    plt.plot([row[0] for row in heun_eoc_m], [row[1] for row in heun_eoc_m], label="Heun")
    plt.plot([row[0] for row in CN_eoc_m], [row[1] for row in CN_eoc_m], label="Crank-Nicholson")
    plt.plot([row[0] for row in DIRK_eoc_m], [row[1] for row in DIRK_eoc_m], label="DIRK")
    plt.legend()
    plt.xlabel("Step size h")
    plt.ylabel("Error")
    plt.title("A graph of step size versus for error for several methods")

    __ = plt.figure(2)
    plt.loglog([row[0] for row in heun_eoc_m], [row[1] for row in heun_eoc_m], label="Heun")
    plt.loglog([row[0] for row in CN_eoc_m], [row[1] for row in CN_eoc_m], label="Crank-Nicholson")
    plt.loglog([row[0] for row in DIRK_eoc_m], [row[1] for row in DIRK_eoc_m], label="DIRK")
    plt.legend()
    plt.xlabel("Log of step size h")
    plt.ylabel("Log of error")
    plt.title("A log-log graph of step size versus for error for several methods")
    
    
    plt.show()
    


    eocs_and_errors = [[heun_eocs, heun_errs], [CN_eocs, CN_errs], [DIRK_eocs, DIRK_errs]]

    for eoc, err in eocs_and_errors:
        print("EOCs for the method: ")
        
        for e in eoc:
            print("%.6f" % e)
        
        print("\n")

        print("Errors for method: ")

        for e in err:
            print("%.6f" % e)

        print("\n\n")

main()