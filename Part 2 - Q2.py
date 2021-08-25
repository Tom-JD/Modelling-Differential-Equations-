from math import log
import numpy as np
from math import sqrt
from math import exp
from math import cos, sin
import matplotlib.pyplot as plt


def forwardEuler(f, Df, tn, yn, h):
    """Computes 1 step of the Forward Euler method"""
    return yn + h * f(tn, yn)


def method_2(f, Df, tn, yn, h):
    """Computes 1 step of the second method as detailed
    in Q2.1 caching the value of f(tn, yn) for efficiency"""
    val = f(tn, yn)
    return yn + (h / 2) * (val + f(tn + h, yn + h * val))


def evolve(phi, f, Df, t0, y0, T, N):
    """Solves the IVP y(t) = f(y(t)) with y(t0) = y0"""
    y = [y0]
    h = T / N
    tn = t0

    while (tn < T):
        curVal = phi(f, Df, tn, y[-1], h)
        tn += h
        y.append(curVal)

    return y


def computeEocs(herr):
    """Computes the EOCs of a 2 x m matrix of the form [[h1, e1], ..., [hm, em]]"""
    ans = []
    for i in range(1, len(herr)):
        curEoc = log(herr[i][1] / herr[i - 1][1]) / log(herr[i][0] / herr[i - 1][0])
        ans.append(curEoc)
    return ans


def y_prime(t, y):
    if (t < 1 / sqrt(2)):
        return -y
    else:
        return y


def dy_prime(t, y):
    if (t < 1 / sqrt(2)):
        return -1
    else:
        return 1


def exact_soln(t):
    if (t < 1 / sqrt(2)):
        return exp(-t)
    else:
        return exp(t - sqrt(2))


def compute_eoc_matrix_and_errors(method_func):
    """Computes the matrix fed into 'computeEocs' for the method 'method_func'"""
    N0 = 2000
    T = 1
    t0 = 0
    y0 = 1

    eoc_m = []
    exact_val = exact_soln(1)
    errors = []

    # Computing EOCs for forwardEuler
    for _ in range(0, 11):
        y = evolve(method_func, y_prime, dy_prime, t0, y0, T, N0)
        cur_error = abs(exact_val - y[-1])
        errors.append(cur_error)
        cur_h = T / N0
        eoc_m.append([cur_h, cur_error])
        N0 *= 2

    return eoc_m, errors


def main():
    """Main function"""
    fe_EOCs, fe_errors = compute_eoc_matrix_and_errors(forwardEuler)
    m2_EOCs, m2_errors = compute_eoc_matrix_and_errors(method_2)

    print("EOCs for the Forward Euler method")
    for eoc in computeEocs(fe_EOCs):
        print("%.6f" % eoc)

    print()
    print("Errors for the Forward Euler method")
    for err in fe_errors:
        print("%.6f" % err)

    print("\n\n")

    print("EOCs for the Heun's method")
    for eoc in computeEocs(m2_EOCs):
        print("%.6f" % eoc)
    print()
    print("Errors for the Heun's method")
    for err in m2_errors:
        print("%.6f" % err)


main()