from math import log
from numpy.linalg import solve
from numpy.linalg import norm
from numpy import identity
from numpy import array
from numpy import linspace
from numpy import abs


def newton(F, DF, x0, eps, K):
    """Uses Newton's method to solve equations of the form F(x) = 0 to within
    a tolerance of eps and max number of iterations K"""
    x = x0
    error = 10000
    count = 0

    while (error >= eps and count < K):
        jacobian = DF(x)
        f = F(x)
        new_x = x - solve(jacobian, f)
        error = norm(new_x - x)
        if (error < eps):
            break
        x = new_x
        count += 1

    return x, count


def get_y1_BE(F, DF, t0, y0, h):
    func = lambda y: y - h * F(t0 + h, y) - y0
    deriv = lambda y: identity(y0.shape[0]) - h * DF(t0 + h, y)
    ans, count = newton(func, deriv, y0, 10 ** -15, 100)
    assert count < 100, "Newton's method did not converge"
    return ans


def get_y1_CN(F, DF, t0, y0, h):
    func = lambda y: y - (h / 2) * (F(t0, y0) + F(t0 + h, y)) - y0
    deriv = lambda y: identity(y0.shape[0]) - (h / 2) * (DF(t0 + h, y))
    ans, count = newton(func, deriv, y0, 10 ** -15, 100)
    assert count < 100, "Newton's method did not converge"
    return ans


def backwardEuler(f, Df, t0, y0, h):
    """Computes 1 step of the Forward Euler method"""
    return get_y1_BE(f, Df, t0, y0, h)


def CrankNicholson(f, Df, t0, y0, h):
    """Computes 1 step of the Crank Nicholson method"""
    return get_y1_CN(f, Df, t0, y0, h)


def y_prime(t, y):
    """ODE system first parameter is time t"""
    return array([(1.5 - y) ** 2])


def dy_prime(t, y):
    deriv = array([-2 * (1.5 - y)])
    return deriv


def exact(t):
    """Exact solution to the ODE with c = 1.5"""
    return (1 + 0.75 * t) / (1 + t * 0.5)


def evolve(method, F, DF, t0, y0, T, N):
    """Solves the ODE system with the required method"""
    t = t0
    y = y0
    h = T / N

    approx = []

    exact_val = exact(T)
    count = 0

    while (count < N):
        count += 1
        y1 = method(F, DF, t, y, h)
        approx.append(y1)
        y = y1

    error = norm(y[-1] - exact_val)
    return approx, error


def get_EOC_matrix_and_errors(method, f, DF, t0, y0, T):
    """Computes the matrix fed into 'computeEocs' for the method 'method_func'"""
    N0 = 20

    h_err = []
    errors = []

    for _ in range(11):
        _, cur_error = evolve(method, f, DF, t0, y0, T, N0)
        h_err.append([T / N0, cur_error])
        errors.append(cur_error)
        N0 *= 2

    return h_err, errors


def computeEocs(herr):
    """Computes the EOCs of a 2 x m matrix of the form [[h1, e1], ..., [hm, em]]"""
    ans = []
    for i in range(1, len(herr)):
        curEoc = log(herr[i][1] / herr[i - 1][1]) / log(herr[i][0] / herr[i - 1][0])
        ans.append(curEoc)
    return ans


def main():
    eoc_m, errors = get_EOC_matrix_and_errors(backwardEuler, y_prime, dy_prime, 0, array([1]), 10)
    eocs = computeEocs(eoc_m)

    print("EOCs for Backward Euler: ")

    for eoc in eocs:
        print("%.6f" % eoc)

    print("\n\n")

    print("Errors for Backward Euler: ")

    for err in errors:
        print("%.6f" % err)

    print("\n\n\n\n")

    CN_eoc_m, CN_errors = get_EOC_matrix_and_errors(CrankNicholson, y_prime, dy_prime, 0, array([1]), 10)
    CN_eocs = computeEocs(CN_eoc_m)

    print("EOCs for Crank Nicholson: ")

    for eoc in CN_eocs:
        print("%.6f" % eoc)

    print("\n\n")

    print("Errors for Crank Nicholson: ")

    for err in CN_errors:
        print("%.6f" % err)


main()