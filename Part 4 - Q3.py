import numpy as np
from numpy.linalg import solve
from numpy import array, diff, exp, sin, log10, linspace, floor
from scipy.integrate import solve_ivp
from matplotlib import pyplot

def newton(F,DF,x0,eps,K):
    k  = 0
    x  = x0.copy().astype(float)  # note: with x=x0 changes to x also changes to x0 with numpy arrays
    delta = np.zeros([len(x0)])
    Fx = F(x)
    while Fx.dot(Fx) > eps*eps and k<K:
        delta[:] = solve(DF(x), Fx)
        x[:] -= delta[:]  # don't construct a new vector in each step - they could be large
        Fx = F(x)
        k += 1
    return [x,k]

####################################################################
####################################################################

def embeddirk(f, Df, t0, y0, h0, alpha, beta, gamma, gamma_star, p, tol, h_max):
    # implemented embeded rk method here - start with the dirk method given
    # below (copy it here since the idea is to only compute stages once, so
    # you can't call the 'dirk' function directly.
    # return values are the new value y_1 and time step h used
    err = 2 * tol

    while (err > tol):

        s  = gamma.size
        m  = y0.size
        k  = np.zeros((s, m))
        f0 = f(t0,y0)
        dy4  = y0.copy().astype(float)
        dy5  = y0.copy().astype(float)

        # calculating the 2 approximations
        for i in range(s):
            ti = t0 + alpha[i] * h0
            yi = y0 + h0 * sum([beta[i, l] * k[l, :] for l in range(i)])

            if beta[i,i]==0:
                k[i,:] = f(ti,yi)
            else:
                k[i,:] = newton( lambda d: d - f(ti, yi + h0 * beta[i,i] * d),
                            lambda d: np.eye(m) - h0 * beta[i,i] * Df(ti,yi + h0 * beta[i,i] * d),
                            f0, 1e-15, 1000)[0]
            
            dy4[:] += h0 * gamma_star[i]*k[i, :]
            dy5[:] += h0 * gamma[i]*k[i, :]

        assert not np.any(np.isnan(dy4)), "DIRK computation failed for method of order p"
        assert not np.any(np.isnan(dy5)), "DIRK computation failed for method of order p + 1"

        err = np.linalg.norm(dy4 - dy5)

        if err < 1e-15:
            h0 = h_max
            break

        h0 = 0.9 * h0 * (tol/err)**(1/p)
        assert h0 >= 1e-10, "Scheme not working"
        
    return dy4, h0

def adaptEvolve(phi, f, Df, t0, y0, T, hStart):
    y = np.zeros([ 1,len(y0) ])
    t = np.zeros([1])
    y[0,:] = y0
    t[0] = 0
    h = hStart
    H = [hStart]

    while t[-1] < T:
        ynew, h_new = phi(f,Df, t[-1],y[-1,:], h)
        tnew = t[-1] + h_new
        y = np.append(y, [ynew], 0)
        t = np.append(t, [tnew], 0)
        H.append(h_new)

        if tnew > T:
            break

    return t, y, H

####################################################################
####################################################################

def compute(problem, explicit):

    if explicit:
        print("problem ",problem," using explicit methods")
    else:
        print("problem ",problem," using implicit methods")

    maxh = 0.05 # maximal step size for adaptive methods

    T  = 100

    f  = lambda t,y: array([2 - 9 * y[0] + y[0] ** 2 * y[1], 8 * y[0] - y[0] ** 2 * y[1]])
    Df = lambda t,y: array([[-9 + 2 * y[0] * y[1], y[0] ** 2], [8 - 2 * y[0] * y[1], -y[0] ** 2]])

    y0 = array([2., 0.])

    tol = 1e-1

    if explicit:
        alpha = array([0.,    0.5,   1.])

        gamma = array([1./6., 2./3., 1./6.])

        beta  = array([ [0.,  0., 0.],
                    [0.5, 0., 0.],
                    [-1., 2., 0.] ])

        gammaStar = array([0,     1.,    0.])
        p = 5
    else:
        alpha = array([1./2.,  2./3.,  1./2.,  1.    ])

        gamma = array([3./2., -3./2.,  1./2.,  1./2. ])

        beta  = array([ [1./2.,  0.,     0.,     0.   ],
                    [1./6.,  1./2.,  0.,     0.   ],
                    [-1./2., 1./2.,  1./2.,  0.   ],
                    [3./2., -3./2.,  1./2.,  1./2.] ])

        gammaStar = array([1.,     0.,     0.,     0.    ])
        p = 5

    print("adaptive:")
    
    stepper = lambda f, Df, t0, y0, h0 : embeddirk(f, Df, t0, y0, h0, alpha, beta, gamma, gammaStar, p, tol, maxh)
    t, y, _ = adaptEvolve(stepper, f, Df, 0, y0, T, 0.05)

    pyplot.figure("X")
    pyplot.title("X versus time")
    pyplot.xlabel("time")
    pyplot.ylabel("X")
    pyplot.plot(t, y[:, 0])

    pyplot.figure("Y")
    pyplot.title("Y versus time")
    pyplot.xlabel("time")
    pyplot.ylabel("Y")
    pyplot.plot(t, y[:, 1])

    pyplot.figure("X and Y")
    pyplot.title("X and Y versus time")
    pyplot.xlabel("time")
    pyplot.ylabel("Chemicals")
    pyplot.plot(t, y)

    pyplot.figure("Phase portrait")
    pyplot.title("Phase portrait")
    pyplot.xlabel("X")
    pyplot.ylabel("Y")
    pyplot.plot(y[:, 0], y[:, 1])
    pyplot.show()

if __name__ == "__main__":
    compute(1, True)
    # compute(1,False)
