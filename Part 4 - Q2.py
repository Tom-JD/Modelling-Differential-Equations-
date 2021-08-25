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


def dirk(f, Df, t0, y0, h, alpha, beta, gamma):

    s  = gamma.size
    m  = y0.size
    k  = np.zeros((s, m))
    f0 = f(t0,y0)
    y  = y0.copy().astype(float)

    for i in range(s):
        ti = t0+alpha[i]*h
        yi = y0+h*sum( [beta[i,l]*k[l, :] for l in range(i)] )

        if beta[i,i]==0:
            k[i,:] = f(ti,yi)

        else:
            k[i,:] = newton( lambda d: d - f(ti, yi + h*beta[i,i]*d),
                           lambda d: np.eye(m) - h*beta[i,i]*Df(ti,yi + h*beta[i,i]*d),
                           f0, 1e-15, 1000)[0]
        
        y[:] += h*gamma[i]*k[i, :]

    assert not np.any(np.isnan(y)), "DIRK computation failed"
    return y

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
        h0 = min(h0, h_max)
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

def get_period(t, y, tol):
    """Calculates the time for the value of y to return to y_0 (within the tolerance @param:tol)"""
    y0 = y[0]
    count = y.shape[0] - 1

    while(count >= 1 and np.abs(y[count] - y0) >= tol):
        count -= 1

    return t[count] - t[0]

def is_done(observed, exact, tol):
    return np.abs(observed - exact) < tol

def compute(explicit):
    print("Explicit method") if explicit else print("Implicit method")
    maxh = 0.05 # maximal step size for adaptive methods

    mu = 100
    T  = 200
    exact_period = 162.844

    f  = lambda t,y: array([ y[1], mu*(1-y[0]*y[0])*y[1]-y[0] ])
    Df = lambda t,y: array([[0, 1],
                            [-2*mu*y[0]*y[1] - 1, mu * (1 - y[0] * y[0])] ])

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

    num_failures = 0

    print("adaptive:")
    
    while True:
        if num_failures == 5:
            print("Never actually converged to the true value of the period")
            break


        stepper = lambda f, Df, t0, y0, h0 : embeddirk(f, Df, t0, y0, h0, alpha, beta, gamma, gammaStar, p, tol, maxh)
        t, y, H = adaptEvolve(stepper, f, Df, 0, y0, T, 0.5)
        pyplot.figure(1)
        pyplot.title("Solution")
        pyplot.plot(t, y[:, 0])
        pyplot.savefig("imgs/ex_solnplot" + str(num_failures) + ".png") if explicit else pyplot.savefig("imgs/im_solnplot" + str(num_failures) + ".png")
        pyplot.clf()

        pyplot.figure(2)
        pyplot.title("Variation of h")
        pyplot.plot(t, H)
        pyplot.semilogy()
        pyplot.savefig("imgs/ex_hplot" + str(num_failures) + ".png") if explicit else pyplot.savefig("imgs/im_hplot" + str(num_failures) + ".png")
        pyplot.clf()

        period = get_period(t, y[:,0], 1e-5)
        print("estimate for the period is", period)

        if is_done(period, exact_period, 1e-4):
            print("The required tolerance for the period to be close enough was", tol)
            break
        else:
            num_failures += 1

        tol /= 10

if __name__ == "__main__":
    compute(True)
    print("\n\n\n")
    compute(False)
