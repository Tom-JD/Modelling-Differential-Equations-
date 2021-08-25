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


def evolve(phi, f,Df, t0,y0, T,N):
    # compute y_{n+1} = phi(t_n,y_n;h) for n=1,...,N and t_i=ih
    # with h=T/N so t_1=t0 and t_{N+1}=T
    h = T/N
    y = np.zeros( [N+1,len(y0)] )
    y[0] = y0
    t = 0

    for i in range(N):
        y[i+1] = phi(f,Df, t,y[i], h)
        t = t+h

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

        dy4 = dirk(f, Df, t0, y0, h0, alpha, beta, gamma_star)
        dy5 = dirk(f, Df, t0, y0, h0, alpha, beta, gamma)

        err = np.linalg.norm(dy4 - dy5)
        h = 0.9 * h0 * (tol/err)**(1/5)
        
    return dy4, h

def adaptEvolve(phi, f, Df, t0, y0, T, hStart):
    y = np.zeros([ 1,len(y0) ])
    t = np.zeros([1])
    y[0,:] = y0
    t[0] = 0
    h = hStart
    H = [hStart]

    for _ in range(20000):
        ynew, h_new = phi(f,Df, t[-1],y[-1,:], h)
        tnew = t[-1] + h_new
        y = np.append(y, [ynew], 0)
        t = np.append(t, [tnew], 0)
        H.append(h_new)

        print(t[-1])

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
    
    # print(t[count] - t[0])
    # assert(False)

    return t[count] - t[0]

def is_done(observed, exact, tol):
    return np.abs(observed - exact) < tol

def compute(explicit):

    print("Explicit method") if explicit else print("Implicit method")

    maxh = 0.05 # maximal step size for adaptive methods

    mu = 100
    T  = 200
    exact_period = 162.844

    f  = lambda t, y: array([ y[1], mu*(1-y[0]*y[0])*y[1]-y[0] ])
    Df = lambda t, y: array([[0, 1],
                            [-2*mu*y[0]*y[1] - 1, mu * (1 - y[0] * y[0])] ])

    y0 = array([2., 0.])

    if explicit:
        N0 = 200
    else:
        N0 = 20

    tol = 1e-1
    # setup ode solver

    num_failures = 0

    if explicit:
        alpha = array([0.,    0.5,   1.])

        gamma = array([1./6., 2./3., 1./6.])

        beta  = array([ [0.,  0., 0.],
                    [0.5, 0., 0.],
                    [-1., 2., 0.] ])
        bi_solver = "RK45"
    else:
        alpha = array([1./2.,  2./3.,  1./2.,  1.    ])

        gamma = array([3./2., -3./2.,  1./2.,  1./2. ])

        beta  = array([ [1./2.,  0.,     0.,     0.   ],
                    [1./6.,  1./2.,  0.,     0.   ],
                    [-1./2., 1./2.,  1./2.,  0.   ],
                    [3./2., -3./2.,  1./2.,  1./2.] ])
        bi_solver = "Radau"

    ######################################################

    print("fixed: ")
    while True:
        if num_failures == 5:
            print("Never converged to actual value of the period")
            break

        stepper = lambda f,Df,t0,y0,h: dirk(f,Df,t0,y0,h,alpha,beta,gamma)
        y = evolve(stepper,f,Df,0,y0, T,N0*T)
        t = linspace(0,T,len(y))

        period = get_period(t, y[:, 0], 1e-5)
        print("estimate for the period is ", period)
        if is_done(period, exact_period, 1e-4) or num_failures > 5:
            print("The value of N0 such that the period was close enough was", N0)
            break
        else:
            num_failures += 1

        N0 = N0*2

    num_failures = 0
    print("\n\n")
    ######################################################

    print("build-in:")
    while 1:

        if num_failures == 7:
            print("Never actually converged to the true value of the period")
            break

        res = solve_ivp(f, [0,T], y0, method=bi_solver, max_step=maxh, rtol=tol)
        y = res.y.transpose()
        t = res.t

        period = get_period(t, y[:, 0], 1e-4)

        print("estimate for the period with the built-in method", period)

        if is_done(period, exact_period, 1e-4):
            print("the value of the tolerance such that the period was close enough was ", tol)
            break
        else:
            num_failures += 1

        tol = tol/10.

##########################################################

if __name__ == "__main__":
    compute(True)
    print("\n\n\n\n")
    compute(False)
