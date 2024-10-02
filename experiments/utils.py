import cvxpy as cp
import numpy as np
import pandas as pd
from collections import namedtuple
from tqdm import trange


Problem = namedtuple('Problem',    [
                                    'B_init',
                                    'I_init',
                                    'R_init',
                                    'rho_B', 
                                    'rho_I', 
                                    'rho_R', 
                                    'kappa', 
                                    'beta', 
                                    'eta', 
                                    'T',
                                    'problem',
                                    'objective',
                                    'constraints',
                                    'c',
                                    'B',
                                    'I',
                                    'R',
                                    'b',
                                    'i',
                                    'r',
                                    'tau',
                                    'omega',
                                    'phi_',
                                    'phi',
                                    ])

def get_retirement_problem(B_init, I_init, R_init, rho_B, rho_I, rho_R, kappa, beta, eta, T):
    """
    param B_init: initial brokerage account balance
    param I_init: initial IRA account balance
    param R_init: initial Roth IRA account balance
    param rho_B: inflation adjusted return on brokerage account
    param rho_I: inflation adjusted return on IRA account
    param rho_R: inflation adjusted return on Roth IRA account
    param kappa: vector of fractions of the IRA account value that must be
    withdrawn each year
    param beta: vector of tax brackets
    param eta: vector of tax rates
    param T: number of years left until death

    returns:
    """

    ### Variables   

    # Balances
    c = cp.Variable(nonneg=True)
    B = cp.Variable(T+1, nonneg=True)
    I = cp.Variable(T+1, nonneg=True)
    R = cp.Variable(T+1, nonneg=True)

    # Withdrawals
    b = cp.Variable(T)
    i = cp.Variable(T, nonneg=True)
    r = cp.Variable(T)

    # Taxes
    tau = cp.Variable(T) # taxes paid
    omega = i + (B[:-1] - b) * (rho_B - 1) # taxable income
    _phi = cp.vstack([eta[0] * omega] + [
         (eta[i] - eta[i-1]) * cp.pos(omega - beta[i-1]) for i in range(1, len(beta))
    ]) # tax owed
    phi = _phi.sum(axis=0)

    ### Objective
    objective = cp.Maximize(c)

    ### Constraints
    constraints = [
        B[0] == B_init, I[0] == I_init, R[0] == R_init,
        b <= B[:-1], i <= I[:-1], r <= R[:-1],
        B[1:] == (B[:-1] - b) * rho_B, I[1:] == (I[:-1] - i) * rho_I, R[1:] == (R[:-1] - r) * rho_R,
        b + i + r == c + tau,
        i >= cp.multiply(kappa, I[:-1]) + cp.pos(-r),
        tau >= phi,
        ]
    
    problem = cp.Problem(objective, constraints)

    return Problem(
        B_init=B_init,
        I_init=I_init,
        R_init=R_init,
        rho_B=rho_B,
        rho_I=rho_I,
        rho_R=rho_R,
        kappa=kappa,
        beta=beta,
        eta=eta,
        T=T,
        problem=problem,
        objective=objective,
        constraints=constraints,
        c=c,
        B=B,
        I=I,
        R=R,
        b=b,
        i=i,
        r=r,
        tau=tau,
        omega=omega,
        phi_=_phi,
        phi=phi,
    )

Backtest = namedtuple('Backtest', [
                                    'cash',
                                    'returns',
                                    'inflations',
                                    'treasuries',
                                    'B',
                                    'I',
                                    'R',
                                    'b',
                                    'i',
                                    'r',
                                    ])

def mean_reverting_walk(mu, sigma, n, theta=0.1):

    X = np.zeros(n)
    X[0] = mu
    epsilon = np.random.normal(0, sigma, n)
    for t in range(1, n):
        X[t] = X[t-1] + theta * (mu - X[t-1]) + epsilon[t]
    return X

def gmm_sample(gmm, component=None):
    """
    param gmm: GaussianMixtureModel

    returns: a sample from the GMM
    """

    component = component or np.random.choice(gmm.n_components, p=gmm.weights_)

    return np.random.normal(gmm.means_[component], np.sqrt(gmm.covariances_[component]))


def run_backtest(B_init, I_init, R_init, rho_B, rho_I, rho_R, age_start, age_end, data, gmm_ret, gmm_inflation, seed=None):
    """
    param age_start: starting age
    param age_end: ending age
    param data: data tuple
    param gmm_ret: gmm for returns
    param gmm_inflation: gmm for inflation

    returns: 
    """
    if seed is not None:
        np.random.seed(seed)

    cash = pd.Series(index=range(age_start, age_end+1), dtype=float)
    returns = pd.Series(index=range(age_start, age_end+1), dtype=float)
    inflations = pd.Series(index=range(age_start, age_end+1), dtype=float)
    treasuries = pd.Series(index=range(age_start, age_end+1), dtype=float)
    B = pd.Series(index=range(age_start, age_end+1), dtype=float)
    I = pd.Series(index=range(age_start, age_end+1), dtype=float)
    R = pd.Series(index=range(age_start, age_end+1), dtype=float)
    b = pd.Series(index=range(age_start, age_end), dtype=float)
    i = pd.Series(index=range(age_start, age_end), dtype=float)
    r = pd.Series(index=range(age_start, age_end), dtype=float)

    simulated_returns = np.array([gmm_sample(gmm_ret) for _ in range(age_end-age_start)])
    simulated_inflation = mean_reverting_walk(data.inflation.mean(), data.inflation.std() / np.sqrt(10), age_end-age_start)
    simulated_treasury = mean_reverting_walk(data.treasury.mean(), data.treasury.std() / np.sqrt(10), age_end-age_start)

    for age in range(age_start, age_end):
        B.loc[age] = B_init
        I.loc[age] = I_init
        R.loc[age] = R_init

        T = age_end - age 

        # get the problem
        problem = get_retirement_problem(B_init, I_init, R_init, rho_B, rho_I, rho_R, data.kappa.loc[age:age_end-1], data.beta, data.eta, T)

        try:
            problem.problem.solve()
        except:
            print(B_init, I_init, R_init)
            print(ret, inflation)
            assert False

        # update the balances
        ret = 0.6 * simulated_returns[age-age_start] + 0.4 * simulated_treasury[age-age_start]
        inflation = simulated_inflation[age-age_start]
        # treasury = simulated_treasury[age-age_start]

        ret_adj = ret - inflation

        # ret = 0.05
        # inflation = 0

        B_interest = (B_init - problem.b.value[0]) * ret_adj
        I_interest = (I_init - problem.i.value[0]) * ret_adj
        R_interest = (R_init - problem.r.value[0]) * ret_adj

        # Realized taxes
        omega_realized = problem.i.value[0] + np.maximum(B_interest, 0) # taxable income XXX maximum or tax deduction?
        taxes_realized = np.sum([problem.eta[0] * omega_realized] + [
            (problem.eta[i] - problem.eta[i-1]) * np.maximum(omega_realized - problem.beta[i-1], 0) for i in range(1, len(problem.beta))
        ]) # tax owed

        cash.loc[age] = problem.c.value + problem.tau.value[0] - taxes_realized # realized tax correction
        returns.loc[age] = ret
        inflations.loc[age] = inflation
        treasuries.loc[age] = simulated_treasury[age-age_start]
        b.loc[age] = problem.b.value[0]
        i.loc[age] = problem.i.value[0]
        r.loc[age] = problem.r.value[0]

        B_init += B_interest - problem.b.value[0]
        I_init += I_interest - problem.i.value[0]
        R_init += R_interest - problem.r.value[0]

    B.loc[age_end] = B_init
    I.loc[age_end] = I_init
    R.loc[age_end] = R_init

    return Backtest(cash, returns, inflations, treasuries, B, I, R, b, i, r)



