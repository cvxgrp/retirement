import cvxpy as cp
import numpy as np
from collections import namedtuple

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
                                    'q',
                                    'B',
                                    'I',
                                    'R',
                                    'b',
                                    'i',
                                    'r',
                                    'i_c',
                                    'i_d',
                                    'i_w',
                                    'r_c',
                                    'r_d',
                                    'r_w',
                                    'tau',
                                    'brokerage_tax',
                                    'omega',
                                    'phi',
                                    'e',
                                    'a',
                                    'l',
                                    'c0',
                                    ])


def _regularizer(c, c0, gamma):
    assert gamma >= 0, "gamma must be nonnegative"

    if gamma == 0:
        return 0
    else:
        return gamma * cp.pos(c0-c)

def _utility(q):
    return q
    
def _objective(c, q, c0, gamma):

    if c0 is not None:
        return cp.Maximize(_utility(q) - _regularizer(c, c0, gamma))
    else:
        return cp.Maximize(_utility(q))
    
def capital_gain_tax(omega_b, capital_tax_rate):
    """
    parameters
    ----------
    b: cp.Variable
        amount of money withdrawn from brokerage account
    price: float
        price of the shares
    basis: float
        average basis of the shares
    omega: cp.Variable
        taxable income
    brackets: list
        tax brackets
    
    returns
    -------
    tax: cp.Expression
        tax paid on the withdrawal
    """

    ### XXX: heuristic
    return capital_tax_rate * omega_b

def get_retirement_problem(B_init, I_init, R_init, basis, price, rho_B, rho_I, rho_R, kappa, beta, eta, e, a, l, T, d_max=8, c0=None, gamma=1, no_brokerage=False, deterministic=False, capital_tax_rate=0.15): #, tax_correction=0):
    """
    param B_init: initial brokerage account balance
    param I_init: initial IRA account balance
    param R_init: initial Roth IRA account balance
    param basis: average basis of the brokerage account
    param rho_B: inflation adjusted return on brokerage account
    param rho_I: inflation adjusted return on IRA account
    param rho_R: inflation adjusted return on Roth IRA account
    param kappa: vector of fractions of the IRA account value that must be
    withdrawn each year
    param beta: vector of tax brackets
    param eta: vector of tax rates
    param e: vector of earned income
    param a: vector of additional income
    param l: vector of liabilities
    param T: number of years left until death
    param d_max: maximum amount that can be contributed to an IRA and Roth IRA
    param c0: target consumption
    param lmbda: bequest priority utility parameter
    param rho: CRRA utility parameter
    param gamma1: regularization parameter
    param gamma2: regularization parameter

    returns:
    """

    assert len(e) == T, "e must be a vector of length T"
    assert len(a) == T, "a must be a vector of length T"
    assert len(l) == T, "l must be a vector of length T"
    assert len(kappa) == T, "kappa must be a vector of length T"

    ### Variables   
    # Balances
    c = cp.Variable(nonneg=True, name='c') # consumption
    q = cp.Variable(nonneg=True, name='q') # bequest
    B = cp.Variable(T+1, nonneg=True, name='B')
    I = cp.Variable(T+1, nonneg=True, name='I')
    R = cp.Variable(T+1, nonneg=True, name='R')

    # Withdrawals
    b = cp.Variable(T, name='b')
    i = cp.Variable(T, name='i'); i_c = cp.Variable(T, nonneg=True, name='i_c'); i_d = cp.Variable(T, nonneg=True, name='i_d'); i_w = cp.Variable(T, nonneg=True, name='i_w')
    r = cp.Variable(T, name='r'); r_c = cp.Variable(T, nonneg=True, name='r_c'); r_d = cp.Variable(T, nonneg=True, name='r_d'); r_w = cp.Variable(T, nonneg=True, name='r_w')

    # Taxes
    tau = cp.Variable(T, nonneg=True, name='tau')
    if deterministic:
        omega = cp.maximum(i_c - i_d + i_w + cp.multiply((B[:-1] - b), (rho_B - 1)) + e + a, 0)
        brokerage_tax = 0
    else: 
        omega = cp.maximum(i_c - i_d + i_w + + e + a, 0)
        if not no_brokerage:
            omega_b = cp.pos(b) * max((1 - basis / price), 0)
            brokerage_tax = capital_gain_tax(omega_b, capital_tax_rate=capital_tax_rate)
        else:
            brokerage_tax = 0

    phi = cp.vstack([eta[0] * omega] + [
         (eta[i] - eta[i-1]) * cp.pos(omega - beta[i-1]) for i in range(1, len(beta))
    ]).sum(axis=0) + brokerage_tax

    ### Objective
    objective = _objective(c, q, c0, gamma)

    ### Constraints
    constraints = [
        B[0] == B_init, I[0] == I_init, R[0] == R_init,
        B >= 0, I >= 0, R >= 0,
        B[1:] == cp.multiply((B[:-1] - b), rho_B), I[1:] == cp.multiply((I[:-1] - i), rho_I), R[1:] == cp.multiply((R[:-1] - r), rho_R),
        i == i_c - i_d + i_w,
        r == - r_c - r_d + r_w,
        b <= B[:-1], i <= I[:-1], r <= R[:-1],
        b + i + r + e + a == c + l + tau,
        i_w >= cp.multiply(kappa, I[:-1]), i_d + r_d <= np.minimum(d_max, e), i_c == r_c,
        tau >= phi,
        q == B[-1] + I[-1] + R[-1], # XXX 
    ]
    

    ### Problem
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
        q=q,
        B=B,
        I=I,
        R=R,
        b=b,
        i=i,
        r=r,
        i_c=i_c,
        i_d=i_d,
        i_w=i_w,
        r_c=r_c,
        r_d=r_d,
        r_w=r_w,
        tau=tau,
        brokerage_tax=brokerage_tax,
        omega=omega,
        phi=phi,
        e=e,
        a=a,
        l=l,
        c0=c0,
    )




