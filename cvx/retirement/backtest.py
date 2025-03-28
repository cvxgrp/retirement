import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from collections import namedtuple
from .retirement_problem import get_retirement_problem
from .black_scholes import get_collar
from .utils import gmm_sample, transform_data, inverse_transform_data, get_stationary_cov, solve_ar
from .brokerage import Brokerage
import time
import xarray as xr

Backtest = namedtuple('Backtest', ['history', 'plans'])

def _get_post_tax_consumption(withdrawal, b, basis, price, brackets, i_w, e, a, l, data, B_init, rho_B):

    if basis is None: # parameters.deterministic == True
        B_interest = (B_init - b) * (rho_B - 1)
        omega = max(i_w + B_interest + e + a, 0)
        brokerage_tax = 0
    else:
        omega = max(i_w + e + a, 0)
        omega_b = max(b, 0) * max((1 - basis / price), 0)
        brokerage_tax = Brokerage.capital_gain_tax(omega_b, omega, brackets)

    tau = np.sum([data.eta[0] * omega] + [
        (data.eta[i] - data.eta[i-1]) * np.maximum(omega - data.beta[i-1], 0) for i in range(1, len(data.beta))
    ]) + brokerage_tax

    return withdrawal + e + a - l - tau, omega, tau

def _get_withdrawals(withdrawal, B_frac, I_frac, R_frac, B_init, I_init, R_init, I_required):

    if withdrawal <= I_required: # negative withdrawal means deposit; deposit to brokerage account
        withdrawal -= I_required
        return withdrawal, I_required, 0, 0, 0, I_required, 0, 0, 0

    else:
        withdrawal -= I_required
        b = min(B_frac * withdrawal, B_init) 
        i = min(I_frac * withdrawal + I_required, I_init) 
        r = min(R_frac * withdrawal, R_init)
        i_c = 0
        i_d = 0
        i_w = i
        r_c = 0
        r_d = 0
        r_w = r


        return b, i, r, i_c, i_d, i_w, r_c, r_d, r_w


def compute_withdrawals(B_init, I_init, R_init, basis, price, brackets, data, parameters, age):
    """
    param B_init: initial brokerage account balance
    param I_init: initial IRA account balance
    param R_init: initial Roth IRA account balance
    param data: data
    param rho_B: total inflation adjusted return on brokerage account
    param parameters: parameters
    param age: age

    returns: b, i, r, i_c, i_d, i_w, r_c, r_d, r_w
    """
    e = parameters.e.loc[age]
    a = parameters.a.loc[age]
    l = parameters.l.loc[age]

    I_required = data.kappa.loc[age] * I_init
    B_frac = B_init / (B_init + I_init - I_required + R_init)
    I_frac = (I_init - I_required) / (B_init + I_init - I_required + R_init)
    R_frac = R_init / (B_init + I_init - I_required + R_init)


    high = B_init + I_init + R_init
    low = - e - a + l

    withdrawal = (high + low) / 2
    b, i, r, i_c, i_d, i_w, r_c, r_d, r_w = _get_withdrawals(withdrawal, B_frac, I_frac, R_frac, B_init, I_init, R_init, I_required)
    post_tax_consumption, omega, tau = _get_post_tax_consumption(withdrawal, b, basis, price, brackets, i_w, e, a, l, data, B_init, parameters.rho_B)
 
    while high - low > 1e-6:
        withdrawal = (high + low) / 2
        b, i, r, i_c, i_d, i_w, r_c, r_d, r_w = _get_withdrawals(withdrawal, B_frac, I_frac, R_frac, B_init, I_init, R_init, I_required)
        post_tax_consumption, omega, tau = _get_post_tax_consumption(withdrawal, b, basis, price, brackets, i_w, e, a, l, data, B_init, parameters.rho_B)

        if post_tax_consumption < parameters.c0:
            low = withdrawal
        else:
            high = withdrawal
    
    # Either cash balance must hold or withdrawal must be equal to total
    # balance, i.e., retiree is broke
    assert np.isclose(b+i+r+e+a, post_tax_consumption + l + tau, atol=1e-6) or np.isclose(withdrawal, B_init+I_init+R_init, atol=1e-6), f'Cash balance descrepancy'
    assert np.isclose(b + i +r, withdrawal), f'Withdraw descrepancy'
    return b, i, r, i_c, i_d, i_w, r_c, r_d, r_w, post_tax_consumption, omega, tau

def run_backtest(params):
    """
    param age_start: starting age param age_end: ending age param data: data
    tuple param gmm_ret: gmm for returns param gmm_inflation: gmm for inflation
    param a0: mean of the inflation and treasury rate

    returns: 
    """

    data, parameters, gmm_ret, inf_treas_ar, algo, seed = params
    age_start, age_end = parameters.age_start, parameters.age_end
    sex = parameters.sex
    
    if seed is not None and seed != 'historical':
        np.random.seed(seed)

    if seed == 'historical':
        _sp500 = data.sp500.values[-(age_end-age_start+1):]
        _inflation = data.inflation.values[-(age_end-age_start+1):]
        _treasury = data.treasury.values[-(age_end-age_start+1):]

    else:
        _sp500 = [gmm_sample(gmm_ret) for _ in range(age_end-age_start+1)]
        _eps = [np.random.multivariate_normal(np.zeros(2), inf_treas_ar.Sigma_eps) for _ in range(age_end-age_start+1)]
    
    history = pd.DataFrame(index=range(age_start, age_end+1),
                           columns=['c',
                                'ret_B',
                                'ret_I',
                                'ret_R',
                                'ret_adj_B',
                                'ret_adj_I',
                                'ret_adj_R',
                                'sp500',
                                'inflations',
                                'treasuries',
                                'earned_incomes',
                                'additional_incomes',
                                'liabilities',
                                'qs',
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
                                'floor_B',
                                'ceiling_B',
                                'floor_I',
                                'ceiling_I',
                                'floor_R',
                                'ceiling_R',
                                'omega',
                                'omega_realized',
                                'tau',
                                'tau_realized',
                                'floor',
                                'ceiling',
                                'age',
                                'q',
                                'solve_time',
                                'cvxpy_time',
                                'eps1',
                                'eps2',
                                'seed',
                                ])
    plans = {
        'B' : pd.DataFrame(index=range(age_start, age_end+1), columns=range(age_start, age_end+1)),
        'R' : pd.DataFrame(index=range(age_start, age_end+1), columns=range(age_start, age_end+1)),
        'I' : pd.DataFrame(index=range(age_start, age_end+1), columns=range(age_start, age_end+1)),
        'b' : pd.DataFrame(index=range(age_start, age_end+1), columns=range(age_start, age_end+1)),
        'i' : pd.DataFrame(index=range(age_start, age_end+1), columns=range(age_start, age_end+1)),
        'r' : pd.DataFrame(index=range(age_start, age_end+1), columns=range(age_start, age_end+1)),
        'tau' : pd.DataFrame(index=range(age_start, age_end+1), columns=range(age_start, age_end+1)),
        'c' : pd.Series(index=range(age_start, age_end+1)),
        'q' : pd.Series(index=range(age_start, age_end+1)),
    }

    k, s1, s2 = parameters.k, parameters.s_minus, parameters.s_plus
    inf_treas = np.random.multivariate_normal(inf_treas_ar.a0, inf_treas_ar.Sigma)
    transformed_inf_prev = inf_treas[0]
    inflation_prev = inverse_transform_data(transformed_inf_prev, k, s1, s2)
    risk_free_prev = inf_treas[1]

    B_init, I_init, R_init = parameters.B_init, parameters.I_init, parameters.R_init
    w_stocks_B, w_stocks_I, w_stocks_R = parameters.w_stocks_B, parameters.w_stocks_I, parameters.w_stocks_R
    b_account = Brokerage(shares=pd.Series(parameters.shares, index=[age_start-1]), basis=pd.Series(parameters.basis, index=[age_start-1]), price=pd.Series(parameters.price, index=[age_start-1]))

    inflation_adj_factor = 1.

    for age in range(age_start, age_end):  
        
        ### Expected remaining life-time
        if not parameters.deterministic: # get expected time to death
            T = int(np.ceil(data.life_expectancy[sex].loc[age])) 
            basis = b_account.average_basis(age-1)
            price = b_account.price.loc[age-1]
        else: # die at pre-determined age
            T = age_end - age
            basis = None
            price = None
        
        if algo == 'MPC':
            
            if B_init <= 1:
                no_brokerage = True
            else:
                no_brokerage = False
            _T = min(int(T * 1.5), age_end - age)
            # _T = min(T + 10, age_end - age)
            # print(age, _T + age)
            # _T = T
            # _T = min(20, age_end - age)
            problem = get_retirement_problem(B_init, 
                                            I_init, 
                                            R_init, 
                                            basis,
                                            price,
                                            parameters.rho_B, 
                                            parameters.rho_I, 
                                            parameters.rho_R,  
                                            data.kappa.loc[age:age+_T-1].values,
                                            data.beta, 
                                            data.eta, 
                                            parameters.e.loc[age:age+_T-1].values, 
                                            parameters.a.loc[age:age+_T-1].values,
                                            parameters.l.loc[age:age+_T-1].values,
                                            _T, 
                                            parameters.d_max, 
                                            parameters.c0, 
                                            parameters.gamma,
                                            no_brokerage=no_brokerage,
                                            deterministic=parameters.deterministic,
                                            capital_tax_rate=parameters.capital_tax_rate,)
        
            try:

                _a = time.time()
                problem.problem.solve(verbose=False)
                _b = time.time()
           
            except:
                # problem.problem.solve(solver='MOSEK', verbose=True)

                print(B_init, I_init, R_init)
                return None
            
            c = problem.c.value
            q = problem.q.value
            b = problem.b.value[0]
            i = problem.i.value[0]
            r = problem.r.value[0]
            i_c = problem.i_c.value[0]
            i_d = problem.i_d.value[0]
            i_w = problem.i_w.value[0]
            r_c = problem.r_c.value[0]
            r_d = problem.r_d.value[0]
            r_w = problem.r_w.value[0]
            omega = problem.omega.value[0]
            tau = problem.tau.value[0]
            e = parameters.e.loc[age]
            a = parameters.a.loc[age]
            l = parameters.l.loc[age]

            # cash balance should always hold (ex-ante) by construction
            assert np.isclose(b+i+r+e+a, c + l + tau, atol=1e-6), f'Cash balance descrepancy'


        elif algo == '4_percent':
            if not parameters.deterministic:
                basis, price, brackets = b_account.average_basis(age-1), b_account.price.loc[age-1], b_account.tax_brackets
            else:
                basis, price, brackets = None, None, None
            b, i, r, i_c, i_d, i_w, r_c, r_d, r_w, c, omega, tau = compute_withdrawals(B_init, I_init, R_init, basis, price, brackets, data, parameters, age)
  
            e = parameters.e.loc[age]
            a = parameters.a.loc[age]
            l = parameters.l.loc[age]
            q = B_init + I_init + R_init - b - i - r

            # either cash balance must hold or withdrawal must be equal to total balance, i.e., retiree goes broke
            assert np.isclose(b+i+r+e+a, c + l + tau, atol=1e-6) or np.isclose(b + i + r, B_init+I_init+R_init, atol=1e-6), f'Cash balance descrepancy'

        def _get_collar(w_stocks):
            r_min = - 0.075
            downside = min((inflation_prev + r_min - (1-w_stocks) * risk_free_prev) / w_stocks, risk_free_prev)
            downside, upside  = get_collar(r=risk_free_prev, T=1, sigma=data.sp500.std(), downside=downside)

            return downside, upside
            
        if parameters.collar:
            downside_B, upside_B = _get_collar(w_stocks_B)
            downside_I, upside_I = _get_collar(w_stocks_I)
            downside_R, upside_R = _get_collar(w_stocks_R)

            ret_stocks_B = np.clip(_sp500[age-age_start], downside_B, upside_B)
            ret_stocks_I = np.clip(_sp500[age-age_start], downside_I, upside_I)
            ret_stocks_R = np.clip(_sp500[age-age_start], downside_R, upside_R)
        else:
            ret_stocks_B = ret_stocks_I = ret_stocks_R = _sp500[age-age_start]

        inf_treas_prev = np.array([transformed_inf_prev, risk_free_prev])
        if seed == 'historical':
            inflation_t = _inflation[age-age_start]
            risk_free_t = _treasury[age-age_start]
        else:
            inf_treas_t = inf_treas_ar.a0 + inf_treas_ar.A @ (inf_treas_prev - inf_treas_ar.a0) + _eps[age-age_start]
            transformed_inflation_t = inf_treas_t[0]
            inflation_t = inverse_transform_data(transformed_inflation_t, k, s1, s2)
            risk_free_t = inf_treas_t[1]

        ### Inflation adjusted returns
        if not parameters.deterministic: # simulate data
            ret_B = w_stocks_B * ret_stocks_B + (1-w_stocks_B) * risk_free_t
            ret_I = w_stocks_I * ret_stocks_I + (1-w_stocks_I) * risk_free_t
            ret_R = w_stocks_R * ret_stocks_R + (1-w_stocks_R) * risk_free_t
            ret_adj_B = ret_B - inflation_t 
            ret_adj_I = ret_I - inflation_t
            ret_adj_R = ret_R - inflation_t
        else: # use expected returns
            ret_adj_B = parameters.rho_B - 1
            ret_adj_I = parameters.rho_I - 1
            ret_adj_R = parameters.rho_R - 1

            ret_B = ret_adj_B + data.inflation.mean() # XXX

        B_interest = (B_init - b) * ret_adj_B
        I_interest = (I_init - i) * ret_adj_I
        R_interest = (R_init - r) * ret_adj_R

        if not parameters.deterministic:
            assert np.isclose(b_account.value(age-1), B_init * inflation_adj_factor), f'Brokerage account value discrepancy between real and nominal'

        ### Realized taxes
        if parameters.deterministic: # Use same tax rates as in the optimization problem
            omega_realized = max(i_c - i_d + i_w + B_interest + e + a, 0)
            brokerage_tax = 0 # taxed as income in the simplified model
            
        else: # Use exact tax rates
            omega_realized = max(i_c - i_d + i_w + e + a, 0) # excluding brokerage account interest
            price_nominal = b_account.price.loc[age-1] #* (1 + ret_B) # nominal price per share of brokerage account investment
            if b <= 0:
                b_account.buy(age-1, -b * inflation_adj_factor / price_nominal, price_nominal)
                brokerage_tax = 0
            elif b > 0:
                    brokerage_tax = b_account.sell(age-1, b * inflation_adj_factor, omega_realized) / inflation_adj_factor # this also sells shares inplace
            b_account.set_price(age, price_nominal * (1 + ret_B))
        taxes_realized = np.sum([data.eta[0] * omega_realized] + [
                (data.eta[i] - data.eta[i-1]) * np.maximum(omega_realized - data.beta[i-1], 0) for i in range(1, len(data.beta))
            ]) + brokerage_tax
                
        ### Save history
        tax_prediction = tau
        tax_correction = tax_prediction - taxes_realized # realized tax correction
        parameters.l.loc[age+1] += tax_correction

        history.loc[age, 'c'] = c 
        history.loc[age, 'ret_B'] = ret_stocks_B
        history.loc[age, 'ret_I'] = ret_stocks_I
        history.loc[age, 'ret_R'] = ret_stocks_R
        history.loc[age, 'ret_adj_B'] = ret_adj_B
        history.loc[age, 'ret_adj_I'] = ret_adj_I
        history.loc[age, 'ret_adj_R'] = ret_adj_R
        history.loc[age, 'sp500'] = _sp500[age-age_start]
        history.loc[age, 'inflations'] = inflation_t
        history.loc[age, 'treasuries'] = risk_free_t
        history.loc[age, 'earned_incomes'] = parameters.e.loc[age]
        history.loc[age, 'additional_incomes'] = parameters.a.loc[age]
        history.loc[age, 'liabilities'] = parameters.l.loc[age]
        history.loc[age, 'qs'] = q
        history.loc[age, 'B'] = B_init
        history.loc[age, 'I'] = I_init
        history.loc[age, 'R'] = R_init
        history.loc[age, 'b'] = b
        history.loc[age, 'i'] = i
        history.loc[age, 'r'] = r
        history.loc[age, 'i_c'] = i_c
        history.loc[age, 'i_d'] = i_d
        history.loc[age, 'i_w'] = i_w
        history.loc[age, 'r_c'] = r_c
        history.loc[age, 'r_d'] = r_d
        history.loc[age, 'r_w'] = r_w
        history.loc[age, 'omega'] = omega
        history.loc[age, 'omega_realized'] = omega_realized
        history.loc[age, 'tau'] = tax_prediction
        history.loc[age, 'tau_realized'] = taxes_realized
        if algo == 'MPC':
            plans['B'].loc[age:age+_T, age] = problem.B.value
            plans['I'].loc[age:age+_T, age] = problem.I.value
            plans['R'].loc[age:age+_T, age] = problem.R.value
            plans['b'].loc[age:age+_T-1, age] = problem.b.value
            plans['i'].loc[age:age+_T-1, age] = problem.i.value
            plans['r'].loc[age:age+_T-1, age] = problem.r.value
            plans['tau'].loc[age:age+_T-1, age] = problem.tau.value
            plans['c'].loc[age] = c
            plans['q'].loc[age] = q
            solve_time = problem.problem.solver_stats.solve_time
            history.loc[age, 'solve_time'] = solve_time
            history.loc[age, 'cvxpy_time'] = _b - _a
        if parameters.collar:
            history.loc[age, 'floor_B'] = downside_B
            history.loc[age, 'ceiling_B'] = upside_B
            history.loc[age, 'floor_I'] = downside_I
            history.loc[age, 'ceiling_I'] = upside_I
            history.loc[age, 'floor_R'] = downside_R
            history.loc[age, 'ceiling_R'] = upside_R

        B_init += B_interest - b
        I_init += I_interest - i
        R_init += R_interest - r

        inflation_prev = inflation_t
        risk_free_prev = risk_free_t
        inflation_adj_factor *= (1 + ret_B) / (1+ret_adj_B)

        ### Check if the retiree dies
        if parameters.deterministic: # die at pre-determined age
            if T == 1: 
                break
        else: # die at random age
            death_prob = data.mortality_rates[sex].loc[age]
            if np.random.rand() < death_prob or T==1:
                break
    
    history['age'] = age
    history['q'] = B_init + I_init + R_init # XXX
    history['eps1'] = [e[0] for e in _eps]
    history['eps2'] = [e[1] for e in _eps]
    history['seed'] = seed
    # history.loc[age+1, 'B'] = B_init
    # history.loc[age+1, 'I'] = I_init
    # history.loc[age+1, 'R'] = R_init
    # history.loc[age+1, 'b'] = 0
    # history.loc[age+1, 'i'] = 0
    # history.loc[age+1, 'r'] = 0
    # history.loc[age+1, 'liabilities'] = 0

    return Backtest(history, plans)
