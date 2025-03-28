import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt

from .backtest import run_backtest
from .utils import solve_ar, get_stationary_cov, transform_data
from .data_loader import load_data
from .visualize import plot

Limits = namedtuple('Limits', ['B_min',
                               'B_max',
                                'I_min',
                                'I_max',
                                'R_min',
                                'R_max',
                                'b_min',
                                'b_max',
                                'i_min',
                                'i_max',
                                'r_min',
                                'r_max',
                                'r_c_min',
                                'r_c_max',
                                'q_max'])

@dataclass(frozen=True)
class Simulation:
    c_matrix: pd.DataFrame
    ret_matrix_B: pd.DataFrame
    ret_matrix_I: pd.DataFrame
    ret_matrix_R: pd.DataFrame
    ret_adj_matrix_B: pd.DataFrame
    ret_adj_matrix_I: pd.DataFrame
    ret_adj_matrix_R: pd.DataFrame
    sp500_matrix: pd.DataFrame     
    eps1_matrix: pd.DataFrame
    eps2_matrix: pd.DataFrame
    seed_matrix: pd.Series
    inflation_matrix: pd.DataFrame
    treasury_matrix: pd.DataFrame
    earned_incomes_matrix: pd.DataFrame
    additional_incomes_matrix: pd.DataFrame
    liabilities_matrix: pd.DataFrame    
    q_matrix: pd.DataFrame
    B_matrix: pd.DataFrame      
    I_matrix: pd.DataFrame
    R_matrix: pd.DataFrame
    b_matrix: pd.DataFrame
    i_matrix: pd.DataFrame
    r_matrix: pd.DataFrame
    i_c_matrix: pd.DataFrame
    i_d_matrix: pd.DataFrame
    i_w_matrix: pd.DataFrame
    r_c_matrix: pd.DataFrame
    r_d_matrix: pd.DataFrame
    r_w_matrix: pd.DataFrame
    floor_B: pd.DataFrame
    ceiling_B: pd.DataFrame
    floor_I: pd.DataFrame
    ceiling_I: pd.DataFrame
    floor_R: pd.DataFrame
    ceiling_R: pd.DataFrame
    omega_matrix: pd.DataFrame
    omega_realized_matrix: pd.DataFrame
    tau_matrix: pd.DataFrame
    tau_realized_matrix: pd.DataFrame   
    floor_matrix: pd.DataFrame
    ceiling_matrix: pd.DataFrame
    ages: pd.Series
    q: pd.Series
    plans: dict
    solve_time_matrix: pd.DataFrame
    cvxpy_time_matrix: pd.DataFrame 

    def summarize(self, figsize=(7,4), save=None, save_dir=None, limits=None):

        """
        parameters
        ----------
        simulation : Simulation
            Simulation object
        figsize : tuple
            Figure size
        save : str or None
            Suffix for the saved figures
        save_dir : str or None
            Directory to save the figures
        limits : namedtuple
            Limits for the y-axis
        """

        if save is not None:
            assert save_dir is not None, 'save_dir must be provided if save is True'

        # return_matrix
        plt.figure(figsize=figsize)
        plot(self.sp500_matrix)
        plt.xlabel('Age')
        plt.ylabel('Market return')
        if save is not None:
            # save_dir = '/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/'
            plt.savefig(f'{save_dir}sp500_{save}.png', bbox_inches='tight')
        else:
            plt.title('Market return')

        # treasury_matrix
        plt.figure(figsize=figsize)
        plot(self.treasury_matrix)
        plt.xlabel('Age')
        plt.ylabel('Bond return')
        if save is not None:
            plt.savefig(f'{save_dir}treasury_{save}.png', bbox_inches='tight')
        else:
            plt.title('Bond return')
        
        # inflation_matrix
        plt.figure(figsize=figsize)
        plot(self.inflation_matrix)
        plt.xlabel('Age')
        plt.ylabel('Inflation rate')
        if save is not None:
            plt.savefig(f'{save_dir}inflation_{save}.png', bbox_inches='tight')
        else:
            plt.title('Inflation rate')

        # bequests
        plt.figure(figsize=figsize)
        plot(self.q_matrix)
        plt.xlabel('Age')
        if save is not None:
            plt.savefig(f'{save_dir}bequest_{save}.png', bbox_inches='tight')
        else:
            plt.title('Bequest')
        if limits is not None:
            plt.ylim(0, limits.q_max)

        # B_matrix
        plt.figure(figsize=figsize)
        plot(self.B_matrix)
        plt.xlabel('Age')
        plt.ylabel('1000 USD')
        if limits is not None:
            plt.ylim(limits.B_min, limits.B_max)
        if save is not None:
            plt.savefig(f'{save_dir}B_account_{save}.png', bbox_inches='tight')
        else:
            plt.title('Brokerage account balance')
        

        # I_matrix
        plt.figure(figsize=figsize)
        plot(self.I_matrix)
        plt.xlabel('Age')
        plt.ylabel('1000 USD')
        if limits is not None:
            plt.ylim(limits.I_min, limits.I_max)
        if save is not None:
            plt.savefig(f'{save_dir}I_account_{save}.png', bbox_inches='tight')
        else:
            plt.title('IRA account balance')

        # R_matrix
        plt.figure(figsize=figsize)
        plot(self.R_matrix)
        plt.xlabel('Age')
        plt.ylabel('1000 USD')
        if limits is not None:
            plt.ylim(limits.R_min, limits.R_max)
        if save is not None:
            plt.savefig(f'{save_dir}R_account_{save}.png', bbox_inches='tight')
        else:
            plt.title('Roth IRA account balance')

        # b_matrix
        plt.figure(figsize=figsize)
        plot(self.b_matrix)
        plt.xlabel('Age')
        plt.ylabel('1000 USD')
        if limits is not None:
            plt.ylim(limits.b_min, limits.b_max)
        if save is not None:
            plt.savefig(f'{save_dir}b_{save}.png', bbox_inches='tight')
        else:
            plt.title('Withdrawals from brokerage account')

        # i_matrix
        plt.figure(figsize=figsize)
        plot(self.i_matrix)
        plt.xlabel('Age')
        plt.ylabel('1000 USD')
        if limits is not None:
            plt.ylim(limits.i_min, limits.i_max)
        if save is not None:
            plt.savefig(f'{save_dir}i_{save}.png', bbox_inches='tight')
        else:
            plt.title('IRA withdrawals')

        # r_matrix
        plt.figure(figsize=figsize)
        plot(self.r_matrix)
        plt.xlabel('Age')
        plt.ylabel('1000 USD')
        if limits is not None:
            plt.ylim(limits.r_min, limits.r_max)
        if save is not None:
            plt.savefig(f'{save_dir}r_{save}.png', bbox_inches='tight')
        else:
            plt.title('Roth IRA withdrawals')

        # r_c_matrix
        plt.figure(figsize=figsize)
        plot(self.r_c_matrix)
        plt.xlabel('Age')
        plt.ylabel('1000 USD')
        if limits is not None:
            plt.ylim(limits.r_c_min, limits.r_c_max)
        if save is not None:
            plt.savefig(f'{save_dir}r_c_{save}.png', bbox_inches='tight')
        else:
            plt.title('Roth conversion')

        # bequest vs age
        plt.figure(figsize=figsize)
        plt.scatter(self.ages, self.q)
        plt.xlabel('Age')
        plt.ylabel('Bequest')
        if save is not None:
            plt.savefig(f'{save_dir}bequest_vs_age_{save}.png', bbox_inches='tight')
        else:
            plt.title('Bequest vs Age')

        # realized taxes
        plt.figure(figsize=figsize)
        plot(self.tau_realized_matrix)
        plt.xlabel('Age')
        plt.ylabel('1000 USD')
        if save is not None:
            plt.savefig(f'{save_dir}tau_realized_{save}.png', bbox_inches='tight')
        else:
            plt.title('Realized taxes')


    

@dataclass(frozen=True)
class Parameters:
    B_init: float
    I_init: float
    R_init: float
    basis: Optional[float] = None
    price: Optional[float] = None
    shares: Optional[float] = None
    age_start: int = 65
    age_end: int = 120
    sex: str = 'F'
    collar: bool=False
    rho_B: float = 1.033
    rho_I: float = 1.057
    rho_R: float = 1.057
    e: pd.Series = field(default_factory=lambda: pd.Series(data=0.,index=range(65,120)))
    a: pd.Series = field(default_factory=lambda: pd.Series(data=0.,index=range(65,120)))
    l: pd.Series = field(default_factory=lambda: pd.Series(data=0.,index=range(65,120)))
    d_max: float=8.
    c0: Optional[float] = None
    gamma: float=500
    w_stocks_B: float=0.2
    w_stocks_I: float=0.6
    w_stocks_R: float=0.6
    k: float = 0.029
    s_minus: float = 2.5
    s_plus: float = 0.75
    deterministic: bool = False
    capital_tax_rate: float=0.15

    def __post_init__(self):
        if self.basis is None:
            object.__setattr__(self, 'basis', self.B_init / 2)
        if self.shares is None:
            object.__setattr__(self, 'shares', 1.)
        if self.price is None:
            object.__setattr__(self, 'price', self.B_init)
        if self.c0 is None:
            object.__setattr__(self, 'c0', (self.B_init + self.I_init + self.R_init + self.a.loc[self.age_start:85].sum() + self.e[self.age_start:85].sum()) * 0.0375) 

InfTreasAR = namedtuple('InfTreasAR', ['a0', 'A', 'Sigma_eps', 'Sigma'])

def simulate(parameters, n_sims=100, algo='MPC', seed=None):
    """
    Paramaters
    ----------
    parameters : Parameters
        Parameters object
    n_sims : int
        Number of simulations
    algo : str = 'MPC' or '4_percent'
        Algorithm to use
    seed : int or None  
        Random seed
    """
    if seed is not None and seed != 'historical':
        np.random.seed(seed)

    data = load_data()

    # Fit sp500 GMM
    gmm_ret = GaussianMixture(n_components=2, random_state=0)
    gmm_ret.fit(data.sp500.values.reshape(-1, 1))

    # Fit inflation and treasury GMM
    k, s1, s2 = parameters.k, parameters.s_minus, parameters.s_plus
    X = pd.concat([transform_data(data.inflation, k, s1, s2), data.treasury], axis=1).dropna()
    a0, A = solve_ar(X.values)
    eps = (X.shift(-1).values - a0 - (X.values-a0) @ A.T)[:-1]
    Sigma_eps = np.cov(eps.T)
    Sigma = get_stationary_cov(A, Sigma_eps)
    inf_treas_ar = InfTreasAR(a0, A, Sigma_eps, Sigma)

    if n_sims > 1:
        all_data = [data] * n_sims
        all_parameters = [parameters] * n_sims
        all_gmm_ret = [gmm_ret] * n_sims
        all_inf_treas_ar = [inf_treas_ar] * n_sims
        all_algos = [algo] * n_sims
        if seed == 'historical':
            all_seeds = [seed] * n_sims
        else:
            all_seeds = np.random.randint(0, n_sims * 100, n_sims)

        params = zip(all_data, all_parameters, all_gmm_ret, all_inf_treas_ar, all_algos, all_seeds)

        results = []
        pool = mp.Pool(mp.cpu_count())
        for res in tqdm(pool.imap(run_backtest, list(params)), total=n_sims):
            results.append(res)
        pool.close()
        pool.join()
    else:
        params = (data, parameters, gmm_ret, inf_treas_ar, algo, seed)
        results = [run_backtest(params)]

    c_matrix = pd.DataFrame([r.history['c'] for r in results if r is not None]).T
    ret_matrix_B = pd.DataFrame([r.history['ret_B'] for r in results if r is not None]).T
    ret_matrix_I = pd.DataFrame([r.history['ret_I'] for r in results if r is not None]).T
    ret_matrix_R = pd.DataFrame([r.history['ret_R'] for r in results if r is not None]).T
    ret_adj_matrix_B = pd.DataFrame([r.history['ret_adj_B'] for r in results if r is not None]).T
    ret_adj_matrix_I = pd.DataFrame([r.history['ret_adj_I'] for r in results if r is not None]).T
    ret_adj_matrix_R = pd.DataFrame([r.history['ret_adj_R'] for r in results if r is not None]).T
    sp500_matrix = pd.DataFrame([r.history['sp500'] for r in results if r is not None]).T
    eps1_matrix = pd.DataFrame([r.history['eps1'] for r in results if r is not None]).T
    eps2_matrix = pd.DataFrame([r.history['eps2'] for r in results if r is not None]).T
    seed_matrix = pd.Series([r.history['seed'].iloc[0] for r in results if r is not None])
    inflation_matrix = pd.DataFrame([r.history['inflations'] for r in results if r is not None]).T
    treasury_matrix = pd.DataFrame([r.history['treasuries'] for r in results if r is not None]).T
    earned_incomes_matrix = pd.DataFrame([r.history['earned_incomes'] for r in results if r is not None]).T
    additional_incomes_matrix = pd.DataFrame([r.history['additional_incomes'] for r in results if r is not None]).T
    liabilities_matrix = pd.DataFrame([r.history['liabilities'] for r in results if r is not None]).T
    q_matrix = pd.DataFrame([r.history['qs'] for r in results if r is not None]).T
    B_matrix = pd.DataFrame([r.history['B'] for r in results if r is not None]).T
    I_matrix = pd.DataFrame([r.history['I'] for r in results if r is not None]).T
    R_matrix = pd.DataFrame([r.history['R'] for r in results if r is not None]).T
    b_matrix = pd.DataFrame([r.history['b'] for r in results if r is not None]).T
    i_matrix = pd.DataFrame([r.history['i'] for r in results if r is not None]).T
    r_matrix = pd.DataFrame([r.history['r'] for r in results if r is not None]).T
    i_c_matrix = pd.DataFrame([r.history['i_c'] for r in results if r is not None]).T
    i_d_matrix = pd.DataFrame([r.history['i_d'] for r in results if r is not None]).T
    i_w_matrix = pd.DataFrame([r.history['i_w'] for r in results if r is not None]).T
    r_c_matrix = pd.DataFrame([r.history['r_c'] for r in results if r is not None]).T
    r_d_matrix = pd.DataFrame([r.history['r_d'] for r in results if r is not None]).T
    r_w_matrix = pd.DataFrame([r.history['r_w'] for r in results if r is not None]).T
    floor_B = pd.DataFrame([r.history['floor_B'] for r in results if r is not None]).T
    ceiling_B = pd.DataFrame([r.history['ceiling_B'] for r in results if r is not None]).T
    floor_I = pd.DataFrame([r.history['floor_I'] for r in results if r is not None]).T
    ceiling_I = pd.DataFrame([r.history['ceiling_I'] for r in results if r is not None]).T
    floor_R = pd.DataFrame([r.history['floor_R'] for r in results if r is not None]).T
    ceiling_R = pd.DataFrame([r.history['ceiling_R'] for r in results if r is not None]).T
    omega_matrix = pd.DataFrame([r.history['omega'] for r in results if r is not None]).T
    omega_realized_matrix = pd.DataFrame([r.history['omega_realized'] for r in results if r is not None]).T
    tau_matrix = pd.DataFrame([r.history['tau'] for r in results if r is not None]).T
    tau_realized_matrix = pd.DataFrame([r.history['tau_realized'] for r in results if r is not None]).T
    floor_matrix = pd.DataFrame([r.history['floor'] for r in results if r is not None]).T
    ceiling_matrix = pd.DataFrame([r.history['ceiling'] for r in results if r is not None]).T
    ages = pd.Series([r.history.age.iloc[-1] for r in results if r is not None])
    q = pd.Series([r.history['q'].iloc[-1] for r in results if r is not None])
    solve_time_matrix = pd.DataFrame([r.history['solve_time'] for r in results if r is not None]).T
    cvxpy_time_matrix = pd.DataFrame([r.history['cvxpy_time'] for r in results if r is not None]).T

    B_plans = {
        i : results[i].plans['B'] for i in range(n_sims) if results[i] is not None
    }
    I_plans = {
        i : results[i].plans['I'] for i in range(n_sims) if results[i] is not None
    }
    R_plans = {
        i : results[i].plans['R'] for i in range(n_sims) if results[i] is not None
    }
    b_plans = {
        i : results[i].plans['b'] for i in range(n_sims) if results[i] is not None
    }
    i_plans = { 
        i : results[i].plans['i'] for i in range(n_sims) if results[i] is not None
    }
    r_plans = {
        i : results[i].plans['r'] for i in range(n_sims) if results[i] is not None
    }
    tau_plans = {
        i : results[i].plans['tau'] for i in range(n_sims) if results[i] is not None
    }
    c_plans = {
        i : results[i].plans['c'] for i in range(n_sims) if results[i] is not None
    }
    q_plans = {
        i : results[i].plans['q'] for i in range(n_sims) if results[i] is not None
    }

    plans = {
        'B' : B_plans,
        'I' : I_plans,
        'R' : R_plans,
        'b' : b_plans,
        'i' : i_plans,
        'r' : r_plans,
        'tau' : tau_plans,
        'c' : c_plans,
        'q' : q_plans
    }

    return Simulation(c_matrix=c_matrix,
                        ret_matrix_B=ret_matrix_B,
                        ret_matrix_I=ret_matrix_I,
                        ret_matrix_R=ret_matrix_R,
                        ret_adj_matrix_B=ret_adj_matrix_B,
                        ret_adj_matrix_I=ret_adj_matrix_I,
                        ret_adj_matrix_R=ret_adj_matrix_R,
                        sp500_matrix=sp500_matrix,
                        eps1_matrix=eps1_matrix,
                        eps2_matrix=eps2_matrix,
                        seed_matrix=seed_matrix,
                        inflation_matrix=inflation_matrix,
                        treasury_matrix=treasury_matrix,
                        earned_incomes_matrix=earned_incomes_matrix,
                        additional_incomes_matrix=additional_incomes_matrix,
                        liabilities_matrix=liabilities_matrix,
                        q_matrix=q_matrix,
                        B_matrix=B_matrix,
                        I_matrix=I_matrix,
                        R_matrix=R_matrix,
                        b_matrix=b_matrix,
                        i_matrix=i_matrix,
                        r_matrix=r_matrix,
                        i_c_matrix=i_c_matrix,
                        i_d_matrix=i_d_matrix,
                        i_w_matrix=i_w_matrix,
                        r_c_matrix=r_c_matrix,
                        r_d_matrix=r_d_matrix,
                        r_w_matrix=r_w_matrix,
                        floor_B=floor_B,
                        ceiling_B=ceiling_B,
                        floor_I=floor_I,
                        ceiling_I=ceiling_I,
                        floor_R=floor_R,
                        ceiling_R=ceiling_R,
                        omega_matrix=omega_matrix,
                        omega_realized_matrix=omega_realized_matrix,
                        tau_matrix=tau_matrix,
                        tau_realized_matrix=tau_realized_matrix,
                        floor_matrix=floor_matrix,
                        ceiling_matrix=ceiling_matrix,
                        ages=ages,
                        q=q,
                        plans=plans,
                        solve_time_matrix=solve_time_matrix,
                        cvxpy_time_matrix=cvxpy_time_matrix,
                        )