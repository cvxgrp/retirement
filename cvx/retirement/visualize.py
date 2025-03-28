import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import namedtuple

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
                                'r_c_max',])

def plot(matrix, mean_legend=False, verbose=True):
    # Plot the main matrix data with no legend label
    np.random.seed(0)
    # random_inds = np.random.choice(np.arange(matrix.shape[1]), 10, replace=False)
    plt.plot(matrix.index, matrix.values[:, :], color='black', alpha=0.05, label=None)#, zorder=10)

    # Plot the mean with a specific label and color
    mean_values = matrix.median(axis=1)
    if verbose:
        mean_line, = plt.plot(matrix.index, mean_values, color='red', linewidth=2, label=f'Median {int(mean_values.mean())}', alpha=1)#, zorder=2)
    else:
        mean_line, = plt.plot(matrix.index, mean_values, color='red', linewidth=2, label='Median', alpha=1)

    # Plot the 1% and 99% quantiles with specific labels and colors
    lower_quantile_values = matrix.quantile(0.05, axis=1)
    upper_quantile_values = matrix.quantile(0.95, axis=1)
    
    if verbose:
        lower_quantile_line, = plt.plot(matrix.index, lower_quantile_values, color='blue', linestyle='--', label=f'5th percentile {int(lower_quantile_values.mean())}', alpha=1, zorder=3)
        upper_quantile_line, = plt.plot(matrix.index, upper_quantile_values, color='blue', linestyle='--', label=f'95th percentile {int(upper_quantile_values.mean())}', alpha=1, zorder=3)
    else:
        lower_quantile_line, = plt.plot(matrix.index, lower_quantile_values, color='blue', linestyle='--', label='5th and 95th percentiles', alpha=1, zorder=3)
        _, = plt.plot(matrix.index, upper_quantile_values, color='blue', linestyle='--', label=None, alpha=1, zorder=3)
    
    
    # Show legend only if mean_legend is True
    if mean_legend:
        # Only include mean and quantile lines in the legend with correct colors
        if verbose:
            plt.legend(handles=[mean_line, lower_quantile_line, upper_quantile_line])
        else:
            plt.legend(handles=[mean_line, lower_quantile_line])

def summarize(simulation, figsize=(7,4), save=False, y_min=None, y_max=None, target=None, verbose=True, bequest_max=5000, limits=None):

    # return_matrix
    plt.figure(figsize=figsize)
    plot(simulation.sp500_matrix)
    plt.xlabel('Age')
    plt.ylabel('Market return')
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/sp500_{save}.png', bbox_inches='tight')
    else:
        plt.title('Market return')

    # treasury_matrix
    plt.figure(figsize=figsize)
    plot(simulation.treasury_matrix)
    plt.xlabel('Age')
    plt.ylabel('Bond return')
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/treasury_{save}.png', bbox_inches='tight')
    else:
        plt.title('Bond return')
    
    # inflation_matrix
    plt.figure(figsize=figsize)
    plot(simulation.inflation_matrix)
    plt.xlabel('Age')
    plt.ylabel('Inflation rate')
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/inflation_{save}.png', bbox_inches='tight')
    else:
        plt.title('Inflation rate')

    # bequests
    plt.figure(figsize=figsize)
    plot(simulation.bequest_matrix)
    plt.xlabel('Age')
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/bequest_{save}.png', bbox_inches='tight')
    else:
        plt.title('Bequest')

    plt.ylim(0, bequest_max)

    # B_matrix
    plt.figure(figsize=figsize)
    plot(simulation.B_matrix)
    plt.xlabel('Age')
    plt.ylabel('1000 USD')
    plt.ylim(limits.B_min, limits.B_max)
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/B_account_{save}.png', bbox_inches='tight')
    else:
        plt.title('Brokerage account balance')
    

    # I_matrix
    plt.figure(figsize=figsize)
    plot(simulation.I_matrix)
    plt.xlabel('Age')
    plt.ylabel('1000 USD')
    plt.ylim(limits.I_min, limits.I_max)
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/I_account_{save}.png', bbox_inches='tight')
    else:
        plt.title('IRA account balance')

    # R_matrix
    plt.figure(figsize=figsize)
    plot(simulation.R_matrix)
    plt.xlabel('Age')
    plt.ylabel('1000 USD')
    plt.ylim(limits.R_min, limits.R_max)
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/R_account_{save}.png', bbox_inches='tight')
    else:
        plt.title('Roth IRA account balance')

    # b_matrix
    plt.figure(figsize=figsize)
    plot(simulation.b_matrix)
    plt.xlabel('Age')
    plt.ylabel('1000 USD')
    plt.ylim(limits.b_min, limits.b_max)
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/b_{save}.png', bbox_inches='tight')   
    else:
        plt.title('Withdrawals from brokerage account')

    # i_matrix
    plt.figure(figsize=figsize)
    plot(simulation.i_matrix)
    plt.xlabel('Age')
    plt.ylabel('1000 USD')
    plt.ylim(limits.i_min, limits.i_max)
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/i_{save}.png', bbox_inches='tight')
    else:
        plt.title('IRA withdrawals')

    # r_matrix
    plt.figure(figsize=figsize)
    plot(simulation.r_matrix)
    plt.xlabel('Age')
    plt.ylabel('1000 USD')
    plt.ylim(limits.r_min, limits.r_max)
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/r_{save}.png', bbox_inches='tight')
    else:
        plt.title('Roth IRA withdrawals')

    # r_c_matrix
    plt.figure(figsize=figsize)
    plot(simulation.r_c_matrix)
    plt.xlabel('Age')
    plt.ylabel('1000 USD')
    plt.ylim(limits.r_c_min, limits.r_c_max)
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/r_c_{save}.png', bbox_inches='tight')
    else:
        plt.title('Roth conversion')

    # bequest vs age
    plt.figure(figsize=figsize)
    plt.scatter(simulation.ages, simulation.bequests)
    plt.xlabel('Age')
    plt.ylabel('Bequest')
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/bequest_vs_age_{save}.png', bbox_inches='tight')
    else:
        plt.title('Bequest vs Age')

    # realized taxes
    plt.figure(figsize=figsize)
    plot(simulation.tau_realized_matrix)
    plt.xlabel('Age')
    plt.ylabel('1000 USD')
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/tau_realized_{save}.png', bbox_inches='tight')
    else:
        plt.title('Realized taxes')


    # plt.figure(figsize=figsize)
    # plot(simulation.treasury_matrix)
    # plt.xlabel('Age')
    # plt.title('10-year treasury rate')
    # plt.axvline(simulation.ages.mean(), color='green', linestyle='--')

    # plt.figure(figsize=figsize)
    # plot(simulation.inflation_matrix)
    # plt.xlabel('Age')
    # plt.title('Inflation rate')
    # plt.axvline(simulation.ages.mean(), color='green', linestyle='--')

    # plt.figure(figsize=figsize)
    # plot(simulation.treasury_matrix - simulation.inflation_matrix)
    # plt.xlabel('Age')
    # plt.title('Real 10-year treasury rate')
    # plt.axvline(simulation.ages.mean(), color='green', linestyle='--')

    # plt.figure(figsize=figsize)
    # plot(simulation.returns_adj_matrix.cumsum(axis=0))
    # plt.xlabel('Age')
    # plt.title('Inflation-adjusted cumulative portfolio returns')
    # plt.axvline(simulation.ages.mean(), color='green', linestyle='--')

    # plt.figure(figsize=figsize)
    # plot(simulation.cash_matrix, mean_legend=True)
    # plt.xlabel('Age')
    # plt.title('Consumption (1000 USD)')
    # plt.axvline(simulation.ages.mean(), color='green', linestyle='--')
    # plt.xlim(65, 100);
    # plt.ylim(0, 200);

    # plt.figure(figsize=figsize)
    # plot(simulation.bequest_matrix)
    # plt.xlabel('Age')
    # plt.title('Bequest')

    # plt.figure(figsize=figsize)
    # plot(simulation.B_matrix)
    # plt.xlabel('Age')
    # plt.title('B')

    # plt.figure(figsize=figsize)
    # plot(simulation.I_matrix)
    # plt.xlabel('Age')
    # plt.title('I')

    # plt.figure(figsize=figsize)
    # plot(simulation.R_matrix)
    # plt.xlabel('Age')
    # plt.title('R')

    # plt.figure(figsize=figsize)
    # plot(simulation.b_matrix)
    # plt.xlabel('Age')
    # plt.title('b')

    # plt.figure(figsize=figsize)
    # plot(simulation.i_matrix)
    # plt.xlabel('Age')
    # plt.title('i')

    # plt.figure(figsize=figsize)
    # plot(simulation.r_matrix)
    # plt.xlabel('Age')
    # plt.title('r')

    # plt.figure(figsize=figsize)
    # plot(simulation.d_B_matrix)
    # plt.xlabel('Age')
    # plt.title('d_B')

    # plt.figure(figsize=figsize)
    # plot(simulation.d_I_matrix)
    # plt.xlabel('Age')
    # plt.title('d_I')

    # plt.figure(figsize=figsize)
    # plot(simulation.d_R_matrix)
    # plt.xlabel('Age')
    # plt.title('d_R')

    # plt.figure(figsize=figsize)
    # plot(simulation.d_c_matrix)
    # plt.xlabel('Age')
    # plt.title('d_c')

    # plt.figure(figsize=figsize)
    # simulation.bequests.hist(bins=30)
    # plt.title('Bequest (1000 USD)')
    # plt.xlabel('Bequest')
    # plt.ylabel('Frequency');

    # plt.figure(figsize=figsize)
    # plt.scatter(simulation.ages, simulation.bequests)
    # plt.xlabel('Age')
    # plt.ylabel('Bequest')
    # plt.title('Bequest vs Age')
    # plt.axvline(simulation.ages.mean(), color='green', linestyle='--')
    # plt.axhline(simulation.bequests.mean(), color='red', linestyle='--')



def ecdf(x):
    return np.arange(1, len(x)+1) / len(x)

def plot_ecdf(data, label=None):
    plt.plot(np.sort(data), ecdf(data), label=label)

def summarize2(simulation, figsize=(7,4), save=False, target=None, verbose=True, x_min=None, x_max=None, y_min=None, y_max=None, legend=True, bequest_max=5000, n_bins_c=50, n_bins_q=50, bar=False):
    ###
    bins_c = np.linspace(x_min, x_max, n_bins_c)
    bins_q = np.linspace(0, bequest_max, n_bins_q)

    ### cash_matrix
    plt.figure(figsize=figsize)
    plot(simulation.cash_matrix, mean_legend=legend, verbose=verbose)
    plt.xlabel('Age')
    if target is not None:
        plt.axhline(target, color='orange', linestyle='--', linewidth=2, label=f'Target')

    if verbose:
        plt.axvline(simulation.ages.mean(), color='green', linestyle='--', label=f'Average age {int(simulation.ages.mean())}')
    else:
        plt.axvline(simulation.ages.mean(), color='green', linestyle='--', label='Average age')

    if legend:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4)
    plt.ylabel('1000 USD')
    if y_min is not None:
        assert y_max is not None
        plt.ylim(y_min, y_max)
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/consumption_{save}.png', bbox_inches='tight')
    else:
        pass
        # plt.title('Consumption')
    # plt.xlim(65, 100);
    # plt.ylim(0, 200);

    # pd.Series(simulation_MPC.cash_matrix.values.flatten()).dropna()

    ### Consumption
    plt.figure(figsize=figsize)
    consumption = pd.Series(simulation.cash_matrix.values.flatten()).dropna()
    # consumption.clip(x_min, x_max).hist(bins=n_bins)
    if not bar or bar:
        consumption.hist(bins=bins_c, density=True)
    # else:
    #     # left align
    #     plt.bar(consumption[0], height=1.0, align='edge')

    percentile_5 = consumption.quantile(0.05)
    percentile_95 = consumption.quantile(0.95)
    mean = consumption.median()
    if verbose:
        plt.title('Consumption')
        plt.axvline(mean, color='black', linestyle='--', label=f'Median {int(mean)}', linewidth=2.5)
        plt.axvline(percentile_5, color='red', linestyle='--', label=f'5th and 95th percentiles {int(percentile_5)}-{int(percentile_95)}', linewidth=2.5)
        plt.axvline(percentile_95, color='red', linestyle='--', linewidth=2.5)
    else:
        plt.axvline(mean, color='black', linestyle='--', linewidth=2.5, label=f'Median')
        plt.axvline(percentile_5, color='red', linestyle='--', linewidth=2.5, label=f'5th and 95th percentiles')
        plt.axvline(percentile_95, color='red', linestyle='--', linewidth=2.5)
    if legend:
        plt.legend()
    plt.xlabel('1000 USD')
    plt.ylabel('Density');
    if x_min is not None:
        assert x_max is not None
        plt.xlim(x_min, x_max)
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
    plt.tight_layout()
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/consumption_hist_{save}.png', bbox_inches='tight')

    ### Minimum consumption
    plt.figure(figsize=figsize)
    simulation.cash_matrix.min(axis=0).clip(x_min, x_max).hist(bins=bins_c, density=True)
    percentile_5 = simulation.cash_matrix.min(axis=0).quantile(0.05)
    percentile_95 = simulation.cash_matrix.min(axis=0).quantile(0.95)
    mean = simulation.cash_matrix.min(axis=0).median()
    if verbose:
        plt.title('Minimum consumption')
        plt.axvline(mean, color='black', linestyle='--', label=f'Median {int(mean)}', linewidth=2)
        plt.axvline(percentile_5, color='red', linestyle='--', label=f'5th and 95th percentiles {int(percentile_5)}-{int(percentile_95)}', linewidth=2)
        plt.axvline(percentile_95, color='red', linestyle='--', linewidth=2)
    else:
        plt.axvline(mean, color='black', linestyle='--', linewidth=2, label=f'Median')
        plt.axvline(percentile_5, color='red', linestyle='--', linewidth=2, label=f'5th and 95th percentiles')
        plt.axvline(percentile_95, color='red', linestyle='--', linewidth=2)
    if legend:
        plt.legend()
    plt.xlabel('1000 USD')
    plt.ylabel('Density');
    if x_min is not None:
        assert x_max is not None
        plt.xlim(x_min, x_max)
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
    plt.tight_layout()
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/min_consumption_hist_{save}.png', bbox_inches='tight')

    ### Bequest
    plt.figure(figsize=figsize)
    simulation.bequests.clip(0, bequest_max).hist(bins=bins_q, density=True)

    # plot but define bin width
    # plt.hist(simulation.bequests.clip(0, bequest_max), bins=np.arange(0, bequest_max, 100))
    mean = simulation.bequests.median()
    percentile_5 = simulation.bequests.quantile(0.1)
    percentile_95 = simulation.bequests.quantile(0.9)
    if verbose:
            plt.title('Bequest')
            plt.axvline(mean, color='black', linestyle='--', label=f'Median {int(mean)}', linewidth=2)
            plt.axvline(percentile_5, color='red', linestyle='--', label=f'5th and 95th percentiles {int(percentile_5)}-{int(percentile_95)}', linewidth=2)
            plt.axvline(percentile_95, color='red', linestyle='--', linewidth=2)
    else:
        plt.axvline(mean, color='black', linestyle='--', linewidth=2, label=f'Median')
        plt.axvline(percentile_5, color='red', linestyle='--', linewidth=2, label=f'5th and 95th percentiles')
        plt.axvline(percentile_95, color='red', linestyle='--', linewidth=2)
    if legend:
        plt.legend()
    plt.xlabel('1000 USD')
    plt.ylabel('Density');
    plt.xlim(0, bequest_max)
    # show four significant digits
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.4f}'))
    plt.tight_layout()
    if legend:
        plt.legend()
    if save:
        plt.savefig(f'/Users/kasper/Documents/Stanford/Research/My papers/retirement/retirement-paper/figures/bequest_hist_{save}.png', bbox_inches='tight')
    else:
        plt.title('Bequest')