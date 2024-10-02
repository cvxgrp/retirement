import numpy as np
import pandas as pd
import yfinance as yf
from collections import namedtuple

Data = namedtuple("Data", ["beta",
                            "eta",
                            "distributions",
                            "kappa",
                            "sp500",
                            "treasury",
                            "inflation"])

# https://www.irs.gov/newsroom/irs-provides-tax-inflation-adjustments-for-tax-year-2024
# Marginal rates: For tax year 2024, the top tax rate remains 37% for individual single taxpayers with incomes greater than $609,350 ($731,200 for married couples filing jointly).
# The other rates are:

# 35% for incomes over $243,725 ($487,450 for married couples filing jointly)
# 32% for incomes over $191,950 ($383,900 for married couples filing jointly)
# 24% for incomes over $100,525 ($201,050 for married couples filing jointly)
# 22% for incomes over $47,150 ($94,300 for married couples filing jointly)
# 12% for incomes over $11,600 ($23,200 for married couples filing jointly)

# The lowest rate is 10% for incomes of single individuals with incomes of
# $11,600 or less ($23,200 for married couples filing jointly).

def load_data():

    # tax brackets and rates
    beta = np.array([11.600, 47.150, 100.525, 191.950, 243.725, 609.350])
    eta = np.array([0.1, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37])

    # minimum required distributions
    distributions = pd.Series({ # https://smartasset.com/retirement/rmd-table
        72: 27.4,
        73: 26.5,
        74: 25.5,
        75: 24.6,
        76: 23.7,
        77: 22.9,
        78: 22.0,
        79: 21.1,
        80: 20.2,
        81: 19.4,
        82: 18.5,
        83: 17.7,
        84: 16.8,
        85: 16.0,
        86: 15.2,
        87: 14.4,
        88: 13.7,
        89: 12.9,
        90: 12.2,
        91: 11.5,
        92: 10.8,
        93: 10.1,
        94: 9.5,
        95: 8.9,
        96: 8.4,
        97: 7.8,
        98: 7.3,
        99: 6.8,
        100: 6.4,
        101: 6.0,
        102: 5.6,
        103: 5.2,
        104: 4.9,
        105: 4.6,
        106: 4.3,
        107: 4.1,
        108: 3.9,
        109: 3.7,
        110: 3.5,
        111: 3.4,
        112: 3.3,
        113: 3.1,
        114: 3.0,
        115: 2.9,
        116: 2.8,
        117: 2.7,
        118: 2.5,
        119: 2.3,
        120: 2.0
    })

    kappa = 1 / distributions
    kappa = kappa.reindex(range(0, 121), fill_value=0)

    # stock data; sp500
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500 = sp500["Close"].pct_change().dropna().resample('YE').sum()
    sp500.index = sp500.index.year

    treasury = yf.Ticker('^TNX')
    treasury = treasury.history(period='max')['Close'].resample('YE').mean() / 100

    inflation = pd.read_csv('../data/inflation.csv', index_col=0, skip_blank_lines=True).Annual.dropna() / 100


    return Data(beta,
                eta,
                distributions,
                kappa,
                sp500,
                treasury,
                inflation)

