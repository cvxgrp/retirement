import numpy as np
import pandas as pd
import yfinance as yf
from collections import namedtuple

import pandas as pd
import importlib.resources as pkg_resources

# # Assuming your CSVs are inside 'cvx/retirement/data'
# with pkg_resources.open_text('cvx.retirement.data', 'your_file.csv') as file:
#     df = pd.read_csv(file)


Data = namedtuple("Data", ["beta",
                            "eta",
                            "distributions",
                            "kappa",
                            "life_expectancy",
                            "mortality_rates",
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

### Data for life expectancy for men and women: https://www.ssa.gov/oact/STATS/table4c6.html
# url = 'https://www.ssa.gov/oact/STATS/table4c6.html'
# tables = pd.read_html(url)  # This reads all tables from the webpage
# df = tables[0]  # Assuming the first table is the one you need

# df_male = df['Male'].iloc[:-1]
# df_female = df['Female'].iloc[:-1]

# df_male.columns = ['death_prob', 'num_lives', 'life_expectancy']
# df_female.columns = ['death_prob', 'num_lives', 'life_expectancy']

# df_male.to_csv('../data/male_life_expectancy_data.csv')
# df_female.to_csv('../data/female_life_expectancy_data.csv')

### SP500 and treasury data (sp500 is from Kenneth French's data library; GSPC does not account for dividends)
# sp500 = yf.Ticker("^GSPC")
# sp500 = sp500.history(period="max")["Close"].resample('YE').last().pct_change().dropna()
# sp500.index = sp500.index.year

# treasury = yf.Ticker('^TNX')
# treasury = treasury.history(period='max')['Close'].resample('YE').mean()
# treasury.index = treasury.index.year

def load_data():

    # federal tax brackets and rates
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

    # Data for life expectancy at different ages https://www.irs.gov/pub/irs-pdf/p590b.pdf
    # life_expectancy = {
    #     'Age': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
    #             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
    #             39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
    #             57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 
    #             75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 
    #             93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 
    #             109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
    #     'Life Expectancy': [84.6, 83.7, 82.8, 81.8, 80.8, 79.8, 78.8, 77.9, 76.9, 75.9, 74.9, 
    #                         73.9, 72.9, 71.9, 70.9, 69.9, 69.0, 68.0, 67.0, 66.0, 65.0, 64.1, 
    #                         63.1, 62.1, 61.1, 60.2, 59.2, 58.2, 57.3, 56.3, 55.3, 54.4, 53.4, 
    #                         52.5, 51.5, 50.5, 49.6, 48.6, 47.7, 46.7, 45.7, 44.8, 43.8, 42.9, 
    #                         41.9, 41.0, 40.0, 39.0, 38.1, 37.1, 36.2, 35.3, 34.3, 33.4, 32.5, 
    #                         31.6, 30.6, 29.8, 28.9, 28.0, 27.1, 26.2, 25.4, 24.5, 23.7, 22.9, 
    #                         22.0, 21.2, 20.4, 19.6, 18.8, 18.0, 17.2, 16.4, 15.6, 14.8, 14.1, 
    #                         13.3, 12.6, 11.9, 11.2, 10.5, 9.9, 9.3, 8.7, 8.1, 7.6, 7.1, 6.6, 
    #                         6.1, 5.7, 5.3, 4.9, 4.6, 4.3, 4.0, 3.7, 3.4, 3.2, 3.0, 2.8, 2.6, 
    #                         2.5, 2.3, 2.2, 2.1, 2.1, 2.1, 2.0, 2.0, 2.0, 2.0, 2.0, 1.9, 1.9, 
    #                         1.8, 1.8, 1.6, 1.4, 1.1, 1.0]
    # }

    csv_files = ['male_life_expectancy_data.csv', 'female_life_expectancy_data.csv', 'ff3.csv', 'treasury.csv', 'inflation.csv']
    dfs = []
    for file_name in csv_files:
        with pkg_resources.open_text('cvx.retirement.data', file_name) as file:
            dfs.append(pd.read_csv(file, index_col=0))

    df_male, df_female, ff3, treasury, inflation = dfs
    sp500 = (ff3['Mkt-RF'] + ff3['RF']) / 100
    treasury = treasury / 100
    inflation = inflation.Annual.dropna() / 100

    # df_male = pd.read_csv('../data/male_life_expectancy_data.csv', index_col=0)
    # df_female = pd.read_csv('../data/female_life_expectancy_data.csv', index_col=0)

    life_expectancy = {
            'M' : df_male.life_expectancy,
            'F' : df_female.life_expectancy
    } 
    mortality_rates = {
        'M' : df_male.death_prob,
        'F' : df_female.death_prob
    }


    # # Data for mortality rates https://www.irs.gov/pub/irs-drop/n-23-73.pdf
    # mortality_rates = pd.Series([0.00331, 0.00024, 0.00015, 0.00011, 0.00010, 0.00008, 0.00008, 0.00007, 0.00006, 0.00005,
    #         0.00006, 0.00006, 0.00007, 0.00009, 0.00011, 0.00013, 0.00015, 0.00018, 0.00020, 0.00022,
    #         0.00023, 0.00024, 0.00024, 0.00026, 0.00027, 0.00027, 0.00028, 0.00030, 0.00030, 0.00032,
    #         0.00034, 0.00035, 0.00037, 0.00040, 0.00042, 0.00044, 0.00047, 0.00049, 0.00051, 0.00054,
    #         0.00056, 0.00057, 0.00059, 0.00061, 0.00063, 0.00066, 0.00070, 0.00074, 0.00079, 0.00084,
    #         0.00092, 0.00102, 0.00114, 0.00127, 0.00144, 0.00172, 0.00212, 0.00246, 0.00285, 0.00328,
    #         0.00379, 0.00433, 0.00512, 0.00591, 0.00656, 0.00740, 0.00832, 0.00920, 0.01017, 0.01126,
    #         0.01251, 0.01396, 0.01559, 0.01745, 0.01959, 0.02204, 0.02485, 0.02805, 0.03169, 0.03583,
    #         0.04079, 0.04583, 0.05150, 0.05787, 0.06508, 0.07327, 0.08259, 0.09310, 0.10497, 0.11814,
    #         0.13262, 0.14773, 0.16329, 0.17927, 0.19542, 0.21170, 0.22904, 0.24682, 0.26513, 0.28401,
    #         0.30325, 0.32271, 0.34211, 0.36140, 0.38043, 0.39884, 0.41675, 0.43400, 0.45044, 0.46615,
    #         0.47865, 0.48673, 0.49436, 0.49788, 0.49885, 0.49978, 0.49990, 0.49998, 0.50000, 0.50000,
    #         1.00000])
    
    # stock data; sp500
    # sp500 = pd.read_csv('../data/sp500.csv', index_col=0).squeeze() ### GSPC does not account for dividends
    # ff3 = pd.read_csv('../data/ff3.csv', index_col=0) / 100
    # sp500 = ff3['Mkt-RF'] + ff3['RF']
    # treasury = pd.read_csv('../data/treasury.csv', index_col=0).squeeze() / 100

    # only keep year
    sp500.index = pd.to_datetime(sp500.index, format='%Y').year
    treasury.index = pd.to_datetime(treasury.index, format='%Y').year
    inflation.index = pd.to_datetime(inflation.index, format='%Y').year

    inflation = inflation.loc[1962:2023].squeeze()
    treasury = treasury.loc[1962:2023].squeeze()

    return Data(beta,
                eta,
                distributions,
                kappa,
                life_expectancy,
                mortality_rates,
                sp500,
                treasury,
                inflation)

