import pandas as pd
import numpy as np

class Brokerage:
    def __init__(self, shares, basis, price):
        """
        parameters
        ----------
        shares: pd.Series
            The number of shares bought indexed by age
        basis: pd.Series
            The per share basis of the shares indexed by age
        price: pd.Series
            The price of the shares indexed by age
        """
        # assert index is positive int
        assert pd.api.types.is_integer_dtype(shares.index), "index must be integer"
        assert pd.api.types.is_integer_dtype(basis.index), "index must be integer"
        assert pd.api.types.is_numeric_dtype(price.index), "index must be numeric"
        assert shares.index.is_monotonic_increasing, "index must be increasing"
        assert basis.index.is_monotonic_increasing, "index must be increasing"
        assert (shares.index >= 0).all().all(), "index must be nonnegative"
        assert (basis.index >= 0).all().all(), "index must be nonnegative"
        assert (price.index >= 0).all().all(), "index must be nonnegative"
        assert shares.index.equals(basis.index), "shares and basis must have the same index"
        assert shares.index.equals(price.index), "shares and price must have the same index"

        self._shares = shares
        self._basis = basis
        self._price = price

        self._tax_brackets = np.array([47.025, 518.900])

    def set_price(self, age, price):
        """
        parameters
        ----------
        age: int
            age at which to set price
        price: float
            price of the shares
        """
        assert price > 0, "price must be positive"
        assert age not in self._price.index, "age already in index"

        self._price.loc[age] = price

    def buy(self, age, shares, basis):
        """
        parameters
        ----------
        age: int
            age at which to append
        shares: float
            number of shares to append
        basis: float
            basis of the shares to append
        """
        # assert age not in self._shares.index, "age already in index"
        # assert age not in self._basis.index, "age already in index"
        assert shares >= 0, "shares must be nonnegative"
        assert basis >= 0, "basis must be nonnegative"

        if age in self._shares.index:
            assert age in self._basis.index, "age not in index"
            
            total_shares = self._shares.loc[age] + shares
            total_basis = (self._basis.loc[age] * self._shares.loc[age] + basis * shares) / total_shares

        else:
            total_shares = shares
            total_basis = basis

        self._shares.loc[age] = total_shares
        self._basis.loc[age] = total_basis # basis
        self._price.loc[age] = total_basis # basis

    @property
    def tax_brackets(self):
        return self._tax_brackets
    
    @property
    def shares(self):
        return self._shares
    
    @property
    def basis(self):
        return self._basis
    
    @property
    def price(self):
        return self._price
    
    @staticmethod
    def capital_gain_tax(gain, taxable_income, brackets):

        ### XXX: heuristic used in optimization problem; optimization problem uses taxable_income = c0
        # if taxable_income < brackets[0]:
        #     return 0.
        # elif brackets[0] <= taxable_income < brackets[1]:
        #     return 0.15 * gain
        # else:
        #     return 0.2 * gain
        #############################
        # return 0.15 * gain

        gain_0 = max(min(gain, brackets[0] - taxable_income), 0)
        gain_15 = max(min(gain, brackets[1] - taxable_income) - gain_0, 0)
        gain_20 = gain - gain_0 - gain_15

        # print((gain_15 * 0.15 + gain_20 * 0.20) / gain)

        return gain_15 * 0.15 + gain_20 * 0.20
    
    def average_basis(self, age):
        """
        returns the average basis of the shares at a given age
        """
        return (self._shares.loc[:age] * self._basis.loc[:age]).sum() / self._shares.loc[:age].sum()
    

            
    def compute_tax(self, age, sell_shares, taxable_income):
        """
        parameters
        ----------
        sell_shares: pd.Series
            The number of shares to sell indexed by age
        price: float
            The price of the shares

        returns
        -------
        tax: float
            tax paid on the sale of the shares
        """
        price = self._price.loc[age]
        assert sell_shares.index.isin(self._shares.index).all(), "index must be in shares index"
        assert price > 0, "price must be positive"
        assert (sell_shares >= -1e-4).all(), "sell_shares must be nonnegative"

        gains = sell_shares * price - sell_shares * self.basis.loc[sell_shares.index]

        return self.capital_gain_tax(gains.sum(), taxable_income, self._tax_brackets)
        

    def compute_tax_average(self, age, sell_amount, taxable_income): 
        """
        parameters
        ----------
        age: int
            age at which to compute tax
        sell_amount: float
            amount to sell in dollars
        taxable_income: float
            taxable income

        returns
        -------
        tax: float
            tax paid using average basis

        ------
        Equivalent to selling sell_amount / price shares, with shares_i / sum(shares) shares sold at age i,
        i.e., 
        """
        price = self._price.loc[age]
        assert price > 0, "price must be positive"  

        if taxable_income < 0:
            raise ValueError("taxable income must be nonnegative")

        if sell_amount <= 0: 
            return 0
        
        avg_basis = self.average_basis(age)

        purchase_amount = avg_basis * sell_amount / price
        gain = sell_amount - purchase_amount

        return self.capital_gain_tax(gain, taxable_income, self._tax_brackets)
        

    def value(self, age):
        """
        parameters
        ----------
        age: int
            age at which to compute value
        price: float
            price of the shares

        returns
        -------
        value: float
            value of the shares at the given age
        """
        # assert age in self._shares.index, "age not in index"
        # assert age in self._basis.index, "age not in index"
        price = self._price.loc[age]
        
        return (self._shares.loc[:age] * price).sum()
        
    def sell(self, age, sell_amount, taxable_income): 
        """
        parameters
        ----------
        age: int
            age at which to sell
        sell_amount: float
            amount to sell in dollars
        price: float
            price of the shares

        returns
        -------
        tax: float
            tax paid
        """
        assert sell_amount >= -1e-4, "sell amount must be nonnegative"
        # assert age in self._shares.index, "age not in index"
        # assert age in self._basis.index, "age not in index"
        
        price = self._price.loc[age]

        assert (self.value(age) - sell_amount) >= -1e-4, "sell amount must be less than value"
        # assert sell_amount <= self.value(age), "sell amount must be less than value"

        # tax using average basis
        tax1 = self.compute_tax_average(age, sell_amount, taxable_income)

        # tax selling lots in proportion to their basis
        sell_shares = sell_amount / price * self._shares.loc[:age] / self._shares.loc[:age].sum()
        tax2 = self.compute_tax(age, sell_shares, taxable_income)

        assert np.isclose(tax1, tax2), "bug, average basis and basis proportional tax must be equal"

        self._shares.loc[:age] -= sell_shares

        return tax1


        
        
    

        
