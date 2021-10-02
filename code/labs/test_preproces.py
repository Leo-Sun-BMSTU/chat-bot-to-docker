import unittest
from numpy.random import uniform
from numpy.random import randint
import pandas as pd
from code.labs.preproces import Prepropcess

class TestPreprocess(unittest.TestCase):

    def setUp(self):
        csv_data = pd.DataFrame({'X1': uniform(-1, 1, 400),
                                 'X2': uniform(-1, 1, 400),
                                 'target_var': randint(0, 2, 400)}).to_csv('data.csv')
        self.prep = Prepropcess('data.csv')

    def test_is_last_column_target(self):
        self.assert_(self.prep.is_last_column_target(), bool)

    def test_row_number(self):
        self.assert_(self.prep.row_number(), bool)

    def test_nan_duplicate_search(self):
        self.assert_(self.prep.nan_duplicate_search(), bool)

    def test_normalization(self):
        self.assert_(self.prep.normalization(), bool)

    def test_correlation(self):
        self.assert_(self.prep.correlation(), bool)

    def test_data_type_checker(self):
        self.assert_(self.prep.data_type_checker(), bool)

    def test_run(self):
        self.assert_(self.prep.run(), bool)

if __name__ == "__main__":
    unittest.main()