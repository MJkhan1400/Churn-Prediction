import unittest
from src.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    def test_preprocessing(self):
        X, y = preprocess_data('data/raw/churn_data.csv')
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(len(X), len(y))

if __name__ == '__main__':
    unittest.main()
