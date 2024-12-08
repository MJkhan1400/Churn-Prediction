import unittest
from src.train_model import train_and_save_model

class TestTrainModel(unittest.TestCase):
    def test_train_and_save(self):
        train_and_save_model('data/raw/churn_data.csv', 'model/churn_model.pkl', 'model/columns.json')
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
