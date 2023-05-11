import unittest
from app import create_gauge_chart, create_candlestick_chart, get_fear_and_greed_value, get_random_tweets
import pandas as pd
class DashboardTest(unittest.TestCase):

    def test_create_gauge_chart(self):
        # Test creating a gauge chart with a value of 50
        chart = create_gauge_chart(50)
        self.assertEqual(len(chart.data), 1)
        self.assertEqual(chart.data[0].type, 'indicator')
        self.assertEqual(chart.layout.width, 400)
        self.assertEqual(chart.layout.height, 400)

    def test_create_candlestick_chart(self):
        # Test creating a candlestick chart with sample data
        df = pd.DataFrame({
            'time': pd.date_range('2022-01-01', periods=5, freq='D'),
            'open': [100, 200, 300, 400, 500],
            'high': [150, 250, 350, 450, 550],
            'low': [50, 150, 250, 350, 450],
            'close': [125, 225, 325, 425, 525],
            'prediction': ['up', 'down', 'up', 'up', 'down']
        })
        chart = create_candlestick_chart(df, '2022-01-01', '2022-01-05', 50, 550)
        self.assertEqual(len(chart.data), 3)
        self.assertEqual(chart.data[0].type, 'candlestick')
        self.assertEqual(chart.data[1].type, 'scatter')
        self.assertEqual(chart.data[2].type, 'scatter')
        self.assertEqual(chart.layout.width, 1200)
        self.assertEqual(chart.layout.height, 650)

    def test_get_fear_and_greed_value(self):
        # Test getting the fear and greed index value
        value = get_fear_and_greed_value()
        self.assertGreaterEqual(value, 0)
        self.assertLessEqual(value, 100)

    def test_get_random_tweets(self):
        # Test getting two random tweets
        tweets_df = get_random_tweets()
        self.assertEqual(len(tweets_df), 2)
        self.assertIn('date', tweets_df.columns)
        self.assertIn('tweet', tweets_df.columns)

if __name__ == '__main__':
    unittest.main()
