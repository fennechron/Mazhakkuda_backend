import unittest
import json
from app import app

class TestTravelAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_travel(self):
        payload = {
            "start": "Kumbanad",
            "dest": "Chengannur",
            "simulate": True
        }
        response = self.app.post('/api/predict/travel', 
                                 data=json.dumps(payload), 
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('trip_timestamp', data)
        self.assertIn('checkpoints', data)
        self.assertTrue(len(data['checkpoints']) > 0)
        
        # Save timestamp for feedback test
        self.trip_timestamp = data['trip_timestamp']
        self.checkpoint = data['checkpoints'][0]['name']
        print("Prediction successful, timestamp:", self.trip_timestamp)

if __name__ == '__main__':
    unittest.main()
