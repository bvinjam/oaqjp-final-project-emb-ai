import unittest
from emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
  
    def test_joy_dominant(self):
        statement = "I am glad this happened"
        result = emotion_detector(statement)
        self.assertEqual(result['dominant_emotion'], 'joy')

    def test_anger_dominant(self):
        statement = "I am really mad about this"
        result = emotion_detector(statement)
        self.assertEqual(result['dominant_emotion'], 'anger')

    def test_disgust_dominant(self):
        statement = "I feel disgusted just hearing about this"
        result = emotion_detector(statement)
        self.assertEqual(result['dominant_emotion'], 'disgust')

    def test_sadness_dominant(self):
        statement = "I am so sad about this"
        result = emotion_detector(statement)
        self.assertEqual(result['dominant_emotion'], 'sadness')

    def test_fear_dominant(self):
        statement = "I am really afraid that this will happen"
        result = emotion_detector(statement)
        self.assertEqual(result['dominant_emotion'], 'fear')

    def test_blank_input(self):
        result = emotion_detector("")
        self.assertIsNone(result['dominant_emotion'])
        self.assertIsNone(result['anger'])
        self.assertIsNone(result['disgust'])
        self.assertIsNone(result['fear'])
        self.assertIsNone(result['joy'])
        self.assertIsNone(result['sadness'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
