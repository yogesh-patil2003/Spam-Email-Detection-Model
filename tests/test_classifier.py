import os
import unittest
from spam_classifier import load_data, train_and_evaluate, predict_message


class TestSpamClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.csv_path = os.path.join(cls.repo_dir, "mail_data.csv")
        cls.model, cls.vectorizer, cls.metrics = train_and_evaluate(
            cls.csv_path, save_cm=False
        )

    def test_dataset_shape(self):
        df = load_data(self.csv_path)
        self.assertEqual(len(df), 5572)
        self.assertEqual(set(df["label"].unique()), {0, 1})

    def test_accuracy_above_threshold(self):
        self.assertGreater(self.metrics["test_accuracy"], 0.96)
        self.assertGreater(self.metrics["train_accuracy"], 0.96)

    def test_spam_and_ham_predictions(self):
        spam_msg = "WINNER!! Claim your FREE prize now! Call 09061701461."
        ham_msg = "Hi, let us meet after work for coffee at 5pm."
        self.assertEqual(predict_message(self.model, self.vectorizer, spam_msg), "Spam")
        self.assertEqual(predict_message(self.model, self.vectorizer, ham_msg), "Ham (Not Spam)")


if __name__ == "__main__":
    unittest.main()
