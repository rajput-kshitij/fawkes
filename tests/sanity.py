import unittest
import sys
import os

#  This is so that the following imports work
sys.path.append(os.path.realpath("."))

import fawkes.algorithms.algo as algo
import fawkes.parse.parse as parse
import fawkes.utils.utils as utils
import fawkes.algorithms.categorisation.text_match.trainer as text_match_trainer

class FawkesSanityTest(unittest.TestCase):
    def test_sanity(self):
        """
        Test for sanity that parsing and algorithms are working
        """
        # First we parse the sample data.
        parse.parse_reviews()
        parsed_output = utils.open_json(
            "data/parsed_data/sample-mint/parsed-user-feedback.json"
        )
        expected_parsed_output = [
            {
                "message": "I just heard about this budgeting app. So I gave it a try. I am impressed thus far. However I still cant add all of my financial institutions so my budget is kind of skewed. But other that I can say Im more aware of my spending",
                "timestamp": "2020/03/15 22:06:17",
                "rating": 5.0,
                "user_id": None,
                "app_name": "sample-mint",
                "channel_name": "appstore",
                "channel_type": "ios",
                "hash_id": "de848685d11742dbea77e1e5ad7b892088ada9c9",
                "derived_insight": {
                    "sentiment": None,
                    "category": "uncategorized",
                    "extra_properties": {}
                }
            }
        ]
        self.assertEqual(parsed_output, expected_parsed_output)
        # Before running the algorithms, we generate the keyword weights.
        text_match_trainer.generate_keyword_weights()

        # We run the algorithms on that data
        algo.run_algo()

        processed_output = utils.open_json(
            "data/processed_data/sample-mint/processed-user-feedback.json"
        )
        expected_processed_output = [
            {
                "message": "I just heard about this budgeting app. So I gave it a try. I am impressed thus far. However I still cant add all of my financial institutions so my budget is kind of skewed. But other that I can say Im more aware of my spending",
                "timestamp": "2020/03/15 22:06:17",
                "rating": 5.0,
                "user_id": None,
                "app_name": "sample-mint",
                "channel_name": "appstore",
                "channel_type": "ios",
                "hash_id": "17663da08ed71f2664446e4c1b2f2a4417fcbb68",
                "derived_insight": {
                    "sentiment": {
                        "neg": 0.0,
                        "neu": 0.951,
                        "pos": 0.049,
                        "compound": 0.2617
                    },
                    "category": "Application",
                    "extra_properties": {
                        "category_scores": {
                            "User Experience": 0,
                            "sign-in/sign-up": 0,
                            "Notification": 0,
                            "Application": 1,
                            "ads": 0
                        },
                        "bug_feature": "feature"
                    }
                }
            }
        ]
        self.assertEqual(processed_output, expected_processed_output)

if __name__ == "__main__":
    unittest.main()
