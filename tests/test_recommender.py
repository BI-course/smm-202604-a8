"""Exercise the API against the actual saved association rules."""
import unittest
from pathlib import Path

import pandas as pd

import api


class RecommenderTests(unittest.TestCase):
    URL = "/api/v1/models/music-track-recommender/predictions"

    def setUp(self):
        self.client = api.app.test_client()

    def test_saved_scores_match_source_sessions(self):
        root = Path(__file__).resolve().parent.parent
        data = pd.read_csv(root / "data/music_streaming_sessions.csv")
        data["transaction_date"] = pd.to_datetime(data["transaction_date"])
        sessions = data.groupby("listening_session_id").agg(
            date=("transaction_date", "min"), tracks=("track_id", lambda ids: set(ids))
        )
        metadata = api.recommender_artifact["metadata"]
        split = pd.Timestamp(metadata["split_date"])
        earlier = sessions.loc[sessions.date <= split, "tracks"]
        later = sessions.loc[sessions.date > split, "tracks"]
        self.assertEqual(len(earlier), metadata["earlier_sessions"])
        self.assertEqual(len(later), metadata["later_sessions"])
        self.assertLessEqual(sessions.loc[earlier.index, "date"].max(), split)
        self.assertGreater(sessions.loc[later.index, "date"].min(), split)
        for rule in api.recommender_artifact["rules_final"].itertuples(index=False):
            with self.subTest(antecedent=rule.antecedents):
                a = sum(rule.antecedents <= tracks for tracks in earlier)
                c = sum(rule.consequents <= tracks for tracks in earlier)
                joint = sum((rule.antecedents | rule.consequents) <= tracks for tracks in earlier)
                self.assertAlmostEqual(rule.support, joint / len(earlier))
                self.assertAlmostEqual(rule.confidence, joint / a)
                self.assertAlmostEqual(rule.lift, (joint / a) / (c / len(earlier)))
                later_a = sum(rule.antecedents <= tracks for tracks in later)
                later_joint = sum((rule.antecedents | rule.consequents) <= tracks for tracks in later)
                self.assertEqual(rule.later_antecedent_count, later_a)
                self.assertAlmostEqual(rule.later_confidence, later_joint / later_a)
                self.assertTrue((rule.antecedents | rule.consequents).issubset(api.track_id_to_name))

    def test_real_saved_rule(self):
        response = self.client.post(self.URL, json={"track_ids": ["M01", "M03"], "top_n": 3})
        self.assertEqual(response.status_code, 200)
        first = response.get_json()["recommendations"][0]
        self.assertEqual(first, {
            "track_id": "M02", "track_name": "Pump It Up",
            "confidence": .8675, "lift": 5.9456, "support": .0389,
        })

    def test_all_saved_antecedents_against_rules_table(self):
        # Derive expected results from the saved rules TABLE, independently of
        # the precomputed lookup that the endpoint uses for inference.
        rules = api.recommender_artifact["rules_final"]
        baskets = {tuple(sorted(values)) for values in rules["antecedents"]}
        for basket in sorted(baskets):
            with self.subTest(cart=basket):
                played = set(basket)
                best = {}
                for _, rule in rules.iterrows():
                    if set(rule["antecedents"]).issubset(played):
                        for track in rule["consequents"]:
                            score = (float(rule["confidence"]), float(rule["lift"]), float(rule["support"]))
                            if track not in played:
                                best[track] = max(best.get(track, score), score)
                ranked = sorted(best, key=lambda track: (-best[track][0], -best[track][1], -best[track][2], track))[:20]
                expected = [{
                    "track_id": track,
                    "track_name": api.track_id_to_name[track],
                    "confidence": round(best[track][0], 4),
                    "lift": round(best[track][1], 4),
                    "support": round(best[track][2], 4),
                } for track in ranked]
                response = self.client.post(self.URL, json={"track_ids": list(basket), "top_n": 20})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["recommendations"], expected)

    def test_track_normalization_and_top_n(self):
        response = self.client.post(self.URL, json={"track_ids": [" M01 ", "M03", "M01"], "top_n": 1})
        result = response.get_json()
        self.assertEqual(result["track_ids"], ["M01", "M03"])
        self.assertEqual(len(result["recommendations"]), 1)

    def test_played_tracks_are_excluded(self):
        response = self.client.post(self.URL, json={"track_ids": list(api.track_id_to_name)})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["recommendations"], [])

    def test_missing_and_invalid_tracks(self):
        for payload in [None, [], "M01", {}, {"track_ids": None}, {"track_ids": "M01"},
                        {"track_ids": []}, {"track_ids": [""]}, {"track_ids": [" "]},
                        {"track_ids": [1]}, {"track_ids": [None]}, {"track_ids": [["M01"]]}]:
            with self.subTest(payload=payload):
                response = self.client.post(self.URL, json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_invalid_json_and_content_type(self):
        for body, content_type in [("{", "application/json"), ("M01", "text/plain")]:
            with self.subTest(content_type=content_type):
                response = self.client.post(self.URL, data=body, content_type=content_type)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_invalid_top_n(self):
        for top_n in [0, -1, 21, 2.5, True, "2", None]:
            with self.subTest(top_n=top_n):
                response = self.client.post(self.URL, json={"track_ids": ["M01"], "top_n": top_n})
                self.assertEqual(response.status_code, 400)

    def test_unknown_tracks(self):
        response = self.client.post(self.URL, json={"track_ids": ["UNKNOWN"]})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["unknown_track_ids"], ["UNKNOWN"])


if __name__ == "__main__":
    unittest.main()
