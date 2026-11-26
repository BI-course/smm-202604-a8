"""Exercise the API against the actual saved association rules."""
import unittest

import api


class RecommenderTests(unittest.TestCase):
    URL = "/api/v1/models/ecommerce-recommender/predictions"

    def setUp(self):
        self.client = api.app.test_client()

    def test_real_saved_rule(self):
        response = self.client.post(self.URL, json={"cart": ["H06", "H07"], "top_n": 3})
        self.assertEqual(response.status_code, 200)
        first = response.get_json()["recommendations"][0]
        self.assertEqual(first, {
            "sku": "H03", "product_name": "Electric Kettle",
            "confidence": .9385, "lift": 12.7163,
        })

    def test_all_saved_antecedents_against_rules_table(self):
        # Derive expected results from the saved rules TABLE, independently of
        # the precomputed lookup that the endpoint uses for inference.
        rules = api.recommender_artifact["cleaned_rules"]
        baskets = {tuple(sorted(values)) for values in rules["antecedents"]}
        for basket in sorted(baskets):
            with self.subTest(cart=basket):
                played = set(basket)
                best = {}
                for _, rule in rules.iterrows():
                    if set(rule["antecedents"]).issubset(played):
                        for sku in rule["consequents"]:
                            score = (float(rule["confidence"]), float(rule["lift"]))
                            if sku not in played:
                                best[sku] = max(best.get(sku, score), score)
                ranked = sorted(best, key=lambda sku: (-best[sku][0], -best[sku][1], sku))[:20]
                expected = [{
                    "sku": sku,
                    "product_name": str(api.sku_lookup.loc[sku, "product_name"]),
                    "confidence": round(best[sku][0], 4),
                    "lift": round(best[sku][1], 4),
                } for sku in ranked]
                response = self.client.post(self.URL, json={"cart": list(basket), "top_n": 20})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["recommendations"], expected)

    def test_cart_normalization_and_top_n(self):
        response = self.client.post(self.URL, json={"cart": [" H06 ", "H07", "H06"], "top_n": 1})
        result = response.get_json()
        self.assertEqual(result["cart"], ["H06", "H07"])
        self.assertEqual(len(result["recommendations"]), 1)

    def test_products_already_in_cart_are_excluded(self):
        response = self.client.post(self.URL, json={"cart": list(api.sku_lookup.index)})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["recommendations"], [])

    def test_missing_and_invalid_cart(self):
        for payload in [None, [], "H06", {}, {"cart": None}, {"cart": "H06"},
                        {"cart": []}, {"cart": [""]}, {"cart": [" "]},
                        {"cart": [1]}, {"cart": [None]}, {"cart": [["H06"]]}]:
            with self.subTest(payload=payload):
                response = self.client.post(self.URL, json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_invalid_json_and_content_type(self):
        for body, content_type in [("{", "application/json"), ("H06", "text/plain")]:
            with self.subTest(content_type=content_type):
                response = self.client.post(self.URL, data=body, content_type=content_type)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_invalid_top_n(self):
        for top_n in [0, -1, 21, 2.5, True, "2", None]:
            with self.subTest(top_n=top_n):
                response = self.client.post(self.URL, json={"cart": ["H06"], "top_n": top_n})
                self.assertEqual(response.status_code, 400)

    def test_unknown_skus(self):
        response = self.client.post(self.URL, json={"cart": ["UNKNOWN"]})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["unknown_skus"], ["UNKNOWN"])


if __name__ == "__main__":
    unittest.main()
