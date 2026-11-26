"""Verify a running API against its saved rules and record live HTTP evidence."""
import argparse
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import requests


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    endpoint = args.base_url.rstrip("/") + "/api/v1/models/ecommerce-recommender/predictions"
    model_path = ROOT / "model/apriori_recommendation_rules.joblib"
    artifact = joblib.load(model_path)
    rules = artifact["cleaned_rules"]
    lookup = artifact["sku_lookup"]
    records = []

    def post(payload, expected_status):
        response = requests.post(endpoint, json=payload, headers={"Origin": "null"}, timeout=15)
        assert response.status_code == expected_status, (payload, response.status_code, response.text)
        assert response.headers.get("Access-Control-Allow-Origin") in ("*", "null")
        body = response.json()
        records.append({"request": payload, "status": response.status_code, "response": body})
        return body

    # Compare live HTTP responses against the rules table, rather than the
    # precomputed lookup the API uses. No mocked models or synthetic rules.
    baskets = {tuple(sorted(values)) for values in rules["antecedents"]}
    for basket in sorted(baskets):
        played = set(basket)
        best = {}
        for _, rule in rules.iterrows():
            if set(rule["antecedents"]).issubset(played):
                for sku in rule["consequents"]:
                    if sku not in played:
                        score = (float(rule["confidence"]), float(rule["lift"]))
                        best[sku] = max(best.get(sku, score), score)
        ranked = sorted(best, key=lambda sku: (-best[sku][0], -best[sku][1], sku))[:20]
        expected = [{
            "sku": sku, "product_name": str(lookup.loc[sku, "product_name"]),
            "confidence": round(best[sku][0], 4), "lift": round(best[sku][1], 4),
        } for sku in ranked]
        actual = post({"cart": list(basket), "top_n": 20}, 200)
        assert actual["recommendations"] == expected, (basket, actual, expected)

    demo = post({"cart": ["H06", "H07"], "top_n": 3}, 200)
    assert demo["recommendations"][0]["product_name"] == "Electric Kettle"
    post({}, 400)
    post({"cart": []}, 400)
    post({"cart": ["UNKNOWN"]}, 400)
    post({"cart": ["H06"], "top_n": 0}, 400)
    empty = post({"cart": list(lookup.index)}, 200)
    assert empty["recommendations"] == []

    report = {
        "tested_at": datetime.now(timezone(timedelta(hours=3))).isoformat(),
        "endpoint": endpoint,
        "artifact": str(model_path.relative_to(ROOT)),
        "artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "saved_rule_count": len(rules),
        "distinct_antecedent_baskets_verified": len(baskets),
        "all_checks_passed": True,
        "requests": records,
    }
    destination = ROOT / "evidence/recommender_live_test.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Passed {len(records)} live HTTP checks across {len(baskets)} saved antecedent baskets.")
    print(json.dumps(demo, indent=2))
    print(f"Evidence saved to {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
