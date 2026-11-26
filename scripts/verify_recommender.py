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
    endpoint = args.base_url.rstrip("/") + "/api/v1/models/music-track-recommender/predictions"
    model_path = ROOT / "model/music_track_recommender.joblib"
    artifact = joblib.load(model_path)
    rules = artifact["rules_final"]
    lookup = artifact["track_id_to_name"]
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
                for track in rule["consequents"]:
                    if track not in played:
                        score = (float(rule["confidence"]), float(rule["lift"]), float(rule["support"]))
                        best[track] = max(best.get(track, score), score)
        ranked = sorted(best, key=lambda track: (-best[track][0], -best[track][1], -best[track][2], track))[:20]
        expected = [{
            "track_id": track, "track_name": lookup[track],
            "confidence": round(best[track][0], 4), "lift": round(best[track][1], 4),
            "support": round(best[track][2], 4),
        } for track in ranked]
        actual = post({"track_ids": list(basket), "top_n": 20}, 200)
        assert actual["recommendations"] == expected, (basket, actual, expected)

    demo = post({"track_ids": ["M01", "M03"], "top_n": 3}, 200)
    assert demo["recommendations"][0]["track_name"] == "Pump It Up"
    post({}, 400)
    post({"track_ids": []}, 400)
    post({"track_ids": ["UNKNOWN"]}, 400)
    post({"track_ids": ["M01"], "top_n": 0}, 400)
    empty = post({"track_ids": list(lookup)}, 200)
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
