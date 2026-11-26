"""Build the missing music artifact offline from the earlier lab's dataset.

This completes an unfinished notebook, rather than recovering historical rules.
Only the earlier period selects rules; the later period is a held-out audit.
The small 16-track catalogue makes a dependency-free Apriori implementation
practical, keeping the saved artifact compatible with the API's pinned pandas.
"""
import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SOURCE_URL = "https://raw.githubusercontent.com/course-files/ClusteringandAssociationRuleMining/main/data/music_streaming_sessions.csv"
MIN_SUPPORT = .01  # Selected in the earlier notebook.
MIN_CONFIDENCE = .5
MIN_LIFT = 1.2


def frequent_itemsets(baskets, min_support):
    """Apriori: join frequent sets, prune infrequent subsets, count candidates."""
    minimum_count = math.ceil(len(baskets) * min_support)
    singles = Counter(track for basket in baskets for track in basket)
    previous = {frozenset([track]): count for track, count in singles.items() if count >= minimum_count}
    counts = dict(previous)
    size = 2
    while previous:
        prior = list(previous)
        candidates = {left | right for left, right in combinations(prior, 2) if len(left | right) == size}
        candidates = {items for items in candidates if all(
            frozenset(subset) in previous for subset in combinations(items, size - 1)
        )}
        if not candidates:
            break
        observed = Counter()
        for basket in baskets:
            for values in combinations(sorted(basket), size):
                itemset = frozenset(values)
                if itemset in candidates:
                    observed[itemset] += 1
        previous = {items: count for items, count in observed.items() if count >= minimum_count}
        counts.update(previous)
        size += 1
    return counts


def generate_rules(counts, session_count):
    rules = []
    for itemset, joint_count in counts.items():
        for size in range(1, len(itemset)):
            for values in combinations(sorted(itemset), size):
                antecedent = frozenset(values)
                consequent = itemset - antecedent
                confidence = joint_count / counts[antecedent]
                rules.append({
                    "antecedents": antecedent, "consequents": consequent,
                    "support": joint_count / session_count,
                    "confidence": confidence,
                    "lift": confidence / (counts[consequent] / session_count),
                    "antecedent_count": counts[antecedent], "joint_count": joint_count,
                })
    return rules


def rule_order(rule):
    return (-rule["confidence"], -rule["lift"], -rule["support"],
            tuple(sorted(rule["antecedents"])), tuple(sorted(rule["consequents"])))


def clean_rules(rules):
    filtered = [rule for rule in rules if rule["confidence"] >= MIN_CONFIDENCE and rule["lift"] >= MIN_LIFT]
    subset_cleaned = [rule for rule in filtered if not any(
        other["consequents"] == rule["consequents"]
        and other["antecedents"] < rule["antecedents"]
        and other["confidence"] >= rule["confidence"]
        for other in filtered
    )]
    cleaned, seen = [], set()
    for rule in sorted(subset_cleaned, key=rule_order):
        direction_pair = frozenset([rule["antecedents"], rule["consequents"]])
        if direction_pair not in seen:
            cleaned.append(dict(rule))
            seen.add(direction_pair)
    return cleaned, len(filtered), len(subset_cleaned)


def main():
    source = ROOT / "data/music_streaming_sessions.csv"
    frame = pd.read_csv(source)
    required = {"listening_session_id", "transaction_date", "track_id", "item_name", "category", "duration_minutes"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError("Dataset contains missing columns or values")
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"], errors="raise")
    if frame.groupby("track_id")[["item_name", "category"]].nunique().gt(1).any().any():
        raise ValueError("Each track ID must have one stable name and category")
    if frame.groupby("listening_session_id")["transaction_date"].nunique().gt(1).any():
        raise ValueError("A session must belong to one date")

    lines = frame.drop_duplicates(["listening_session_id", "track_id"])
    sessions = lines.groupby("listening_session_id").agg(
        date=("transaction_date", "min"), tracks=("track_id", lambda values: frozenset(values))
    )
    split = sessions["date"].quantile(.8)
    earlier = list(sessions.loc[sessions.date <= split, "tracks"])
    later = list(sessions.loc[sessions.date > split, "tracks"])
    if not earlier or not later:
        raise ValueError("Both temporal periods must contain sessions")
    counts = frequent_itemsets(earlier, MIN_SUPPORT)
    candidates = generate_rules(counts, len(earlier))
    final, filtered_count, subset_count = clean_rules(candidates)
    if not final:
        raise ValueError("No rules survived the documented thresholds")

    validation = []
    for rule in final:
        antecedent, consequent = rule["antecedents"], rule["consequents"]
        antecedent_count = sum(antecedent <= basket for basket in later)
        consequent_count = sum(consequent <= basket for basket in later)
        joint_count = sum((antecedent | consequent) <= basket for basket in later)
        confidence = joint_count / antecedent_count if antecedent_count else None
        lift = confidence / (consequent_count / len(later)) if antecedent_count and consequent_count else None
        rule.update({
            "later_antecedent_count": antecedent_count,
            "later_joint_count": joint_count,
            "later_confidence": confidence,
            "later_lift": lift,
        })
        validation.append({
            "antecedents": sorted(antecedent), "consequents": sorted(consequent),
            "earlier_confidence": rule["confidence"], "later_confidence": confidence,
            "confidence_change": confidence - rule["confidence"] if confidence is not None else None,
            "later_lift": lift, "later_antecedent_count": antecedent_count,
            "meets_thresholds_later": confidence is not None and lift is not None
            and confidence >= MIN_CONFIDENCE and lift >= MIN_LIFT,
        })

    catalogue = lines[["track_id", "item_name", "category"]].drop_duplicates().sort_values("track_id").reset_index(drop=True)
    names = dict(zip(catalogue.track_id, catalogue.item_name))
    lookup = {}
    for rule in final:
        for track in sorted(rule["consequents"]):
            lookup.setdefault(rule["antecedents"], []).append(
                (track, rule["confidence"], rule["lift"], rule["support"])
            )
    summary = pd.DataFrame(candidates)[["support", "confidence", "lift"]].describe().to_dict()
    metadata = {
        "provenance": "New reconstruction completing the unfinished earlier-lab notebook; not recovered historical rules",
        "source_url": SOURCE_URL, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "algorithm": "Apriori with candidate joining and subset pruning",
        "min_support": MIN_SUPPORT, "min_confidence": MIN_CONFIDENCE, "min_lift": MIN_LIFT,
        "threshold_rationale": "Keep the notebook's 1% support. Training candidate confidence median is about 0.116 and upper quartile about 0.412; 0.5 selects stronger majority co-listening. Lift 1.2 requires at least 20% above baseline. Choices use only the earlier period.",
        "raw_rows": len(frame), "sessions": len(sessions), "tracks": len(names),
        "split_date": split.isoformat(), "earlier_sessions": len(earlier), "later_sessions": len(later),
        "minimum_earlier_session_count": math.ceil(len(earlier) * MIN_SUPPORT),
        "frequent_itemsets": len(counts), "candidate_rules": len(candidates),
        "filtered_rules": filtered_count, "after_subset_pruning": subset_count, "final_rules": len(final),
        "candidate_metric_summary": summary,
        "validation_policy": "Later sessions audit fixed earlier-selected rules; held-out scores never select or remove deployed rules",
        "rules_meeting_thresholds_later": sum(rule["meets_thresholds_later"] for rule in validation),
        "temporal_validation": validation,
    }
    artifact = {
        "rules_final": pd.DataFrame(final),
        "track_id_to_name": names,
        "track_lookup_df": catalogue,
        "recommendation_lookup": lookup,
        "metadata": metadata,
    }
    destination = ROOT / "model/music_track_recommender.joblib"
    joblib.dump(artifact, destination)
    loaded = joblib.load(destination)
    pd.testing.assert_frame_equal(loaded["rules_final"], artifact["rules_final"])
    assert loaded["track_id_to_name"] == names
    assert loaded["recommendation_lookup"] == lookup
    report = ROOT / "evidence/music_rules_build.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    readable = artifact["rules_final"].copy()
    for column in ["antecedents", "consequents"]:
        readable[column + "_names"] = readable[column].map(lambda values: "; ".join(names[track] for track in sorted(values)))
        readable[column] = readable[column].map(lambda values: ", ".join(sorted(values)))
    readable.to_csv(ROOT / "evidence/music_rules.csv", index=False)
    print(f"Saved {len(final)} music rules and {len(names)} track names to {destination.relative_to(ROOT)}")
    print(f"Earlier/later sessions: {len(earlier)}/{len(later)}; split: {split.date()}")
    print(f"Itemsets/candidates/filtered/subset-cleaned/final: {len(counts)}/{len(candidates)}/{filtered_count}/{subset_count}/{len(final)}")
    print(f"Rules meeting thresholds in the later audit: {metadata['rules_meeting_thresholds_later']}/{len(final)}")
    for rule in final:
        print(sorted(rule['antecedents']), '->', sorted(rule['consequents']),
              f"confidence={rule['confidence']:.4f}, lift={rule['lift']:.4f}")


if __name__ == "__main__":
    main()
