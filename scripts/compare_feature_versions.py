"""
compare_feature_versions.py — For each versioned feature in feature_spec.md (and any other
feature that has accumulated more than one version), load every version from the feature
store (never recomputed) and compare consecutive versions: values changed, distribution
shift, rank-order change, and customers crossing a churn-risk flag. Writes VERSION_COMPARISON.md.

No churn model exists yet, so the "risk flag" is a per-feature rule on the feature itself
(RISK_FLAGS). The churned label is joined only to describe how well each version's flag
separates churners — it is never a feature input.

    python scripts/compare_feature_versions.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

from feature_store import store
from features import ingest
from features.build import DEFAULT_SOURCE_PATHS
from features.schema import VERSIONED_FEATURES

# (description, rule) — the customers a churn ranking built on this feature would flag.
RISK_FLAGS = {
    "txn_count_window": ("no transaction in the window (value == 0)", lambda s: s == 0),
    "return_count_recent": ("at least one recent return (value >= 1)", lambda s: s >= 1),
    "spend_ratio_last_3m_vs_previous": ("recent spend under half the own baseline (value < 0.5)", lambda s: s < 0.5),
}


def load_versions(feature: str) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    meta = store.list_versions(feature).sort_values("version")
    return meta, {int(v): store.get_version(feature, int(v)) for v in meta["version"]}


def churn_rate(flag: pd.Series, churned: pd.Series) -> tuple[float, float]:
    on, off = churned[flag], churned[~flag]
    return (on.mean() * 100 if len(on) else float("nan"), off.mean() * 100 if len(off) else float("nan"))


def auc(values: pd.Series, churned: pd.Series) -> float:
    """Mann-Whitney AUC of the raw feature for churned=1 (0.5 = no separation, ties averaged)."""
    ranks = values.rank()
    pos = churned == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compare(feature: str, labels: pd.DataFrame) -> dict:
    meta, data = load_versions(feature)
    flag_desc, flag_rule = RISK_FLAGS.get(feature, ("top decile of the feature", lambda s: s >= s.quantile(0.9)))

    summary = []
    for _, row in meta.iterrows():
        v = int(row["version"])
        s = data[v][feature]
        flag = flag_rule(s)
        merged = data[v][["customer_id"]].assign(flag=flag.values).merge(labels, on="customer_id")
        on_rate, off_rate = churn_rate(merged["flag"], merged["churned"])
        summary.append({
            "version": v, "params": row["parameters"], "is_latest": int(row["is_latest"]),
            "n": len(s), "mean": s.mean(), "std": s.std(),
            "q": s.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).tolist(),
            "flagged": int(flag.sum()), "churn_flagged": on_rate, "churn_unflagged": off_rate,
            "auc": auc(data[v][[feature]].assign(customer_id=data[v]["customer_id"]).merge(labels, on="customer_id")[feature],
                       data[v][["customer_id"]].merge(labels, on="customer_id")["churned"]),
            "rationale": row["business_rationale"],
        })

    pairs = []
    versions = sorted(data)
    for a, b in zip(versions, versions[1:]):
        m = data[a][["customer_id", feature]].merge(
            data[b][["customer_id", feature]], on="customer_id", suffixes=("_a", "_b"))
        va, vb = m[f"{feature}_a"], m[f"{feature}_b"]
        fa, fb = flag_rule(va), flag_rule(vb)
        pairs.append({
            "a": a, "b": b, "n": len(m),
            "changed": int(((va - vb).abs() > 1e-12).sum()),
            "mean_shift": vb.mean() - va.mean(),
            "spearman": va.rank().corr(vb.rank()),
            "flag_on": int((~fa & fb).sum()), "flag_off": int((fa & ~fb).sum()),
        })
    return {"feature": feature, "flag_desc": flag_desc, "summary": summary, "pairs": pairs}


def interpret(r: dict) -> list[str]:
    s0, s1 = r["summary"][-2], r["summary"][-1]
    p = r["pairs"][-1]
    noticeable = p["changed"] / p["n"] > 0.05 or p["spearman"] < 0.95
    return [
        f"- {p['changed']} of {p['n']} customers ({p['changed'] / p['n']:.1%}) change value; "
        f"Spearman rank correlation v{p['a']}→v{p['b']} is {p['spearman']:.3f}.",
        f"- {p['flag_on'] + p['flag_off']} customers cross the risk flag ({r['flag_desc']}): "
        f"{p['flag_on']} newly flagged, {p['flag_off']} no longer flagged.",
        "- **Would a model notice?** "
        + ("Yes — enough customers move and the ranking reorders enough that a model trained on one version "
           "and scored on the other would see a different input distribution; the version must be pinned "
           "identically at training and scoring time (scripts/score_customer.py reads it from the store)."
           if noticeable else
           "Barely — few customers move and rank order is essentially preserved, so a consuming model's output "
           "would change only for the affected customers."),
        f"- Univariate AUC vs `churned` (descriptive only, see the note at the top): v{s0['version']} {s0['auc']:.3f}, "
        f"v{s1['version']} {s1['auc']:.3f}.",
    ]


def label_rule_match(labels: pd.DataFrame) -> dict:
    """How often the recency rule `has_txn_history == 0 or days_since_last_txn >= 90`, read from the
    stored latest snapshots, reproduces `churned`. Near-total agreement means the features restate the
    label (as they did at 2026-08-01); the training as_of is chosen to avoid that."""
    snap = (store.get_latest("has_txn_history")
            .merge(store.get_latest("days_since_last_txn")[["customer_id", "days_since_last_txn"]], on="customer_id")
            .merge(labels, on="customer_id"))
    rule = ((snap["has_txn_history"] == 0) | (snap["days_since_last_txn"] >= 90)).astype(int)
    return {"as_of": ", ".join(map(str, snap["as_of_date"].unique())), "n": len(snap),
            "match": int((rule == snap["churned"]).sum()), "churn_rate": snap["churned"].mean() * 100}


def to_markdown(results: list[dict], spec_names: set[str], rule: dict) -> str:
    out = [
        "# Version Comparison",
        "",
        "Generated by `scripts/compare_feature_versions.py` from snapshots in the feature store "
        "(`feature_store.store.get_version`), not recomputed. Consecutive versions are compared. "
        "There is no churn model yet, so each feature gets a simple risk flag; `churned` is joined "
        "only to describe how well each version's flag separates churners and is never a feature input.",
        "",
        f"> **Population and label.** Snapshots were built at as_of = {rule['as_of']}: {rule['n']:,} customers "
        f"with `signup_date < as_of`, churn rate {rule['churn_rate']:.1f}%. The rule "
        "`has_txn_history == 0 or days_since_last_txn >= 90` on these snapshots reproduces `churned` for "
        f"{rule['match']:,} of {rule['n']:,} customers. (At 2026-08-01 the same rule matched 1,499 of 1,500, which is "
        "why feature_spec.md moved the training as_of to the start of the apparent 90-day label window.) "
        "Churn rates and AUCs below are single-date, univariate descriptions, not evidence that one version "
        "predicts better.",
        "",
    ]
    for r in results:
        tag = "spec-defined versioning exercise" if r["feature"] in spec_names else \
            "revision outside the spec's versioning exercise (computation fix)"
        out += [f"## {r['feature']} — {tag}", "", "| Version | Parameters | Latest | Mean | Std | Min | P25 | P50 | P75 | P90 | P99 | Max | Flagged | Churn % flagged | Churn % not | AUC |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in r["summary"]:
            q = " | ".join(f"{x:.2f}" for x in s["q"])
            out.append(f"| v{s['version']} | `{s['params']}` | {'yes' if s['is_latest'] else ''} | {s['mean']:.3f} | "
                       f"{s['std']:.3f} | {q} | {s['flagged']} | {s['churn_flagged']:.1f} | {s['churn_unflagged']:.1f} | {s['auc']:.3f} |")
        out.append("")
        for s in r["summary"]:
            out.append(f"- **v{s['version']} rationale:** {s['rationale']}")
        out.append("")
        for p in r["pairs"]:
            out.append(f"**v{p['a']} → v{p['b']}:** {p['changed']} values changed, mean shift {p['mean_shift']:+.3f}, "
                       f"Spearman {p['spearman']:.3f}, {p['flag_on']} newly flagged / {p['flag_off']} unflagged.")
        out += ["", "### Interpretation", "", *interpret(r), ""]
    return "\n".join(out)


def main():
    os.chdir(ROOT)  # data/ and feature_store/ paths are repo-relative
    labels = ingest.load_customers_csv(DEFAULT_SOURCE_PATHS["customers"])[["customer_id", "churned"]]
    spec_names = [vf.name for vf in VERSIONED_FEATURES]
    extra = [f for f in store.list_feature_names() if f not in spec_names and len(store.list_versions(f)) > 1]
    results = []
    for feature in spec_names + extra:
        r = compare(feature, labels)
        results.append(r)
        for p in r["pairs"]:
            print(f"{feature} v{p['a']}->v{p['b']}: changed={p['changed']} spearman={p['spearman']:.3f} "
                  f"flag +{p['flag_on']}/-{p['flag_off']}")
    with open("VERSION_COMPARISON.md", "w", encoding="utf-8") as f:
        f.write(to_markdown(results, set(spec_names), label_rule_match(labels)))
    print("Wrote VERSION_COMPARISON.md")


if __name__ == "__main__":
    main()
