"""Single source of truth for the logical-name -> metrics.json key map.

All extension reports (CPS shift, rhoalg sweep, KL smoothing, extended) import
from this module so the field map is recorded in exactly one place. The same
table is rendered into each *_report.md so reviewers can verify it.
"""
from typing import Any, Dict, List, Tuple

# logical name -> metrics.json key (under final.{train,val,test})
FIELD_MAP: Dict[str, str] = {
    "E[r*]":          "expected_true_utility",
    "R":              "sensitivity_R",
    "L_SAIL(P*)":     "nominal_bt_loss",
    "wcl(rho_mis)":   "wcl_true_rho_mis",
    "biased_bt_loss": "biased_bt_loss",
    "entropy":        "entropy",
    "top1":           "top1_prob",
    "theta_norm":     "theta_norm",
    "bad_mass":       "bad_bias_candidate_mass",
    "good_mass":      "good_candidate_mass",
    "E[bias_attr]":   "expected_bias_attr",
    "KL_to_ref":      "KL_to_ref",
}


def resolve(metrics_dict: Dict[str, Any], logical_name: str, *, default=None):
    """Look up a logical metric in a metrics.json `final.*` block.

    Raises KeyError with a clear message when the key is missing AND no
    default is provided. Use default=None for fields like KL_to_ref that may
    legitimately be missing on older runs.
    """
    if logical_name not in FIELD_MAP:
        raise KeyError(
            f"unknown logical metric {logical_name!r}; known: {list(FIELD_MAP)}"
        )
    key = FIELD_MAP[logical_name]
    if key not in metrics_dict:
        if default is not None:
            return default
        raise KeyError(
            f"metric key {key!r} (logical {logical_name!r}) not found in metrics dict; "
            f"available keys: {sorted(metrics_dict.keys())[:8]}..."
        )
    return metrics_dict[key]


def field_map_markdown_table() -> str:
    """Render FIELD_MAP as a markdown table for inclusion in *_report.md."""
    out = ["| logical name | metrics.json key |", "|---|---|"]
    for k, v in FIELD_MAP.items():
        out.append(f"| `{k}` | `{v}` |")
    return "\n".join(out)


def has_field(metrics_dict: Dict[str, Any], logical_name: str) -> bool:
    return FIELD_MAP.get(logical_name) in metrics_dict
