#!/usr/bin/env python3
"""Schema validator for surrogate-program design_space.yaml.

Validates against the expected structure and prints a status report.
Usage: python validate_design_space.py <path>
"""

import json
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

SCHEMA = {
    "version": {"type": int, "required": True, "valid": [1]},
    "description": {"type": str, "required": False},
    "fibres": {
        "type": list,
        "required": True,
        "item_keys": {
            "name": {"type": str, "required": True},
            "description": {"type": str, "required": False},
            "source": {"type": str, "required": True},
            "type": {"type": str, "required": True, "valid": ["transverse_isotropic"]},
            "e_l": {"type": (int, float), "required": True},
            "e_t": {"type": (int, float), "required": True},
            "g_lt": {"type": (int, float), "required": True},
            "nu_lt": {"type": (int, float), "required": True},
            "nu_tt": {"type": (int, float), "required": True},
            "density_kg_m3": {"type": (int, float), "required": False},
        },
    },
    "matrices": {
        "type": list,
        "required": True,
        "item_keys": {
            "name": {"type": str, "required": True},
            "description": {"type": str, "required": False},
            "source": {"type": str, "required": True},
            "type": {"type": str, "required": True, "valid": ["isotropic"]},
            "youngs_modulus": {"type": (int, float), "required": True},
            "poisson_ratio": {"type": (int, float), "required": True},
        },
    },
    "weave_architectures": {
        "type": list,
        "required": True,
        "item_keys": {
            "name": {
                "type": str,
                "required": True,
                "valid": [
                    "plain",
                    "twill",
                    "satin",
                    "basket",
                    "3d_orthogonal",
                    "layer_to_layer",
                    "ncf",
                    "braid",
                ],
            },
            "code": {"type": int, "required": True, "valid_range": (0, 7)},
            "description": {"type": str, "required": False},
            "parameter_bounds": {"type": dict, "required": True, "has_keys": True},
            "typical": {"type": dict, "required": True},
        },
    },
    "vf_ranges": {
        "type": list,
        "required": True,
        "item_keys": {
            "weave": {"type": str, "required": True},
            "vf_min": {"type": (int, float), "required": True},
            "vf_max": {"type": (int, float), "required": True},
            "comment": {"type": str, "required": False},
        },
    },
    "sweep_presets": {
        "type": list,
        "required": True,
        "item_keys": {
            "name": {"type": str, "required": True},
            "description": {"type": str, "required": True},
            "config": {"type": dict, "required": True, "has_keys": True},
        },
    },
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    pass


def validate_scalar(value, spec, path):
    """Validate a scalar value against its schema spec."""
    # Handle numeric union types (int, float) — accept numeric strings too,
    # since PyYAML may parse e.g. "76.0e9" as str in some environments.
    if spec["type"] in ((int, float), (int,)):
        # Accept int, float, or numeric string
        if isinstance(value, (int, float)):
            numeric_val = float(value)
        elif isinstance(value, str):
            try:
                numeric_val = float(value)
            except ValueError:
                raise ValidationError(f"{path}: expected numeric value, got {value!r}")
        else:
            raise ValidationError(
                f"{path}: expected numeric value, got {type(value).__name__}"
            )

        if "valid" in spec and numeric_val not in spec["valid"]:
            raise ValidationError(
                f"{path}: value {value!r} not in allowed {spec['valid']}"
            )

        if "valid_range" in spec:
            lo, hi = spec["valid_range"]
            if not (lo <= numeric_val <= hi):
                raise ValidationError(
                    f"{path}: value {value} not in range [{lo}, {hi}]"
                )

    elif spec["type"] is str:
        if not isinstance(value, str):
            raise ValidationError(f"{path}: expected str, got {type(value).__name__}")
        if "valid" in spec and value not in spec["valid"]:
            raise ValidationError(
                f"{path}: value {value!r} not in allowed {spec['valid']}"
            )


def validate_list(key, value, spec, path_prefix):
    """Validate a list field and recurse into each item."""
    if not isinstance(value, list):
        raise ValidationError(
            f"{path_prefix}.{key}: expected list, got {type(value).__name__}"
        )

    for i, item in enumerate(value):
        item_prefix = f"{path_prefix}.{key}[{i}]"
        if not isinstance(item, dict):
            raise ValidationError(
                f"{item_prefix}: expected dict, got {type(item).__name__}"
            )

        item_spec = spec["item_keys"]
        # Required keys
        for k, v in item_spec.items():
            if v["required"] and k not in item:
                raise ValidationError(f"{item_prefix}: missing required key {k!r}")
        # Present keys
        for k, val in item.items():
            if k not in item_spec:
                continue
            spec_k = item_spec[k]
            if spec_k.get("has_keys") and isinstance(val, dict) and len(val) == 0:
                raise ValidationError(f"{item_prefix}.{k}: expected non-empty dict")
            else:
                validate_scalar(val, spec_k, f"{item_prefix}.{k}")


def validate_design_space(data):
    """Validate a parsed design_space.yaml against the schema.

    Returns (errors: list[str]). Empty list means valid.
    """
    errors = []

    # Top-level required keys
    for key, spec in SCHEMA.items():
        if spec["required"] and key not in data:
            errors.append(f"missing required top-level key {key!r}")

    # Type checks on top-level values
    for key, value in data.items():
        if key not in SCHEMA:
            continue
        spec = SCHEMA[key]
        if not isinstance(value, spec["type"]):
            errors.append(
                f"{key}: expected {spec['type'].__name__}, got {type(value).__name__}"
            )
            continue

        if spec["type"] is list:
            try:
                validate_list(key, value, spec, "design_space")
            except ValidationError as e:
                errors.append(str(e))

    return errors


# ---------------------------------------------------------------------------
# Sweep size estimation
# ---------------------------------------------------------------------------


def estimate_sweep_size(data):
    """Estimate the number of FEA solves for each sweep preset.

    Returns a dict of {preset_name: {"micromech_solves": N, "b3tex_solves": M, "total": N*M*len(weaves)}}.
    """
    results = {}
    presets = data.get("sweep_presets", [])

    for preset in presets:
        name = preset.get("name", "unnamed")
        config = preset.get("config", {})
        sweep = config.get("sweep", {})

        # Count unique values per sweep key
        counts = {}
        for key, spec in sweep.items():
            if key == "mesh":
                continue
            if isinstance(spec, dict):
                if "hex_vf_sweep" in spec:
                    # hex_vf_sweep expands to an array
                    counts[key] = estimate_hex_vf_count(spec["hex_vf_sweep"])
                elif "linspace" in spec:
                    counts[key] = spec["linspace"][2]
                elif "value" in spec:
                    counts[key] = 1
                elif "values" in spec:
                    counts[key] = len(spec["values"])
                else:
                    counts[key] = 0
            elif isinstance(spec, (int, float)):
                counts[key] = 1

        # Count sweep keys (exclude 'mesh')
        sweep_keys = [k for k in sweep if k != "mesh"]
        if sweep_keys:
            from math import prod

            solves_per_weave = prod(counts.get(k, 0) for k in sweep_keys)
        else:
            solves_per_weave = 0

        num_weaves = len(config.get("weave_architectures", []))
        if num_weaves == 0 and "weave_architectures" not in config:
            # Default to all 5 architectures
            num_weaves = 5

        micromech_solves = solves_per_weave * num_weaves
        b3tex_solves = micromech_solves  # 1:1 mapping for now

        results[name] = {
            "micromech_solves": micromech_solves,
            "b3tex_solves": b3tex_solves,
            "sweep_grid_size": solves_per_weave,
            "num_weave_architectures": num_weaves,
            "per_key_counts": counts,
        }

    return results


def estimate_hex_vf_count(spec):
    """Estimate the number of Vf points from a hex_vf_sweep spec."""
    if isinstance(spec, str):
        presets = {"full": 50, "compact": 30, "high": 20}
        return presets.get(spec, 50)
    elif isinstance(spec, dict):
        preset = spec.get("preset", "full")
        return {"full": 50, "compact": 30, "high": 20}.get(preset, 50)
    return 50


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_design_space.py <path>")
        print("       validate_design_space.py <path> --dry-run")
        print("       validate_design_space.py <path> --schema")
        print("       validate_design_space.py <path> --sweep-size")
        print("       validate_design_space.py <path> --all")
        sys.exit(0)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    mode = sys.argv[-1].lstrip("-") if len(sys.argv) > 2 else "validate"

    import os

    os.environ["PYTHONUNBUFFERED"] = "1"

    if mode in ("validate", "all"):
        errors = validate_design_space(data)
        if errors:
            print("VALIDATION FAILED")
            for err in errors:
                print(f"  ✗ {err}")
            sys.exit(1)
        else:
            print("VALID: design_space.yaml passes schema validation")

    if mode in ("schema", "all"):
        print("\n--- Schema Summary ---")
        print(json.dumps(SCHEMA, indent=2, default=str))

    if mode in ("dry-run", "sweep-size", "all"):
        sizes = estimate_sweep_size(data)
        print("\n--- Sweep Size Estimates ---")
        for name, info in sizes.items():
            print(f"\n  Preset: {name}")
            print(f"    Sweep grid points: {info['sweep_grid_size']}")
            print(f"    Weave architectures: {info['num_weave_architectures']}")
            print(f"    Key counts: {json.dumps(info['per_key_counts'], indent=4)}")

            print(f"\n    FEA solves (micromech): {info['micromech_solves']}")
            print(f"    FEA solves (b3_tex):    {info['b3tex_solves']}")
            if info["micromech_solves"] > 1000:
                print(f"    ⚠ WARNING: {info['micromech_solves']} solves is expensive!")
            elif info["micromech_solves"] > 500:
                print(f"    ⚠ CAUTION: {info['micromech_solves']} solves may take time")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
