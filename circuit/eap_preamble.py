"""CircuitKIT EAP-IG on the preamble contrast (clean = P-default, corrupt = P-none). Run in the circuit env:

    python -m venv .venv-circuit && .venv-circuit/bin/pip install -r requirements-circuit.txt
    .venv-circuit/bin/python circuit/eap_preamble.py --pairs results/e3/circuit/pairs.csv --out results/e3/circuit

API names follow CircuitKIT 0.2.1 (Pipeline.from_custom_data / discover / top_nodes / ck.faithfulness);
verify against the installed version before the run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import circuitkit as ck  # must be imported before transformer_lens
from circuitkit import Pipeline

FULL_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--ig-steps", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline.from_custom_data(args.model, args.pairs, clean_prompt="clean_prompt", clean_answer="clean_answer",
                                     corrupt_prompt="corrupt_prompt", corrupt_answer="corrupt_answer", templated=True)
    circuit = pipe.discover(algorithm="eap-ig", level="node", scope="heads", ig_steps=args.ig_steps)
    top = circuit.top_nodes(args.top_k)  # {"A{layer}.{head}": score, ...}
    mass = {"full": 0.0, "sliding": 0.0}
    for name, score in top.items():
        if name.startswith("A"):
            layer = int(name[1:].split(".")[0])
            mass["full" if layer in FULL_LAYERS else "sliding"] += abs(float(score))
    total = sum(mass.values()) or 1.0
    faith = ck.faithfulness(circuit, pillars=["patching", "ablation"])
    (out / "top_nodes.json").write_text(json.dumps(top, indent=2, default=float))
    (out / "summary.json").write_text(json.dumps({
        "full_layer_mass_fraction": mass["full"] / total, "faithfulness": faith,
    }, indent=2, default=str))
    print(out / "summary.json")


if __name__ == "__main__":
    main()
