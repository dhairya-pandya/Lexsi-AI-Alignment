# Circuit discovery (E3d)

Separate environment (CircuitKIT pins TransformerLens 3.8.0):

```bash
python -m venv .venv-circuit && .venv-circuit/bin/pip install -r requirements-circuit.txt
CIRCUITKIT_RUN_TINY_AYA=1 .venv-circuit/bin/python -m pytest $(python -c "import circuitkit,os;print(os.path.dirname(circuitkit.__file__))")/../tests/backends/tl_compat/test_cohere_parity.py  # fp32 parity gate
```

1. In the main env, export pairs (fit split, <= 512 tokens, single-token answers taken from `results/gate_g1/openers.json`):
   `hsafe.interp.circuit_pairs.export_circuit_pairs(...)` -> `results/e3/circuit/pairs.csv`.
2. Run `circuit/eap_preamble.py` on an L4.

Parity gate result: _not yet run_.
