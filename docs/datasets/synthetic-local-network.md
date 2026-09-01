# Synthetic local-network fixture

## Purpose

This deterministic fixture verifies the end-to-end IDS, SHAP, LLM-interpretation, human-review, validated-feedback, and IDS-retraining path without claiming that synthetic traffic is a substitute for an operational capture.

## Topology

| Device | Address | Role |
|---|---:|---|
| MikroTik router | 192.168.88.1 | Gateway and management services |
| Wi-Fi access point | 192.168.88.2 | Wireless service target |
| Laptop | 192.168.88.10 | Administrator/client endpoint |
| Phone | 192.168.88.20 | Mobile client endpoint |

## Scenarios and labels

1. `normal_web`: ordinary phone web/DNS activity; label `Normal`.
2. `wifi_ap_syn_burst`: repeated incomplete TCP SYN attempts to one AP service; label `DoS`.
3. `router_port_probe`: laptop contacts multiple router service ports; label `Probe`.
4. `authorized_router_health_check`: scheduled administrator maintenance that resembles scanning; label `Normal` and intentionally tests false-positive handling.

The generator writes a 231-packet PCAP and a companion CSV containing documented NSL-KDD-compatible scenario rows. The CSV is derived from the deterministic scenario definitions; it is not a general PCAP-to-NSL-KDD flow extractor.

## Storage and provenance

The PCAP, processed CSV, feedback logs, and generated results are stored outside the repository and excluded from Git. Only the generator, schema documentation, evaluation code, and tests are versioned. Fixed timestamps, addresses, MAC addresses, and random seeds make regeneration auditable.

## Reproduction

Run `examples/generate_synthetic_mikrotik_lab.py` with explicit `--pcap` and `--flows` output paths, then run `python -m src.pipeline.synthetic_lab_evaluation --flows <CSV> --scenario all`. The evaluator uses the published NSL-KDD Random Forest preprocessing and model block, SHAP evidence, the independent validator, the local model, and the analyst interface.
