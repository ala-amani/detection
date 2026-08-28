from examples.generate_synthetic_mikrotik_lab import build_packets


def test_synthetic_lab_contains_documented_scenarios() -> None:
    packets, rows = build_packets()
    assert len(packets) == 231
    assert [row["scenario"] for row in rows] == [
        "normal_web", "router_port_probe", "wifi_ap_syn_burst", "authorized_router_health_check"
    ]
    assert [row["synthetic_label"] for row in rows] == ["Normal", "Probe", "DoS", "Normal"]
    assert rows[1]["diff_srv_rate"] > rows[2]["diff_srv_rate"]
    assert rows[2]["srv_count"] > rows[1]["srv_count"]
