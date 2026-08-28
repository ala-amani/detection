"""Generate a safe synthetic MikroTik/phone/laptop/Wi-Fi PCAP and flow rows.

The packets are constructed offline and are never transmitted. The companion
CSV records the NSL-KDD-compatible flow statistics used by the IDS adapter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scapy.all import DNS, DNSQR, Ether, IP, TCP, UDP, Raw, wrpcap


NUMERIC_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]


def base_row(scenario: str, protocol: str, service: str, flag: str, label: str) -> dict:
    row = {name: 0.0 for name in NUMERIC_FEATURES}
    row.update({"scenario": scenario, "protocol_type": protocol, "service": service, "flag": flag, "synthetic_label": label})
    return row


def build_packets() -> tuple[list, list[dict]]:
    router, access_point = "192.168.88.1", "192.168.88.2"
    laptop, phone = "192.168.88.10", "192.168.88.20"
    dns_server, web_server = "1.1.1.1", "93.184.216.34"
    mac = {
        router: "02:00:00:00:88:01", access_point: "02:00:00:00:88:02",
        laptop: "02:00:00:00:88:10", phone: "02:00:00:00:88:20",
        dns_server: "02:00:00:00:00:53", web_server: "02:00:00:00:00:80",
    }

    def link(src: str, dst: str):
        return Ether(src=mac[src], dst=mac[dst])
    packets = []
    timestamp = 1_800_000_000.0

    def add(packet, offset: float) -> None:
        packet.time = timestamp + offset
        packets.append(packet)

    # Normal phone DNS exchange and laptop web session.
    add(link(phone, dns_server)/IP(src=phone, dst=dns_server)/UDP(sport=53000, dport=53)/DNS(rd=1, qd=DNSQR(qname="example.org")), 0.00)
    add(link(dns_server, phone)/IP(src=dns_server, dst=phone)/UDP(sport=53, dport=53000)/DNS(id=1, qr=1, qd=DNSQR(qname="example.org")), 0.03)
    add(link(laptop, web_server)/IP(src=laptop, dst=web_server)/TCP(sport=51000, dport=80, flags="S", seq=1), 0.10)
    add(link(web_server, laptop)/IP(src=web_server, dst=laptop)/TCP(sport=80, dport=51000, flags="SA", seq=10, ack=2), 0.13)
    add(link(laptop, web_server)/IP(src=laptop, dst=web_server)/TCP(sport=51000, dport=80, flags="A", seq=2, ack=11), 0.15)
    add(link(laptop, web_server)/IP(src=laptop, dst=web_server)/TCP(sport=51000, dport=80, flags="PA")/Raw(b"GET / HTTP/1.1\r\nHost: example.org\r\n\r\n"), 0.18)
    add(link(web_server, laptop)/IP(src=web_server, dst=laptop)/TCP(sport=80, dport=51000, flags="PA")/Raw(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello"), 0.24)

    # Offline synthetic Probe: laptop sends SYN packets to many router ports.
    for index, port in enumerate([21, 22, 23, 53, 80, 443, 8291, 8728, 8729, 8080] * 4):
        add(link(laptop, router)/IP(src=laptop, dst=router)/TCP(sport=40000 + index, dport=port, flags="S", seq=index), 1.0 + index * 0.01)

    # Offline synthetic DoS-like burst: repeated SYN packets to the AP web service.
    for index in range(160):
        add(link(phone, access_point)/IP(src=phone, dst=access_point)/TCP(sport=45000 + index % 20, dport=80, flags="S", seq=index), 2.0 + index * 0.002)

    # Authorized maintenance check: scan-like, but operationally benign.
    management_ports = [22, 80, 443, 8291, 8728, 8729]
    for index, port in enumerate(management_ports * 4):
        add(link(laptop, router)/IP(src=laptop, dst=router)/TCP(sport=47000 + index, dport=port, flags="S", seq=index), 3.0 + index * 0.015)

    normal = base_row("normal_web", "tcp", "http", "SF", "Normal")
    normal.update(duration=0.14, src_bytes=41, dst_bytes=43, logged_in=1, count=3, srv_count=3,
                  same_srv_rate=1.0, dst_host_count=12, dst_host_srv_count=10,
                  dst_host_same_srv_rate=0.83, dst_host_same_src_port_rate=0.08)
    probe = base_row("router_port_probe", "tcp", "private", "S0", "Probe")
    probe.update(duration=0.39, src_bytes=0, dst_bytes=0, count=40, srv_count=4,
                 serror_rate=1.0, srv_serror_rate=1.0, same_srv_rate=0.10, diff_srv_rate=0.90,
                 dst_host_count=40, dst_host_srv_count=4, dst_host_same_srv_rate=0.10,
                 dst_host_diff_srv_rate=0.90, dst_host_same_src_port_rate=0.03,
                 dst_host_serror_rate=1.0, dst_host_srv_serror_rate=1.0)
    dos = base_row("wifi_ap_syn_burst", "tcp", "http", "S0", "DoS")
    dos.update(duration=0.318, src_bytes=0, dst_bytes=0, count=160, srv_count=160,
               serror_rate=1.0, srv_serror_rate=1.0, same_srv_rate=1.0, diff_srv_rate=0.0,
               dst_host_count=160, dst_host_srv_count=160, dst_host_same_srv_rate=1.0,
               dst_host_diff_srv_rate=0.0, dst_host_same_src_port_rate=0.05,
               dst_host_serror_rate=1.0, dst_host_srv_serror_rate=1.0)
    maintenance = base_row("authorized_router_health_check", "tcp", "private", "S0", "Normal")
    maintenance.update(duration=0.345, src_bytes=0, dst_bytes=0, count=24, srv_count=4,
                       serror_rate=0.75, srv_serror_rate=0.75, same_srv_rate=0.17,
                       diff_srv_rate=0.83, dst_host_count=24, dst_host_srv_count=4,
                       dst_host_same_srv_rate=0.17, dst_host_diff_srv_rate=0.83,
                       dst_host_same_src_port_rate=0.04, dst_host_serror_rate=0.75,
                       dst_host_srv_serror_rate=0.75)
    return packets, [normal, probe, dos, maintenance]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap", required=True, type=Path)
    parser.add_argument("--flows", required=True, type=Path)
    args = parser.parse_args()
    packets, rows = build_packets()
    args.pcap.parent.mkdir(parents=True, exist_ok=True)
    args.flows.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(args.pcap), packets)
    pd.DataFrame(rows).to_csv(args.flows, index=False)
    print(f"Wrote {len(packets)} offline packets to {args.pcap}")
    print(f"Wrote {len(rows)} documented flow rows to {args.flows}")


if __name__ == "__main__":
    main()
