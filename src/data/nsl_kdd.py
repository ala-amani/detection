"""NSL-KDD loader matching the five attack families used by the base study."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

DOS = {"neptune", "back", "land", "pod", "smurf", "teardrop", "mailbomb", "apache2", "processtable", "udpstorm", "worm"}
PROBE = {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"}
R2L = {"ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack", "snmpguess", "xlock", "xsnoop", "httptunnel"}
U2R = {"buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"}


def attack_family(label: str) -> str:
    value = label.strip().rstrip(".").casefold()
    if value == "normal":
        return "Normal"
    if value in DOS:
        return "DoS"
    if value in PROBE:
        return "Probe"
    if value in R2L:
        return "R2L"
    if value in U2R:
        return "U2R"
    raise ValueError(f"Unmapped NSL-KDD attack label: {label!r}")


def load_nsl_kdd(root: Path, max_rows_per_file: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    frames = []
    for filename in ("KDDTrain+.txt", "KDDTest+.txt"):
        source = root / filename
        if not source.exists():
            raise FileNotFoundError(f"Required NSL-KDD file is missing: {source}")
        frames.append(pd.read_csv(source, names=COLUMNS, nrows=max_rows_per_file))
    frame = pd.concat(frames, ignore_index=True).drop_duplicates()
    labels = frame.pop("label").map(attack_family)
    frame = frame.drop(columns=["difficulty"])
    return frame, labels

