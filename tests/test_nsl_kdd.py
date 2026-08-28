from pathlib import Path

import pytest

from src.data.nsl_kdd import attack_family, load_nsl_kdd


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("normal", "Normal"),
        ("smurf.", "DoS"),
        ("nmap", "Probe"),
        ("guess_passwd", "R2L"),
        ("buffer_overflow", "U2R"),
    ],
)
def test_attack_family_maps_base_study_classes(label: str, expected: str) -> None:
    assert attack_family(label) == expected


def test_attack_family_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="Unmapped"):
        attack_family("not-a-real-class")


def test_loader_requires_both_official_file_names(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"KDDTrain\+\.txt"):
        load_nsl_kdd(tmp_path)
