"""The refiner-side fields of crates/phonon-core/src/dictionary_cases.json, the reference cases the
phones' dictionary ports are checked against: the vocabulary line terms (refiner_vocabulary) of each
envelope, and restore_terms. PHONON_WRITE_DICTIONARY_CASES=1 rewrites them instead of comparing."""

import json
import os
from pathlib import Path

from sidecar.polish_server import refiner_vocabulary, restore_terms

CASES = Path(__file__).resolve().parents[1] / "crates/phonon-core/src/dictionary_cases.json"


def test_dictionary_cases_match_the_refiner_side():
    cases = json.loads(CASES.read_text())
    write = os.environ.get("PHONON_WRITE_DICTIONARY_CASES") == "1"
    checked = 0
    for case in cases["envelope"]:
        vocab = refiner_vocabulary(case["envelope"])
        if write:
            case["vocab"] = vocab
        else:
            assert case["vocab"] == vocab, case["id"]
        checked += 1
    for case in cases["restore"]:
        out = restore_terms(case["spoken"], case["output"], case["vocab"])
        if write:
            case["expected"] = out
        else:
            assert case["expected"] == out, case["id"]
        checked += 1
    assert checked >= 25
    if write:
        CASES.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
