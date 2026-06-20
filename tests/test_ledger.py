import json

from qsfl.ledger.base import OwnershipRecord
from qsfl.ledger.hashchain import HashChainLedger


def test_register_and_verify(tmp_path):
    led = HashChainLedger(tmp_path / "chain.jsonl")
    proof = led.register(OwnershipRecord(model_hash="abc123", owner="alice"))
    assert proof["index"] == 0
    res = led.verify("abc123")
    assert res["found"] and res["integrity_ok"]
    assert led.verify("does-not-exist")["found"] is False


def test_chain_links_and_persists(tmp_path):
    p = tmp_path / "chain.jsonl"
    led = HashChainLedger(p)
    for i in range(3):
        led.register(OwnershipRecord(model_hash=f"h{i}", owner="o"))
    assert led.verify_integrity()
    # reload from disk -> still valid and finds records
    reloaded = HashChainLedger(p)
    assert reloaded.verify_integrity()
    assert reloaded.verify("h2")["found"]


def test_tampering_breaks_chain(tmp_path):
    p = tmp_path / "chain.jsonl"
    led = HashChainLedger(p)
    for i in range(3):
        led.register(OwnershipRecord(model_hash=f"h{i}", owner="o"))

    lines = p.read_text().splitlines()
    block = json.loads(lines[1])
    block["record"]["owner"] = "mallory"  # edit a registered record
    lines[1] = json.dumps(block, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")

    assert HashChainLedger(p).verify_integrity() is False
