from valora.ledger import Ledger


def test_chain_verifies_and_detects_tampering():
    led = Ledger()
    for i in range(5):
        led.record("tick", {"tick": i, "gdp": 1000 + i})
        led.mine()
    ok, errors = led.verify()
    assert ok and not errors
    led.blocks[2]["transactions"][0]["payload"]["gdp"] = 1
    ok, errors = led.verify()
    assert not ok
    assert any("Block 3" in e for e in errors)


def test_transfer_validation():
    led = Ledger()
    try:
        led.add_transaction({"from": "a", "to": "b", "amount": -1})
    except ValueError:
        pass
    else:
        raise AssertionError("negative amount accepted")
    tx = led.add_transaction({"from": "a", "to": "b", "amount": 5})
    assert tx["to"] == "b" and len(led.pending) == 1
