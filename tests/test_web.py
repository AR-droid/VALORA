def test_pages_render(client):
    for path in ["/", "/studio", "/simulation", "/copilot", "/bench"]:
        r = client.get(path)
        assert r.status_code == 200, path


def test_status_and_step(client):
    r = client.get("/api/economic/status").get_json()
    assert {"gdp", "inflation", "unemployment", "consumer_confidence"} <= set(r["indicators"])
    r = client.post("/api/economic/step", json={"ticks": 3}).get_json()
    assert r["status"]["tick"] == 3
    assert client.get("/api/blockchain/verify").get_json()["valid"]


def test_council_run_apply_and_report(client):
    r = client.post("/api/agents/run", json={"text": "Increase transfers by 6% for 12 months"}).get_json()
    assert len(r["agents"]) == 5 and r["status"] in {"approved", "rejected"}
    rep = client.get("/api/agents/report")
    assert rep.status_code == 200 and b"Council decision" in rep.data
    r = client.post("/api/policies", json={"text": "Cut taxes", "levers": {"spending_change": 0.04}}).get_json()
    assert r["policy"]["final_levers"]
    assert client.get(f"/api/policies/{r['policy']['id']}").status_code == 200
    assert client.post("/api/agents/run", json={}).status_code == 400


def test_shock_validation(client):
    assert client.post("/api/economic/shock", json={"type": "aliens"}).status_code == 400
    r = client.post("/api/economic/shock", json={"type": "pandemic", "magnitude": 1, "duration": 6})
    assert r.get_json()["active_shocks"][0]["type"] == "pandemic"


def test_copilot_endpoint(client):
    r = client.post("/api/copilot/run", json={"goal": "Get inflation back to 2%", "horizon": 12}).get_json()
    assert r["proposal"]["levers"]
    assert client.get("/api/copilot/runs").get_json()["runs"]
    c = client.post(f"/api/copilot/{r['id']}/submit").get_json()
    assert c["status"] in {"applied", "rejected"}


def test_sector_simulation_and_meta(client):
    r = client.post("/api/simulation/run", json={"policy": "Ease credit for small firms", "time_steps": 4})
    assert len(r.get_json()["global_series"]) == 4
    m = client.get("/api/meta").get_json()
    assert m["llm_enabled"] is False and "copilot-offline" in m["bench_agents"]
    assert client.get("/api/nope").status_code == 404
