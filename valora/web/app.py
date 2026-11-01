"""Flask app: the original VALORA studio UI plus the new agent, copilot and benchmark APIs."""

from __future__ import annotations

import io
import os
import secrets
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for

from .. import __version__
from ..engine import LEVER_BOUNDS, SHOCK_PROFILES
from ..engine.actors import simulate_economy
from ..harness.bench import AGENTS, load_scenarios
from ..runtime import Runtime, get_runtime


def _json() -> dict[str, Any]:
    return request.get_json(silent=True) or {}


def _int_arg(name: str, default: int | None = None) -> int | None:
    try:
        v = request.args.get(name)
        return int(v) if v is not None else default
    except ValueError:
        return default


def create_app(runtime: Runtime | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    rt = runtime or get_runtime()
    app.config["RUNTIME"] = rt
    app.secret_key = rt.s.secret_key or os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
    app.config["JSON_SORT_KEYS"] = False

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def landing_page():
        return render_template("landing.html")

    @app.get("/studio")
    def studio():
        session.setdefault("session_id", secrets.token_hex(8))
        return render_template("index.html", max_tokens=rt.s.llm_max_tokens, model=rt.llm.model)

    @app.get("/simulation")
    def simulation_page():
        return render_template("simulation.html")

    @app.get("/copilot")
    def copilot_page():
        return render_template("copilot.html", model=rt.llm.model)

    @app.get("/bench")
    def bench_page():
        return render_template("bench.html", model=rt.llm.model)

    @app.get("/healthz")
    @app.get("/api/health")
    def health():
        ok, _ = rt.ledger.verify()
        return jsonify({"status": "ok", "version": __version__, "tick": rt.economy.tick, "ledger_valid": ok})

    @app.get("/api/meta")
    def meta():
        return jsonify({**rt.meta(), "version": __version__, "levers": LEVER_BOUNDS,
                        "shocks": list(SHOCK_PROFILES), "bench_agents": list(AGENTS),
                        "scenarios": {k: v.description for k, v in load_scenarios().items()}})

    # ---------------------------------------------------------------- economy
    @app.get("/api/economic/status")
    def api_economic_status():
        with rt.lock:
            return jsonify(rt.economy.status())

    @app.get("/api/economic/history")
    def api_economic_history():
        limit = _int_arg("limit", 120) or 120
        with rt.lock:
            return jsonify({"history": rt.economy.history[-limit:]})

    @app.post("/api/economic/step")
    def api_economic_step():
        n = int(_json().get("ticks", 1))
        snaps = rt.step(n)
        return jsonify({"ticks": len(snaps), "status": rt.economy.status(), "last": snaps[-1]})

    @app.post("/api/economic/shock")
    def api_economic_shock():
        data = _json()
        shock = data.get("type", "pandemic")
        if shock not in SHOCK_PROFILES:
            return jsonify({"error": f"unknown shock; use one of {list(SHOCK_PROFILES)}"}), 400
        months = int(data.get("duration", data.get("duration_ticks", max(1, int(data.get("duration_days", 180)) // 30))))
        status = rt.apply_shock(shock, float(data.get("magnitude", 1.0)), months)
        return jsonify({"status": "ok", "active_shocks": status["active_shocks"]})

    @app.get("/api/agents/status")
    def api_agents_status():
        with rt.lock:
            return jsonify(rt.economy.agents_status())

    @app.post("/api/whatif")
    def api_whatif():
        from ..agentic.policy import what_if

        data = _json()
        levers = data.get("levers") or {}
        ctx = rt.snapshot_ctx()
        return jsonify(what_if(ctx.economy, levers, int(data.get("duration", 12)), int(data.get("horizon", 24))))

    # ---------------------------------------------------------------- council
    @app.post("/api/agents/run")
    def api_agents_run():
        data = _json()
        text = (data.get("text") or data.get("policy_text") or "").strip()
        if not text:
            return jsonify({"error": "policy text is required"}), 400
        result = rt.deliberate(text, apply=bool(data.get("apply", False)), author=session.get("session_id", "web"))
        session["last_council_id"] = result["id"]
        return jsonify(result)

    @app.post("/api/agents/<run_id>/apply")
    def api_agents_apply(run_id: str):
        prev = rt.council_runs.get(run_id)
        if not prev:
            return jsonify({"error": "run not found"}), 404
        result = rt.deliberate(prev["policy"], apply=True, author=session.get("session_id", "web"),
                               levers=prev["final_levers"], duration=prev["final_duration"])
        return jsonify(result)

    @app.get("/api/agents/report")
    def api_agents_report():
        run = rt.council_runs.get(session.get("last_council_id", ""))
        if not run:
            return jsonify({"error": "no agent run found for this session"}), 404
        buf = io.StringIO()
        buf.write(f"VALORA council report\nPolicy: {run['policy']}\nMode: {run['mode']} ({run['model']})\n\n")
        for i, a in enumerate(run["agents"], 1):
            buf.write(f"{i}. {a['agent']} (vote: {a['vote']})\n{a['output'].strip()}\n\n")
        buf.write("Debate:\n")
        for d in run["debate"]:
            buf.write(f"- {d['speaker']}: {d['message']}\n")
        buf.write("\n" + run["final_output"] + "\n")
        payload = io.BytesIO(buf.getvalue().encode("utf-8"))
        return send_file(payload, as_attachment=True, download_name="policy_report.txt", mimetype="text/plain")

    @app.get("/api/crew/runs")
    def api_crew_runs():
        limit = _int_arg("limit", 20) or 20
        runs = list(rt.council_runs.values())[-limit:]
        return jsonify({"runs": [{"id": r["id"], "policy": r["policy"], "status": r["status"],
                                  "verdict": r["verdict"]["decision"], "created_at": r["created_at"]} for r in runs]})

    @app.get("/api/policies")
    def api_policies_list():
        with rt.lock:
            active = rt.economy.status()["active_policies"]
        return jsonify({"policies": rt.store.decisions(limit=_int_arg("limit", 50) or 50), "active_effects": active})

    @app.post("/api/policies")
    def api_submit_policy():
        data = _json()
        text = (data.get("text") or data.get("policy_text") or "").strip()
        if not text:
            return jsonify({"error": "Policy text is required"}), 400
        result = rt.deliberate(text, apply=True, author=str(data.get("author", "api")),
                               levers=data.get("levers"), duration=data.get("duration"))
        return jsonify({"status": result["status"], "policy": result, "indicators": rt.economy.indicators()})

    @app.get("/api/policies/active")
    def api_policies_active():
        with rt.lock:
            return jsonify({"active_effects": rt.economy.status()["active_policies"]})

    @app.get("/api/policies/<policy_id>")
    def api_policy_detail(policy_id: str):
        rec = rt.store.get_decision(policy_id)
        return (jsonify(rec), 200) if rec else (jsonify({"error": "policy not found"}), 404)

    @app.post("/api/policies/<policy_id>/revoke")
    def api_policy_revoke(policy_id: str):
        ok = rt.revoke(policy_id)
        return jsonify({"status": "revoked" if ok else "no_active_effects", "policy_id": policy_id})

    @app.post("/submit_policy")
    def submit_policy():
        text = (request.form.get("policy_text") or "").strip()
        if not text:
            return jsonify({"error": "no policy text"}), 400
        result = rt.deliberate(text, apply=True, author=session.get("session_id", "web"))
        return jsonify({"status": result["status"], "policy": result, "indicators": rt.economy.indicators()})

    @app.post("/reset_session")
    def reset_session():
        session.clear()
        return redirect(url_for("studio"))

    # ---------------------------------------------------------------- copilot
    @app.post("/api/copilot/run")
    def api_copilot_run():
        data = _json()
        goal = (data.get("goal") or "").strip()
        if len(goal) > 600:
            return jsonify({"error": "goal is too long (600 characters max)"}), 400
        return jsonify(rt.copilot(goal, horizon=int(min(60, max(6, int(data.get("horizon", 24)))))))

    @app.get("/api/copilot/runs")
    def api_copilot_runs():
        return jsonify({"runs": rt.store.runs("copilot", limit=_int_arg("limit", 20) or 20)})

    @app.post("/api/copilot/<run_id>/submit")
    def api_copilot_submit(run_id: str):
        run = rt.copilot_runs.get(run_id)
        if not run or not run.get("proposal"):
            return jsonify({"error": "copilot run not found"}), 404
        p = run["proposal"]
        result = rt.deliberate(f"{p['label']}: {p.get('rationale', '')}".strip(), apply=True, author="copilot",
                               levers=p["levers"], duration=p.get("duration", 12))
        return jsonify(result)

    # -------------------------------------------------------------- benchmark
    @app.post("/api/bench/run")
    def api_bench_run():
        data = _json()
        agents = [a for a in (data.get("agents") or []) if a in AGENTS] or None
        if agents and "copilot-llm" in agents and not rt.llm.enabled:
            agents.remove("copilot-llm")
        job = rt.start_bench(agents, data.get("scenarios") or None)
        return jsonify({k: v for k, v in job.items() if k != "result"})

    @app.get("/api/bench/status")
    def api_bench_status():
        return jsonify(rt.bench_job)

    @app.get("/api/bench/leaderboard")
    def api_bench_leaderboard():
        return jsonify({"leaderboard": rt.store.leaderboard()})

    # ------------------------------------------------------- sector simulation
    @app.post("/api/simulation/run")
    def run_policy_simulation():
        data = _json()
        text = (data.get("policy") or data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Policy text is required."}), 400
        steps = int(min(36, max(1, int(data.get("time_steps") or 8))))
        return jsonify(simulate_economy(text, time_steps=steps))

    # ----------------------------------------------------------------- ledger
    @app.post("/api/tax/calc")
    def api_tax_calc():
        data = _json()
        value = float(data.get("order_value", 0.0))
        return jsonify({"order_value": value, "tax_rate": rt.economy.tax_rate, "tax": value * rt.economy.tax_rate})

    @app.post("/api/blockchain/tx")
    def api_add_tx():
        try:
            tx = rt.ledger.add_transaction(_json())
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        return jsonify({"status": "tx_added", "tx": tx, "pending": len(rt.ledger.pending)})

    @app.post("/api/blockchain/mine")
    def api_mine():
        block = rt.ledger.mine()
        return jsonify({"status": "mined", "block": block} if block else {"status": "no_pending"})

    @app.get("/api/blockchain/ledger")
    def api_ledger():
        return jsonify({"ledger": rt.ledger.get_blocks(_int_arg("limit"))})

    @app.get("/api/blockchain/pending")
    def api_pending():
        return jsonify({"pending": rt.ledger.get_pending()})

    @app.get("/api/blockchain/block/<int:index>")
    def api_block_detail(index: int):
        block = rt.ledger.get_block(index)
        return (jsonify(block), 200) if block else (jsonify({"error": "block_not_found"}), 404)

    @app.get("/api/blockchain/verify")
    def api_verify_chain():
        ok, errors = rt.ledger.verify()
        return jsonify({"valid": ok, "errors": errors, "audit": rt.ledger.get_audit(25), "mock_running": False,
                        **rt.ledger.status()})

    @app.get("/api/blockchain/audit")
    def api_audit_log():
        return jsonify({"audit": rt.ledger.get_audit(_int_arg("limit"))})

    @app.errorhandler(404)
    def not_found(_e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not found"}), 404
        return redirect(url_for("landing_page"))

    @app.errorhandler(500)
    def server_error(_e):
        return jsonify({"error": "internal server error"}), 500

    return app


def main() -> None:  # pragma: no cover - manual entry point
    app = create_app()
    app.run(host="0.0.0.0", port=get_runtime().s.port, debug=False, use_reloader=False)


if __name__ == "__main__":  # pragma: no cover
    main()
