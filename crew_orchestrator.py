#!/usr/bin/env python3
"""
Crew Orchestrator for Valora — CrewAI + Groq LLM integrated example.

Notes:
- Set environment variable GROQ_API_KEY before running.
- If your crewai.LLM expects a different kwarg name for the provider key
  (e.g. groq_api_key), switch in the 'init_llm' function below.
- Disable Flask auto reloader (use_reloader=False) to avoid debugpy threading issues.
"""

import os
import uuid
import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict

from flask import Flask, request, jsonify, render_template, session, redirect, url_for


from crewai import Agent, Task, Crew, Process, LLM

try:
    from economic_council import EconomicCouncil
except Exception:
    class EconomicCouncil:
        """Simple fallback stub for development if economic_council isn't available."""
        def __init__(self, llm):
            self.llm = llm

        def analyze_policy(self, text):
            return {"policy_type": "fiscal_policy", "parameters": {"spending_change": 0.02}, "summary": "stub analysis"}

        def debate_policy(self, analysis):
            return [{"agent": "stub", "opinion": "implement"}]

        def reach_consensus(self, debate):
            return {"consensus_reached": True, "notes": "stub consensus"}


MODEL_ID = os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # must be set in env
PORT = int(os.getenv("PORT", 5004))


app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-123")
os.makedirs("templates", exist_ok=True)


def init_llm():
    """
    Best-effort LLM initializer. Different versions of crewai/litellm expect different kwarg names:
      - api_key
      - groq_api_key
    We'll try a couple of common options and fall back to a minimal local stub LLM.
    """
    if not GROQ_API_KEY:
        print("[WARN] GROQ_API_KEY is not set. Agents will use fallback stub responses.")
        return None 
    try:
        llm = LLM(model=MODEL_ID, api_key=GROQ_API_KEY)
        print("[LLM] Initialized with LLM(model=..., api_key=...)")
        return llm
    except Exception as e1:
        print("[LLM] api_key arg failed:", e1)

   
        llm = LLM(model=MODEL_ID, groq_api_key=GROQ_API_KEY)
        print("[LLM] Initialized with LLM(model=..., groq_api_key=...)")
        return llm
    except Exception as e2:
        print("[LLM] groq_api_key arg failed:", e2)

  
    try:
        llm = LLM(model=MODEL_ID)
        print("[LLM] Initialized with LLM(model=...) (no key arg — relying on library env behavior)")
        return llm
    except Exception as e3:
        print("[LLM] model-only init failed:", e3)


    return None


class FallbackLLM:
    def __init__(self):
        pass

    def chat(self, prompt: str, **kwargs):
        # Very simple canned reply — allows app to keep working even without LLM access
        return {"content": "LLM unavailable — this is a fallback response for development."}


llm_instance = init_llm()
if llm_instance is None:
    fallback_llm = FallbackLLM()
    economic_council = EconomicCouncil(fallback_llm)
else:
    economic_council = EconomicCouncil(llm_instance)


simulation_state: Dict[str, dict] = {}


class EconomicCycleManager:
    def __init__(self):
        self.current_cycle = "expansion"
        self.cycle_start_date = datetime.now()
        self.cycle_duration = 365
        self.economic_indicators = {
            "gdp": 1000.0,
            "inflation": 2.0,
            "unemployment": 5.0,
            "consumer_confidence": 70.0
        }
        self.shock_active = False
        self.shock_details = {}

    def update_cycle(self):
        days = (datetime.now() - self.cycle_start_date).days
        if days > self.cycle_duration:
            order = ["expansion", "peak", "recession", "trough"]
            idx = order.index(self.current_cycle)
            self.current_cycle = order[(idx + 1) % len(order)]
            self.cycle_start_date = datetime.now()

    def apply_economic_shock(self, shock_type: str, magnitude: float, duration_days: int):
        self.shock_active = True
        self.shock_details = {
            "type": shock_type,
            "magnitude": magnitude,
            "start_time": datetime.now().isoformat(),
            "end_time": (datetime.now() + timedelta(days=duration_days)).isoformat()
        }
        # immediate simple effects
        if shock_type == "financial_crisis":
            self.economic_indicators["gdp"] *= (1 - 0.03 * magnitude)
            self.economic_indicators["unemployment"] += 1.0 * magnitude
        elif shock_type == "pandemic":
            self.economic_indicators["consumer_confidence"] -= 10.0 * magnitude
            self.economic_indicators["gdp"] *= (1 - 0.02 * magnitude)

    def get_status(self):
        self.update_cycle()
        return {
            "current_cycle": self.current_cycle,
            "days_in_cycle": (datetime.now() - self.cycle_start_date).days,
            "indicators": self.economic_indicators,
            "shock_active": self.shock_active,
            "shock_details": self.shock_details if self.shock_active else None
        }

class OrderTaxAdjustment:
    def __init__(self):
        self.base_tax_rate = 0.10
        self.adjustments = {}

    def calculate_tax(self, order_value: float, order_type: str = None):
        rate = self.base_tax_rate
        if order_type and order_type in self.adjustments:
            rate += self.adjustments[order_type]
        return order_value * rate

class BlockchainIntegration:
    def __init__(self):
        self.ledger = []
        self.pending_transactions = []

    def add_transaction(self, tx: dict):
        tx2 = dict(tx)
        tx2["timestamp"] = datetime.now().isoformat()
        tx2["status"] = "pending"
        self.pending_transactions.append(tx2)

    def mine_block(self):
        if not self.pending_transactions:
            return None
        block = {
            "index": len(self.ledger) + 1,
            "timestamp": datetime.now().isoformat(),
            "transactions": self.pending_transactions.copy(),
            "previous_hash": self.ledger[-1]["hash"] if self.ledger else "0",
            "hash": f"block_{len(self.ledger)+1}"
        }
        self.ledger.append(block)
        self.pending_transactions = []
        return block

economic_manager = EconomicCycleManager()
tax_manager = OrderTaxAdjustment()
blockchain = BlockchainIntegration()


agent_llm = llm_instance if llm_instance is not None else FallbackLLM()

economic_analyst = Agent(
    role="Economic Analyst",
    goal="Analyze economic indicators and suggest policy actions",
    backstory="Economist with macro and policy expertise",
    llm=agent_llm,
    verbose=True
)

tax_advisor = Agent(
    role="Tax Advisor",
    goal="Recommend tax or fiscal adjustments given economic state",
    backstory="Tax expert focusing on efficient fiscal policy",
    llm=agent_llm,
    verbose=True
)

blockchain_expert = Agent(
    role="Blockchain Specialist",
    goal="Validate and log transactions to ledger; audit blocks",
    backstory="Blockchain engineer and audit specialist",
    llm=agent_llm,
    verbose=True
)

def task_economic_analysis():
    return Task(
        description="Analyze the current macro indicators (gdp, inflation, unemployment) and produce a short report with 3 bullet insights and 1 recommended policy action.",
        expected_output="short_report",
        agent=economic_analyst
    )

def task_tax_review():
    return Task(
        description="Given the current economic indicators, recommend any tax adjustments or fiscal policy with pros/cons.",
        expected_output="tax_recommendations",
        agent=tax_advisor,
        dependencies=[task_economic_analysis()]
    )

def task_blockchain_audit():
    return Task(
        description="Audit pending transactions and confirm integrity of ledger; list pending tx count.",
        expected_output="audit_report",
        agent=blockchain_expert
    )

crew = Crew(
    agents=[economic_analyst, tax_advisor, blockchain_expert],
    tasks=[task_economic_analysis(), task_tax_review(), task_blockchain_audit()],
    verbose=True,
    process=Process.sequential,
    manager_llm=agent_llm
)


@app.route("/api/economic/status", methods=["GET"])
def api_economic_status():
    return jsonify(economic_manager.get_status())

@app.route("/api/economic/shock", methods=["POST"])
def api_economic_shock():
    data = request.get_json(force=True)
    economic_manager.apply_economic_shock(
        data.get("type", "pandemic"),
        float(data.get("magnitude", 1.0)),
        int(data.get("duration_days", 90))
    )
    return jsonify({"status": "ok", "shock": economic_manager.shock_details})

@app.route("/api/tax/calc", methods=["POST"])
def api_tax_calc():
    data = request.get_json(force=True)
    order_value = float(data.get("order_value", 0.0))
    order_type = data.get("order_type")
    tax = tax_manager.calculate_tax(order_value, order_type)
    return jsonify({"order_value": order_value, "tax": tax})

@app.route("/api/blockchain/tx", methods=["POST"])
def api_add_tx():
    data = request.get_json(force=True)
    blockchain.add_transaction(data)
    return jsonify({"status": "tx_added", "pending": len(blockchain.pending_transactions)})

@app.route("/api/blockchain/mine", methods=["POST"])
def api_mine():
    block = blockchain.mine_block()
    if block:
        return jsonify({"status": "mined", "block": block})
    return jsonify({"status": "no_pending"})

@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        simulation_state[session["session_id"]] = {
            "economic_indicators": economic_manager.economic_indicators.copy(),
            "policies": [],
            "debates": []
        }

    sid = session["session_id"]
    return render_template("index.html",
                           indicators=simulation_state[sid]["economic_indicators"],
                           policies=simulation_state[sid]["policies"],
                           debates=simulation_state[sid]["debates"]
                           )

@app.route("/submit_policy", methods=["POST"])
def submit_policy():
    if "session_id" not in session:
        return redirect(url_for("index"))
    sid = session["session_id"]
    policy_text = request.form.get("policy_text", "")
    if not policy_text:
        return jsonify({"error": "no policy text"}), 400

    # 1) Analyze with economic_council (may be fallback)
    try:
        analysis = economic_council.analyze_policy(policy_text)
    except Exception as e:
        analysis = {"error": str(e)}

    if "error" in analysis:
        return jsonify({"error": "policy analysis failed", "details": analysis}), 500

    # 2) Debate & consensus
    try:
        debate = economic_council.debate_policy(analysis)
        consensus = economic_council.reach_consensus(debate)
    except Exception as e:
        debate = []
        consensus = {"consensus_reached": False, "error": str(e)}

    implemented = False
    if consensus.get("consensus_reached"):
        # Apply a very simple policy effect
        params = analysis.get("parameters", {})
        if analysis.get("policy_type") == "fiscal_policy":
            spending_change = params.get("spending_change", 0.0)
            economic_manager.economic_indicators["gdp"] *= (1 + 0.8 * spending_change)
            economic_manager.economic_indicators["inflation"] += 0.3 * spending_change
            implemented = True

    entry = {
        "id": str(uuid.uuid4()),
        "text": policy_text,
        "analysis": analysis,
        "debate": debate,
        "consensus": consensus,
        "timestamp": datetime.now().isoformat(),
        "implemented": implemented
    }
    simulation_state[sid]["policies"].append(entry)
    simulation_state[sid]["debates"].append(entry)
    return jsonify({"status": "ok", "policy": entry, "indicators": economic_manager.economic_indicators})

@app.route("/reset_session", methods=["POST"])
def reset_session():
    if "session_id" in session:
        sid = session.pop("session_id")
        simulation_state.pop(sid, None)
    return redirect(url_for("index"))


def econ_cycle_loop():
    while True:
        try:
            economic_manager.update_cycle()
            # mine a block every minute (simple)
            blockchain.mine_block()
            time.sleep(60)
        except Exception:
            print("[ERROR] econ_cycle_loop", traceback.format_exc())
            time.sleep(5)

def crew_loop():
    while True:
        try:
            print(f"[{datetime.now().isoformat()}] Running Crew kickoff...")
            # kickoff returns results or raises if LLM error; we guard exceptions
            try:
                res = crew.kickoff()
                # you can store or process res here
                print("[Crew] kickoff completed.")
            except Exception as e:
                print("[Crew] kickoff failed:", repr(e))
                # print minimal trace
                traceback.print_exc()
        except Exception as outer:
            print("[ERROR] crew_loop outer", traceback.format_exc())
        time.sleep(3600)  # run hourly

if __name__ == "__main__":
    # Background threads
    threading.Thread(target=econ_cycle_loop, daemon=True).start()
    threading.Thread(target=crew_loop, daemon=True).start()

    # Run Flask without the reloader to avoid debugpy/threading issues
    app.run(host="0.0.0.0", port=PORT, debug=True, use_reloader=False)
