# VALORA

**An agentic AI economy.** Learning agents (households, firms, a bank, a regulator) run a simulated economy month by month. AI agents make policy for it: a tool-using **Policy Copilot** works toward goals you set in plain English, and a five-member **Economic Council** debates every proposal and votes on it before anything reaches the economy. A **benchmark harness** scores policy-making agents on seeded crisis scenarios, and every tick, decision and vote is written to a **hash-chained ledger**.

It runs fully offline with deterministic agents. Add a Groq key (or any OpenAI-compatible API) and the copilot and council switch to LLM reasoning.

## What's inside

| Part | What it does | Code |
| --- | --- | --- |
| Agent economy | 60 households (Q-learning spending), 5 firms (learning markups), a bank (RandomForest credit model refit on real defaults), a regulator (Taylor-style rule, or PPO with `.[rl]`). Markets clear every month; GDP, inflation, unemployment, confidence and wealth inequality come out of agent behaviour. Business-cycle phases and shocks act through productivity, costs, credit and confidence. | `valora/engine/economy.py`, `valora/agents/` |
| Policy levers | Tax, government spending, interest rate, household transfers and regulation, each with hard bounds that every agent's output is clamped to. | `valora/engine/economy.py` (`LEVER_BOUNDS`) |
| What-if simulation | Forks the live economy and runs it with and without a policy from the same random state, so the difference is the policy's effect. Every agent decision is based on this evidence. | `valora/agentic/policy.py` |
| Policy Copilot | Goal-driven agent with 8 tools (status, simulate, compare, stress test, sector impact, parse, memory recall, ledger audit). With an LLM it uses native tool calling with a step budget; offline it runs a search planner. Its final proposal is re-simulated before it's shown, so the numbers are checked rather than taken from the model. | `valora/agentic/copilot.py`, `tools.py` |
| Economic Council | Parser, Economic Analyst, Tax Advisor, a two-round Supporter vs Opposer debate, and a Critic who rules Proceed, Modify or Pause. A weighted vote decides. Approved policies can then be applied to the live economy. | `valora/agentic/council.py` |
| Memory | Every decision is stored with its prediction. Twelve months later the platform records what actually happened, and the agents can recall past decisions. | `valora/store.py`, `valora/runtime.py` |
| Benchmark harness | 6 seeded scenarios (baseline, demand collapse, energy shock, financial crisis, pandemic, overheating), 5 agents (no-policy, fixed stimulus, Taylor rule, offline copilot, LLM copilot), a welfare score, and a reproducible fingerprint hash per run. | `valora/harness/` |
| Ledger | SHA-256 blocks with Merkle roots, persisted to SQLite. It detects tampering. | `valora/ledger/chain.py` |
| Sector model | Heuristic model showing which of 15 sectors a policy text hits hardest (from the original VALORA prototype). | `valora/engine/actors.py` |
| Web app | Studio (council, sector simulation, ledger), Copilot and Benchmark pages, plus a REST API. | `valora/web/` |

Benchmark result with the offline agents (higher is better):

| Agent | Mean score | Best on |
| --- | --- | --- |
| copilot-offline | 0.44 | 5 of 6 scenarios |
| fixed-stimulus | -5.63 | overheating |
| taylor-rule | -6.62 | none |
| no-policy | -10.86 | none |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export GROQ_API_KEY=...        # optional; omit to run offline
python -m valora serve         # http://localhost:5004
```

Other commands:

```bash
python -m valora step 24                                  # run the economy for 24 months
python -m valora council "Cut income taxes by 3% for 2 years"
python -m valora copilot "Bring unemployment down without pushing inflation above 4%"
python -m valora bench                                    # benchmark table
pytest -q                                                 # tests
```

Docker: `docker build -t valora . && docker run -p 7860:7860 -e GROQ_API_KEY=... valora`

## Deploy to Hugging Face Spaces

1. Create a Hugging Face access token with **write** permission.
2. In this GitHub repo, add the secret `HF_TOKEN` and the variable `HF_SPACE` (for example `your-hf-name/valora`) under Settings → Secrets and variables → Actions.
3. Push to `main` or run the **Deploy to Hugging Face Spaces** workflow. It creates the Docker Space if needed and uploads the app.
4. In the Space settings, add `GROQ_API_KEY` as a secret to turn on the LLM agents. Without it, the Space runs in offline mode.

You can also deploy from your machine: `pip install huggingface_hub && HF_TOKEN=... HF_SPACE=your-name/valora python deploy/hf_space.py`.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `GROQ_API_KEY` / `LLM_API_KEY` | empty | Turns on LLM agents |
| `LLM_BASE_URL` | Groq's OpenAI-compatible URL | Any OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama, ...) |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model name |
| `AGENT_MAX_STEPS` | 6 | Model calls allowed per copilot run |
| `MAX_AGENT_COMPLETION_TOKENS` | 450 | Token cap per agent reply |
| `VALORA_TICK_SECONDS` | 30 | Seconds between live economy months (0 = manual only) |
| `VALORA_DB` | `valora.db` | SQLite file |
| `VALORA_SEED` | 7 | Economy seed |
| `VALORA_RL` | 0 | 1 = PPO regulator (needs `pip install ".[rl]"`) |
| `FLASK_SECRET_KEY` | random | Session signing key |

## API

| Method and path | Purpose |
| --- | --- |
| `GET /api/economic/status`, `/api/economic/history` | Live indicators and history |
| `POST /api/economic/step` `{ticks}` | Advance the economy |
| `POST /api/economic/shock` `{type, magnitude, duration}` | Inject a shock |
| `POST /api/whatif` `{levers, duration, horizon}` | Simulate a policy against a baseline |
| `POST /api/agents/run` `{text, apply?}` | Council deliberation |
| `POST /api/agents/<id>/apply` | Apply an approved council decision |
| `POST /api/policies` `{text, levers?}` | Deliberate and apply if approved |
| `POST /api/policies/<id>/revoke` | Revoke a live policy |
| `POST /api/copilot/run` `{goal, horizon}` | Run the Policy Copilot |
| `POST /api/copilot/<id>/submit` | Send a copilot proposal to the council |
| `POST /api/bench/run`, `GET /api/bench/status`, `/api/bench/leaderboard` | Benchmark |
| `GET /api/blockchain/ledger`, `/verify`, `/audit` | Ledger |
| `POST /api/simulation/run` `{policy, time_steps}` | Sector impact model |
| `GET /api/meta`, `/healthz` | Config and health |

## Credits

The web studio, sector model and avatar images come from the VALORA prototype at [Siddharth-777/VALORA-REPO](https://github.com/Siddharth-777/VALORA-REPO). Chart.js 4.4.1 is vendored under its MIT licence.
