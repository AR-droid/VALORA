# VALORAValora: Agentic AI Economic Simulation (Backend + ML + CrewAI + Flask)

Valora is a multi-agent artificial economy built entirely in Python.
It combines machine learning, agentic AI (CrewAI), and a simulation engine to model how consumers, firms, banks, and regulators behave inside an economy under different shocks and policies.

This version includes:

Machine learning agent brains

CrewAI orchestration layer

Simulation engine

Flask backend

Blockchain logging hooks

The metaverse visualization layer will be added in a later stage.

Project Overview

Valora simulates an economy in cycles.
In each cycle:

Agents observe the economic state

ML models decide actions (spending, pricing, lending, taxation)

The simulation engine updates the economy (market clearing, inflation, GDP)

Agents learn from rewards

A snapshot is generated and hashed

Later, the hash is posted on the blockchain

This allows the entire simulated economy to be:

transparent

reproducible

ML-driven

agentically orchestrated

Project Structure
valora/
│
├── agents/
│   ├── consumer/
│   │   ├── consumer_brain.py        # Q-learning spending model
│   │   ├── consumer_wrapper.py      # CrewAI wrapper (later)
│   │   ├── consumer_state.py        # State container
│   │   └── config.json
│   │
│   ├── firm/
│   │   ├── firm_brain.py            # heuristic pricing + production logic
│   │   ├── firm_wrapper.py
│   │   ├── firm_state.py
│   │   └── config.json
│   │
│   ├── bank/
│   │   ├── bank_brain.py            # RandomForest credit risk model
│   │   ├── bank_wrapper.py
│   │   ├── bank_state.py
│   │   └── config.json
│   │
│   ├── regulator/
│   │   ├── regulator_brain.py       # PPO model for tax-rate adjustment
│   │   ├── regulator_wrapper.py
│   │   ├── regulator_state.py
│   │   └── config.json
│   │
│   └── shared/
│       ├── base_agent.py            # base interface for all ML brains
│       ├── rl_utils.py              # common RL utilities
│       └── preprocessors.py         # state preprocessing and normalization
│
│
├── simulation/
│   ├── simulator.py                 # main cycle loop
│   ├── market.py                    # supply-demand clearing logic
│   ├── shocks.py                    # external shocks (inflation, covid)
│   ├── metrics.py                   # GDP, inflation, unemployment calculations
│   ├── rewards.py                   # reward functions for agents
│   ├── scenario_loader.py           # load scenario configurations
│   └── state_manager.py             # global economic state container
│
│
├── crew/
│   ├── crew_orchestrator.py         # CrewAI orchestrator for one simulation cycle
│   ├── crew_factory.py              # builds all agent crews
│   └── context_builder.py           # converts simulation state into CrewAI-readable context
│
│
├── blockchain/
│   ├── contracts/
│   │   ├── ValoraLedger.sol         # stores hashed snapshots per cycle
│   │   └── CurrencyToken.sol        # ERC-20 token used internally
│   ├── scripts/
│   │   ├── deploy.js                # Hardhat deploy script
│   │   └── post_cycle.js            # upload snapshot hash
│   ├── blockchain_client.py         # Python interface to Ethereum node
│   └── hashing.py                   # SHA256 hash utilities for snapshots
│
│
├── api/
│   ├── app.py                       # Flask entrypoint
│   └── routes/
│       ├── simulation_routes.py     # /run, /cycle, /snapshot
│       ├── agent_routes.py          # debugging routes for agent info
│       └── blockchain_routes.py     # verify snapshot hashes
│
│
├── data/
│   ├── snapshots/                   # JSON snapshots per cycle
│   ├── models/                      # saved PPO/Q-learning/random forest models
│   └── scenarios/                   # scenario templates
│
├── tests/                           # unit tests
│
├── run.py                           # end-to-end runner
└── requirements.txt



ML Agent Brains Included
ConsumerAgent (Q-Learning)

Inputs: inflation, savings, income

Actions: spending fractions (0.3, 0.5, 0.7, 0.9)

Reward: utility minus inflation penalty

FirmAgent (Heuristic, DQN later)

Adjusts price based on demand

Plans production based on capacity

Reward: profit

BankAgent (RandomForest)

Approve/deny loans based on credit risk

Reward: minimize default probability

RegulatorAgent (PPO)

Adjusts tax rate

Reward: closer inflation + unemployment to targets
