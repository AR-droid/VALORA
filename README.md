#Valora: Agentic AI Economic Simulation (Backend + ML + CrewAI + Flask)

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
