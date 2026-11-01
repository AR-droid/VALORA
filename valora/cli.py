"""Command line: ``valora serve | step | council | copilot | bench | scenarios``."""

from __future__ import annotations

import argparse
import json
import os


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="valora", description="VALORA agentic economy")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--port", type=int, default=int(os.getenv("PORT", 5004)))

    st = sub.add_parser("step", help="run the economy for N months and print indicators")
    st.add_argument("ticks", type=int, nargs="?", default=12)
    st.add_argument("--seed", type=int, default=7)

    c = sub.add_parser("council", help="put a policy to the economic council")
    c.add_argument("policy")

    cp = sub.add_parser("copilot", help="ask the policy copilot to reach a goal")
    cp.add_argument("goal")
    cp.add_argument("--horizon", type=int, default=24)

    b = sub.add_parser("bench", help="run the agent benchmark")
    b.add_argument("--agents", nargs="*")
    b.add_argument("--scenarios", nargs="*")
    b.add_argument("--json", action="store_true", help="print full JSON results")

    sub.add_parser("scenarios", help="list benchmark scenarios")

    args = ap.parse_args(argv)

    if args.cmd == "serve":
        from .web.app import create_app

        create_app().run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
    elif args.cmd == "step":
        from .engine import Economy, EconomyConfig

        e = Economy(EconomyConfig(seed=args.seed))
        for snap in e.run(args.ticks):
            print(f"t={snap['tick']:3d} {snap['phase']:9s} gdp={snap['gdp']:8.1f} infl={snap['inflation']:5.2f}% "
                  f"unemp={snap['unemployment']:5.2f}% conf={snap['consumer_confidence']:5.1f} tax={snap['tax_rate']:.3f}")
    elif args.cmd in {"council", "copilot"}:
        from .agentic.tools import ToolContext
        from .engine import Economy

        ctx = ToolContext(Economy())
        if args.cmd == "council":
            from .agentic.council import EconomicCouncil

            r = EconomicCouncil(ctx).deliberate(args.policy)
            for a in r["agents"]:
                print(f"\n## {a['agent']} (vote: {a['vote']})\n{a['output']}")
            print("\n" + r["final_output"])
        else:
            from .agentic.copilot import PolicyCopilot

            r = PolicyCopilot(ctx).run(args.goal, horizon=args.horizon)
            for step in r["trace"]:
                label = step.get("tool") or step["type"]
                print(f"[{step['step']}] {label}")
            print("\n" + r["answer"])
            print(json.dumps(r["proposal"], indent=2))
    elif args.cmd == "bench":
        from .harness.bench import run_benchmark

        res = run_benchmark(args.agents, args.scenarios, progress=lambda m: print("running", m, flush=True))
        if args.json:
            for r in res["results"]:
                r.pop("series", None)
            print(json.dumps(res, indent=2))
        else:
            print("\nAgent".ljust(20), "mean", *[s[:12].rjust(13) for s in res["scenarios"]])
            for row in res["ranking"]:
                print(row["agent"].ljust(19), f"{row['mean_score']:6.2f}",
                      *[f"{row['scores'].get(s, 0):13.2f}" for s in res["scenarios"]])
    elif args.cmd == "scenarios":
        from .harness.bench import load_scenarios

        for name, sc in load_scenarios().items():
            print(f"{name:18s} {sc.description}")


if __name__ == "__main__":
    main()
