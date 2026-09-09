#!/usr/bin/env python
"""
scripts/run_agent.py
─────────────────────────────────────────────────────────────────────────────
CLI entry point for the SpotifyCares support agent.

Usage:
  python scripts/run_agent.py --message "Why did I get charged twice?"
  python scripts/run_agent.py --message "App crashes on iPhone" --provider stub
  python scripts/run_agent.py --interactive

Output:
  JSON to stdout (pretty-printed)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def run_once(message: str, provider: str, config_path: Path) -> dict:
    from src.agent import SpotifySupportAgent
    agent = SpotifySupportAgent(config_path=config_path)
    # Override provider from CLI
    agent.provider = provider
    agent._generator = None  # force re-init with new provider
    return agent.run(message)


def print_result(result: dict) -> None:
    """Pretty-print the agent result with colour hints for the terminal."""
    try:
        from rich.console import Console
        from rich.syntax import Syntax
        console = Console()
        text = json.dumps(result, indent=2, ensure_ascii=False)
        console.print(Syntax(text, "json", theme="monokai"))
    except ImportError:
        print(json.dumps(result, indent=2, ensure_ascii=False))


def interactive_mode(provider: str, config_path: Path) -> None:
    """REPL mode — type messages, see responses."""
    print("SpotifyCares Support Agent (interactive mode). Type 'quit' to exit.\n")
    from src.agent import SpotifySupportAgent
    agent = SpotifySupportAgent(config_path=config_path)
    agent.provider = provider
    agent._generator = None

    while True:
        try:
            msg = input("Customer > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if msg.lower() in ("quit", "exit", "q"):
            break
        if not msg:
            continue
        result = agent.run(msg)
        print_result(result)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SpotifyCares AI support agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_agent.py --message "I can't log into my account"
  python scripts/run_agent.py --message "Why am I seeing ads on Premium?" --provider stub
  python scripts/run_agent.py --interactive
        """,
    )
    parser.add_argument("--message", "-m", type=str, default=None,
                        help="Customer message to process")
    parser.add_argument("--provider", type=str, default=None,
                        help="LLM provider: gemini | openai | stub (overrides config.yaml)")
    parser.add_argument("--config", type=str, default=str(ROOT / "config.yaml"),
                        help="Path to config.yaml")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive REPL mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show debug logs")
    args = parser.parse_args()

    setup_logging(args.verbose)
    config_path = Path(args.config)

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    provider = args.provider or cfg["generation"].get("provider", "stub")

    if args.interactive:
        interactive_mode(provider, config_path)
    elif args.message:
        result = run_once(args.message, provider, config_path)
        print_result(result)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
