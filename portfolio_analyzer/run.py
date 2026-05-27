"""
CLI entry point for the Portfolio Analyzer.

Usage:
    python -m portfolio_analyzer.run AAPL NVDA MSFT
    python -m portfolio_analyzer.run AAPL --provider llama --debate-rounds 3
    python -m portfolio_analyzer.run AAPL NVDA --output results.json --language English
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="portfolio_analyzer",
        description="Multi-agent LLM stock portfolio analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m portfolio_analyzer.run AAPL
  python -m portfolio_analyzer.run AAPL NVDA MSFT --provider anthropic
  python -m portfolio_analyzer.run AAPL --debate-rounds 3 --language English
  python -m portfolio_analyzer.run AAPL NVDA --output results.json
        """,
    )

    parser.add_argument(
        "tickers",
        nargs="+",
        metavar="TICKER",
        help="One or more stock symbols (e.g. AAPL NVDA MSFT)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "llama"],
        default=None,
        help="LLM provider (default: from config / anthropic)",
    )
    parser.add_argument(
        "--debate-rounds",
        type=int,
        default=None,
        metavar="N",
        help="Number of bull/bear debate rounds (default: 2)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Output language for reports (default: German)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--deep-model",
        default=None,
        metavar="MODEL",
        help="Override deep model name (for portfolio manager / researchers)",
    )
    parser.add_argument(
        "--fast-model",
        default=None,
        metavar="MODEL",
        help="Override fast model name (for analysts)",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        metavar="FILE",
        help="Path to .env file to load (default: .env or .env.portfolio)",
    )

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load .env file if requested or auto-detect
    # ------------------------------------------------------------------
    _load_env(args.env_file)

    # ------------------------------------------------------------------
    # Build config
    # ------------------------------------------------------------------
    from portfolio_analyzer.config import Config

    config = Config()

    if args.provider:
        config.llm_provider = args.provider
    if args.debate_rounds is not None:
        config.debate_rounds = args.debate_rounds
    if args.language:
        config.output_language = args.language
    if args.deep_model:
        config.deep_model = args.deep_model
    if args.fast_model:
        config.fast_model = args.fast_model

    # Print config summary
    _print_header(config, args.tickers)

    # ------------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------------
    from portfolio_analyzer.orchestrator import PortfolioAnalyzer

    analyzer = PortfolioAnalyzer(config)

    tickers = [t.upper().strip() for t in args.tickers]

    if len(tickers) == 1:
        results = [analyzer.analyze(tickers[0])]
    else:
        results = analyzer.analyze_portfolio(tickers)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    _print_summary_table(results)

    # ------------------------------------------------------------------
    # Save to file if requested
    # ------------------------------------------------------------------
    if args.output:
        output_path = Path(args.output)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n  Ergebnisse gespeichert: {output_path.resolve()}")
        except OSError as e:
            print(f"\n  Fehler beim Speichern: {e}", file=sys.stderr)

    # Exit code: 0 if at least one successful result, 1 if all failed
    any_success = any(not r.get("error") for r in results)
    sys.exit(0 if any_success else 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_env(env_file: Optional[str]) -> None:
    """Load environment variables from a .env file."""
    candidates: List[str] = []

    if env_file:
        candidates = [env_file]
    else:
        # Auto-detect common .env file names
        candidates = [
            ".env.portfolio",
            ".env",
            os.path.join(os.path.dirname(__file__), ".env.portfolio"),
            os.path.join(os.path.dirname(__file__), "..", ".env.portfolio"),
            os.path.join(os.path.dirname(__file__), "..", ".env"),
        ]

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            _parse_env_file(path)
            print(f"  Loaded environment from: {path.resolve()}")
            return


def _parse_env_file(path: Path) -> None:
    """Parse a simple .env file and set environment variables."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and blank lines
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key   = key.strip()
                value = value.strip()
                # Remove surrounding quotes
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                # Only set if not already set
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


def _print_header(config: Any, tickers: List[str]) -> None:
    """Print a startup header."""
    print("\n" + "=" * 70)
    print("  PORTFOLIO ANALYZER – Multi-Agent LLM Aktienanalyse")
    print("=" * 70)
    print(f"  Provider:      {config.llm_provider.upper()}")
    if config.llm_provider == "anthropic":
        print(f"  Analysten:     {config.fast_model}")
        print(f"  Manager:       {config.deep_model}")
    else:
        print(f"  Modell:        {config.llama_model}")
    print(f"  Debatte:       {config.debate_rounds} Runden")
    print(f"  Sprache:       {config.output_language}")
    print(f"  Aktien:        {', '.join(t.upper() for t in tickers)}")
    print(f"  FRED:          {'✓ Aktiviert' if config.fred_api_key else '✗ Kein Key'}")
    print(f"  Finnhub:       {'✓ Aktiviert' if config.finnhub_api_key else '✗ Kein Key'}")
    print("=" * 70)


def _print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted summary table of all analysis results."""
    print("\n" + "=" * 70)
    print("  ANALYSE-ZUSAMMENFASSUNG")
    print("=" * 70)

    if not results:
        print("  Keine Ergebnisse.")
        return

    # Header
    header = (
        f"  {'Ticker':<8} {'Name':<20} {'Action':<6} {'Conv.':<8} "
        f"{'Kurs':>8} {'Ziel12M':>9} {'Stop':>8} {'Score':>6}"
    )
    print(header)
    print("  " + "-" * 66)

    action_icons = {"Buy": "🟢", "Hold": "🟡", "Sell": "🔴"}

    for r in results:
        ticker  = r.get("ticker", "?")
        error   = r.get("error")
        decision = r.get("decision", {})
        summary  = r.get("data_summary", {})

        if error:
            print(f"  {ticker:<8} {'FEHLER':<20} {'–':<6} {'–':<8} {'–':>8} {'–':>9} {'–':>8} {'–':>6}")
            continue

        name       = (summary.get("company") or ticker)[:18]
        action     = decision.get("action", "?")
        conviction = decision.get("conviction", "?")[:3]
        price      = summary.get("current_price")
        target     = decision.get("target_price_12m")
        stop_loss  = decision.get("stop_loss_price")

        # Composite score
        scores = [
            decision.get("score_technical"),
            decision.get("score_fundamental"),
            decision.get("score_macro"),
            decision.get("score_sentiment"),
        ]
        valid_scores = [s for s in scores if s is not None]
        avg_score = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else None

        price_str  = f"${price:.2f}" if price is not None else "?"
        target_str = f"${target:.2f}" if target is not None else "?"
        stop_str   = f"${stop_loss:.2f}" if stop_loss is not None else "?"
        score_str  = f"{avg_score}/10" if avg_score is not None else "?"

        icon = action_icons.get(action, " ")
        print(
            f"  {ticker:<8} {name:<20} {icon}{action:<5} {conviction:<8} "
            f"{price_str:>8} {target_str:>9} {stop_str:>8} {score_str:>6}"
        )

    print("  " + "-" * 66)

    # Print individual summaries
    print("\n  ENTSCHEIDUNGSZUSAMMENFASSUNGEN:")
    print("  " + "-" * 66)
    for r in results:
        ticker   = r.get("ticker", "?")
        decision = r.get("decision", {})
        error    = r.get("error")

        if error:
            print(f"\n  {ticker}: FEHLER – {str(error)[:100]}")
            continue

        summary  = decision.get("summary", "")
        action   = decision.get("action", "?")
        conv     = decision.get("conviction", "?")
        risks    = decision.get("key_risks", [])
        catalysts = decision.get("key_catalysts", [])

        print(f"\n  {ticker} [{action} | {conv}]")
        if summary:
            # Word-wrap summary at ~60 chars
            words = summary.split()
            line  = "    "
            for word in words:
                if len(line) + len(word) > 66:
                    print(line)
                    line = "    " + word + " "
                else:
                    line += word + " "
            if line.strip():
                print(line)

        if catalysts:
            print(f"    Katalysatoren: {' | '.join(catalysts[:3])}")
        if risks:
            print(f"    Risiken:       {' | '.join(risks[:3])}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
