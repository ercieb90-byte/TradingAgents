"""
Portfolio Analyzer Orchestrator.

Manages the full multi-agent analysis pipeline:
  1. Fetch market data (yfinance, FRED, Finnhub)
  2. Run 4 specialist analyst agents in sequence
  3. Run bull/bear researcher debate for N rounds
  4. Run portfolio manager for final structured decision

Main classes:
    PortfolioAnalyzer  – public API for single or batch analysis
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

from portfolio_analyzer.config import Config
from portfolio_analyzer.llm.client import LLMClient
from portfolio_analyzer.data import yfinance_fetcher, fred_fetcher, finnhub_fetcher
from portfolio_analyzer.agents import (
    market_analyst,
    macro_analyst,
    fundamental_analyst,
    sentiment_analyst,
    researcher,
    portfolio_manager,
)


class PortfolioAnalyzer:
    """
    Multi-agent portfolio analyzer.

    Usage::

        from portfolio_analyzer import PortfolioAnalyzer
        from portfolio_analyzer.config import Config

        pa = PortfolioAnalyzer(Config())
        result = pa.analyze("AAPL")
        print(result["decision"]["action"])

    Or analyze a whole portfolio::

        results = pa.analyze_portfolio(["AAPL", "NVDA", "MSFT"])
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.llm    = LLMClient(self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Run the full analysis pipeline for a single *ticker*.

        Returns a dict with:
            ticker          - stock symbol
            decision        - structured decision dict (matches DECISION_SCHEMA)
            analyst_reports - dict with keys: technical, macro, fundamental, sentiment
            bull_arguments  - list of bull arguments per debate round
            bear_arguments  - list of bear arguments per debate round
            data_summary    - key metrics snapshot
            error           - None on success, error message on failure
        """
        ticker = ticker.upper().strip()
        print(f"\n{'='*60}")
        print(f"  Portfolio Analyzer: {ticker}")
        print(f"{'='*60}")

        result: Dict[str, Any] = {
            "ticker":          ticker,
            "decision":        {},
            "analyst_reports": {},
            "bull_arguments":  [],
            "bear_arguments":  [],
            "data_summary":    {},
            "error":           None,
        }

        try:
            # ----------------------------------------------------------------
            # Step 1: Fetch market data
            # ----------------------------------------------------------------
            data_bundle = self._fetch_all_data(ticker)

            # ----------------------------------------------------------------
            # Step 2: Run 4 analyst agents
            # ----------------------------------------------------------------
            analyst_reports = self._run_analysts(ticker, data_bundle)
            result["analyst_reports"] = analyst_reports

            # ----------------------------------------------------------------
            # Step 3: Bull/Bear debate
            # ----------------------------------------------------------------
            bull_args, bear_args = self._run_debate(
                ticker, data_bundle, analyst_reports
            )
            result["bull_arguments"] = bull_args
            result["bear_arguments"] = bear_args

            # ----------------------------------------------------------------
            # Step 4: Portfolio manager final decision
            # ----------------------------------------------------------------
            print(f"  [5/5] Portfolio Manager – Treffe finale Entscheidung...")
            decision = portfolio_manager.decide(
                ticker=ticker,
                data_bundle=data_bundle,
                analyst_reports=analyst_reports,
                bull_arguments=bull_args,
                bear_arguments=bear_args,
                llm_client=self.llm,
            )
            result["decision"] = decision

            # ----------------------------------------------------------------
            # Build data summary for quick reference
            # ----------------------------------------------------------------
            result["data_summary"] = _build_data_summary(data_bundle)

            print(f"\n  ✓ Analyse abgeschlossen: {ticker}")
            action     = decision.get("action", "?")
            conviction = decision.get("conviction", "?")
            target     = decision.get("target_price_12m", "?")
            print(f"    → Entscheidung: {action} | Überzeugung: {conviction} | Kursziel: ${target}")

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            result["error"] = error_msg
            print(f"\n  ✗ Fehler bei der Analyse von {ticker}: {exc}")

        return result

    def analyze_portfolio(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze a list of tickers and return results sorted by action priority.

        Sort order: Buy → Hold → Sell → Error

        Args:
            tickers: List of stock symbols.

        Returns:
            List of result dicts, sorted as described above.
        """
        results = []
        total = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{total}] Analysiere {ticker.upper()}...")
            result = self.analyze(ticker)
            results.append(result)

        # Sort: Buy first, then Hold, then Sell, errors last
        action_order = {"Buy": 0, "Hold": 1, "Sell": 2}
        conviction_order = {"High": 0, "Medium": 1, "Low": 2}

        def sort_key(r: Dict[str, Any]):
            if r.get("error"):
                return (10, 10, r.get("ticker", ""))
            decision = r.get("decision", {})
            action    = decision.get("action", "Hold")
            conviction = decision.get("conviction", "Low")
            a_order   = action_order.get(action, 5)
            c_order   = conviction_order.get(conviction, 5)
            return (a_order, c_order, r.get("ticker", ""))

        return sorted(results, key=sort_key)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _fetch_all_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch data from all three data sources."""
        print(f"\n  [1/5] Lade Marktdaten für {ticker}...")

        # yfinance (always)
        print(f"    → yfinance: Kursdaten, Technicals, Fundamentals...")
        yf_data = yfinance_fetcher.fetch(ticker)
        if yf_data.get("error"):
            print(f"    ⚠ yfinance Fehler: {yf_data['error']}")
        else:
            current = yf_data.get("price", {}).get("current", "?")
            name    = yf_data.get("meta", {}).get("name", ticker)
            print(f"    ✓ {name} – Aktueller Kurs: ${current}")

        # FRED (if key available)
        if self.config.fred_api_key:
            print(f"    → FRED: Makrodaten...")
            fred_data = fred_fetcher.fetch(self.config.fred_api_key)
            if fred_data.get("available"):
                n_series = len([s for s in fred_data.get("series", {}).values()
                                if s.get("latest") is not None])
                print(f"    ✓ FRED: {n_series} Zeitreihen geladen")
            else:
                print(f"    ⚠ FRED: {fred_data.get('reason', 'Nicht verfügbar')}")
        else:
            print(f"    → FRED: Kein API-Key (übersprungen)")
            fred_data = {"available": False, "reason": "FRED_API_KEY nicht konfiguriert"}

        # Finnhub (if key available)
        if self.config.finnhub_api_key:
            print(f"    → Finnhub: News, Sentiment, Earnings...")
            finnhub_data = finnhub_fetcher.fetch(ticker, self.config.finnhub_api_key)
            if finnhub_data.get("available"):
                n_news = len(finnhub_data.get("news", []))
                n_earnings = len(finnhub_data.get("earnings", []))
                print(f"    ✓ Finnhub: {n_news} Artikel, {n_earnings} Earnings-Quartale")
            else:
                print(f"    ⚠ Finnhub: {finnhub_data.get('reason', 'Nicht verfügbar')}")
        else:
            print(f"    → Finnhub: Kein API-Key (übersprungen)")
            finnhub_data = {"available": False, "reason": "FINNHUB_API_KEY nicht konfiguriert"}

        bundle = {
            "yfinance": yf_data,
            "fred":     fred_data,
            "finnhub":  finnhub_data,
            "_language": self.config.output_language,
            "_ticker":  ticker,
        }
        return bundle

    def _run_analysts(
        self,
        ticker: str,
        data_bundle: Dict[str, Any],
    ) -> Dict[str, str]:
        """Run the four specialist analyst agents sequentially."""
        print(f"\n  [2/5] Lasse Analysten arbeiten...")

        reports: Dict[str, str] = {}

        # Technical analyst
        print(f"    → Technischer Analyst...")
        try:
            reports["technical"] = market_analyst.analyze(ticker, data_bundle, self.llm)
            print(f"    ✓ Technische Analyse abgeschlossen ({len(reports['technical'])} Zeichen)")
        except Exception as exc:
            reports["technical"] = f"Fehler in der technischen Analyse: {exc}"
            print(f"    ✗ Technische Analyse fehlgeschlagen: {exc}")

        # Macro analyst
        print(f"    → Makro-Analyst (FRED)...")
        try:
            reports["macro"] = macro_analyst.analyze(ticker, data_bundle, self.llm)
            print(f"    ✓ Makro-Analyse abgeschlossen ({len(reports['macro'])} Zeichen)")
        except Exception as exc:
            reports["macro"] = f"Fehler in der Makro-Analyse: {exc}"
            print(f"    ✗ Makro-Analyse fehlgeschlagen: {exc}")

        # Fundamental analyst
        print(f"    → Fundamental-Analyst...")
        try:
            reports["fundamental"] = fundamental_analyst.analyze(ticker, data_bundle, self.llm)
            print(f"    ✓ Fundamental-Analyse abgeschlossen ({len(reports['fundamental'])} Zeichen)")
        except Exception as exc:
            reports["fundamental"] = f"Fehler in der Fundamental-Analyse: {exc}"
            print(f"    ✗ Fundamental-Analyse fehlgeschlagen: {exc}")

        # Sentiment analyst
        print(f"    → Sentiment-Analyst (Finnhub)...")
        try:
            reports["sentiment"] = sentiment_analyst.analyze(ticker, data_bundle, self.llm)
            print(f"    ✓ Sentiment-Analyse abgeschlossen ({len(reports['sentiment'])} Zeichen)")
        except Exception as exc:
            reports["sentiment"] = f"Fehler in der Sentiment-Analyse: {exc}"
            print(f"    ✗ Sentiment-Analyse fehlgeschlagen: {exc}")

        return reports

    def _run_debate(
        self,
        ticker: str,
        data_bundle: Dict[str, Any],
        analyst_reports: Dict[str, str],
    ) -> tuple[List[str], List[str]]:
        """Run the bull/bear debate for the configured number of rounds."""
        rounds = self.config.debate_rounds
        language = self.config.output_language
        print(f"\n  [3-4/5] Bullen vs. Bären Debatte ({rounds} Runden)...")

        bull_args: List[str] = []
        bear_args: List[str] = []

        last_bear: Optional[str] = None
        last_bull: Optional[str] = None

        for round_num in range(1, rounds + 1):
            # Bull makes their case
            print(f"    → Runde {round_num}: Bulle argumentiert...")
            try:
                bull = researcher.make_bull_case(
                    ticker=ticker,
                    all_analyst_reports=analyst_reports,
                    previous_bear_argument=last_bear,
                    llm_client=self.llm,
                    language=language,
                )
                bull_args.append(bull)
                last_bull = bull
                print(f"    ✓ Bullen-Argument Runde {round_num} ({len(bull)} Zeichen)")
            except Exception as exc:
                msg = f"Fehler im Bullen-Argument Runde {round_num}: {exc}"
                bull_args.append(msg)
                last_bull = msg
                print(f"    ✗ {msg}")

            # Bear makes their case
            print(f"    → Runde {round_num}: Bär argumentiert...")
            try:
                bear = researcher.make_bear_case(
                    ticker=ticker,
                    all_analyst_reports=analyst_reports,
                    previous_bull_argument=last_bull,
                    llm_client=self.llm,
                    language=language,
                )
                bear_args.append(bear)
                last_bear = bear
                print(f"    ✓ Bären-Argument Runde {round_num} ({len(bear)} Zeichen)")
            except Exception as exc:
                msg = f"Fehler im Bären-Argument Runde {round_num}: {exc}"
                bear_args.append(msg)
                last_bear = msg
                print(f"    ✗ {msg}")

        return bull_args, bear_args


# ---------------------------------------------------------------------------
# Data summary helper
# ---------------------------------------------------------------------------

def _build_data_summary(data_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics for a concise data summary."""
    yf   = data_bundle.get("yfinance", {})
    fred = data_bundle.get("fred", {})

    price = yf.get("price", {})
    tech  = yf.get("technicals", {})
    funda = yf.get("fundamentals", {})
    meta  = yf.get("meta", {})

    summary: Dict[str, Any] = {
        "company":      meta.get("name"),
        "sector":       meta.get("sector"),
        "current_price": price.get("current"),
        "52w_high":     price.get("52w_high"),
        "52w_low":      price.get("52w_low"),
        "return_1m":    price.get("return_1m"),
        "return_1y":    price.get("return_1y"),
        "rsi_14":       tech.get("rsi_14"),
        "golden_cross": tech.get("golden_cross"),
        "pe_forward":   funda.get("pe_forward"),
        "pe_trailing":  funda.get("pe_trailing"),
        "market_cap":   funda.get("market_cap"),
        "beta":         funda.get("beta"),
        "roe":          funda.get("roe"),
        "profit_margin": funda.get("profit_margin"),
        "revenue_growth": funda.get("revenue_growth"),
        "dividend_yield": funda.get("dividend_yield"),
    }

    # Add key FRED indicators if available
    if fred.get("available"):
        series  = fred.get("series", {})
        derived = fred.get("derived", {})
        summary["macro"] = {
            "fed_funds_rate":        series.get("FEDFUNDS", {}).get("latest"),
            "cpi_yoy_pct":           derived.get("cpi_yoy_pct"),
            "yield_curve_inverted":  derived.get("yield_curve_inverted"),
            "vix":                   series.get("VIXCLS", {}).get("latest"),
            "unemployment":          series.get("UNRATE", {}).get("latest"),
        }

    return summary
