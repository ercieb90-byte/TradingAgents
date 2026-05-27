"""
Portfolio Manager Agent.

Synthesizes all analyst reports, bull/bear debate arguments, and raw data
into a final structured investment decision with actionable recommendations.

The decision schema is enforced via:
  - Anthropic: tool_use with tool_choice={"type": "tool", "name": "output"}
  - Llama:     JSON mode with schema appended to system prompt
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Decision schema (JSON Schema)
# ---------------------------------------------------------------------------

DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["Buy", "Hold", "Sell"],
            "description": "Investment action recommendation",
        },
        "conviction": {
            "type": "string",
            "enum": ["High", "Medium", "Low"],
            "description": "Confidence level in the recommendation",
        },
        "position_size_pct": {
            "type": "number",
            "description": "Suggested position size as % of portfolio (0-100)",
            "minimum": 0,
            "maximum": 100,
        },
        "entry_price": {
            "type": "number",
            "description": "Suggested entry price or current price for immediate entry",
        },
        "stop_loss_price": {
            "type": "number",
            "description": "Stop-loss price level for risk management",
        },
        "target_price_12m": {
            "type": "number",
            "description": "12-month price target",
        },
        "time_horizon": {
            "type": "string",
            "enum": ["Short (<1M)", "Medium (1-6M)", "Long (6M+)"],
            "description": "Recommended investment time horizon",
        },
        "key_risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Top 3-5 key risks to the thesis",
        },
        "key_catalysts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Top 3-5 key catalysts that could drive the thesis",
        },
        "score_technical": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "Technical analysis score (1=very bearish, 10=very bullish)",
        },
        "score_fundamental": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "Fundamental analysis score (1=very poor, 10=excellent)",
        },
        "score_macro": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "Macro environment score for this stock (1=very unfavorable, 10=very favorable)",
        },
        "score_sentiment": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "Sentiment score (1=very negative, 10=very positive)",
        },
        "summary": {
            "type": "string",
            "description": "2-3 sentence executive summary of the investment thesis",
        },
    },
    "required": [
        "action", "conviction", "position_size_pct", "entry_price",
        "stop_loss_price", "target_price_12m", "time_horizon",
        "key_risks", "key_catalysts",
        "score_technical", "score_fundamental", "score_macro", "score_sentiment",
        "summary",
    ],
}


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def decide(
    ticker: str,
    data_bundle: Dict[str, Any],
    analyst_reports: Dict[str, str],
    bull_arguments: List[str],
    bear_arguments: List[str],
    llm_client: Any,
) -> Dict[str, Any]:
    """
    Make the final investment decision for *ticker*.

    Args:
        ticker:           Stock symbol.
        data_bundle:      Full data dict from the orchestrator.
        analyst_reports:  Dict with keys "technical", "macro", "fundamental", "sentiment".
        bull_arguments:   List of bull arguments from each debate round.
        bear_arguments:   List of bear arguments from each debate round.
        llm_client:       LLMClient instance.

    Returns:
        Dict matching DECISION_SCHEMA.
    """
    yf_data      = data_bundle.get("yfinance", {})
    meta         = yf_data.get("meta", {})
    price_data   = yf_data.get("price", {})
    tech_data    = yf_data.get("technicals", {})
    language     = data_bundle.get("_language", "German")

    company_name  = meta.get("name", ticker)
    current_price = price_data.get("current")
    atr           = tech_data.get("atr_14")

    context_block = _build_context(
        ticker, company_name, current_price, atr,
        analyst_reports, bull_arguments, bear_arguments
    )

    system_prompt = (
        "Du bist ein erfahrener Portfolio-Manager mit 25 Jahren Erfahrung in der Vermögensverwaltung. "
        "Du hast eine Vielzahl von Analysten- und Research-Berichten sowie eine Bullen/Bären-Debatte "
        "gesehen und musst jetzt eine finale, fundierte Investitionsentscheidung treffen. "
        "Deine Entscheidungen sind strukturiert, risikobasiert und klar kommuniziert. "
        f"Du antwortest auf {language} (außer für technische Felder wie 'action', 'conviction', "
        f"'time_horizon' die in Englisch bleiben müssen gemäß dem Schema). "
        "Sei präzise und nutze alle vorliegenden Informationen."
    ) if language == "German" else (
        "You are an experienced portfolio manager with 25 years in asset management. "
        "You have reviewed analyst reports and a bull/bear debate and must now make a "
        "final, well-founded investment decision. "
        "Your decisions are structured, risk-based and clearly communicated. "
        f"Respond in {language}. "
        "Be precise and use all available information."
    )

    user_prompt = f"""Du hast die Analyse für **{ticker} ({company_name})** abgeschlossen.
Aktueller Kurs: ${current_price if current_price else 'N/A'}
ATR(14): ${atr if atr else 'N/A'} (für Stop-Loss Kalkulation)

{context_block}

Auf Basis aller obigen Informationen – Analystenberichte, Bullen-Argumente, Bären-Argumente
und Rohdaten – treffe jetzt eine finale strukturierte Investitionsentscheidung.

Wichtige Hinweise für die Entscheidungsfelder:
- **action**: "Buy" / "Hold" / "Sell"
- **conviction**: "High" / "Medium" / "Low"
- **position_size_pct**: 0-5% = kleine Position, 5-10% = normale Position, 10%+ = große Überzeugungsposition
- **entry_price**: aktueller Kurs oder leicht günstigerer Einstiegskurs
- **stop_loss_price**: typisch 1-3x ATR unterhalb des Einstiegs für Long-Positionen
- **target_price_12m**: realistisches 12-Monats-Kursziel
- **scores**: 1-10 für jeden Bereich (Technical, Fundamental, Macro, Sentiment)
- **summary**: 2-3 Sätze Executive Summary auf {language}

Rufe das 'output' Tool mit deiner vollständigen strukturierten Entscheidung auf.
""" if language == "German" else f"""You have completed the analysis for **{ticker} ({company_name})**.
Current Price: ${current_price if current_price else 'N/A'}
ATR(14): ${atr if atr else 'N/A'} (for stop-loss calculation)

{context_block}

Based on all the above information – analyst reports, bull arguments, bear arguments,
and raw data – make a final structured investment decision.

Key notes for decision fields:
- **action**: "Buy" / "Hold" / "Sell"
- **conviction**: "High" / "Medium" / "Low"
- **position_size_pct**: 0-5% = small position, 5-10% = normal position, 10%+ = high conviction
- **entry_price**: current price or slightly better entry price
- **stop_loss_price**: typically 1-3x ATR below entry for long positions
- **target_price_12m**: realistic 12-month price target
- **scores**: 1-10 for each area (Technical, Fundamental, Macro, Sentiment)
- **summary**: 2-3 sentences executive summary

Call the 'output' tool with your complete structured decision.
"""

    return llm_client.structured(system_prompt, user_prompt, DECISION_SCHEMA)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context(
    ticker: str,
    company_name: str,
    current_price: Optional[float],
    atr: Optional[float],
    analyst_reports: Dict[str, str],
    bull_arguments: List[str],
    bear_arguments: List[str],
) -> str:
    """Build the full context block for the portfolio manager prompt."""
    sections = []

    # Analyst reports summary
    sections.append("=" * 60)
    sections.append("ANALYSTEN-BERICHTE")
    sections.append("=" * 60)

    report_labels = {
        "technical":    "TECHNISCHE ANALYSE",
        "macro":        "MAKRO-ANALYSE",
        "fundamental":  "FUNDAMENTALANALYSE",
        "sentiment":    "SENTIMENT-ANALYSE",
    }
    for key, label in report_labels.items():
        report = analyst_reports.get(key)
        if report:
            sections.append(f"\n--- {label} ---")
            # Truncate very long reports to keep prompt manageable
            sections.append(report[:2000] if len(report) > 2000 else report)

    # Bull/Bear debate
    sections.append("")
    sections.append("=" * 60)
    sections.append("BULLEN vs. BÄREN DEBATTE")
    sections.append("=" * 60)

    n_rounds = max(len(bull_arguments), len(bear_arguments))
    for i in range(n_rounds):
        round_num = i + 1
        sections.append(f"\n--- Runde {round_num} ---")

        if i < len(bull_arguments):
            bull = bull_arguments[i]
            sections.append(f"\n[BULLE - Runde {round_num}]")
            sections.append(bull[:1500] if len(bull) > 1500 else bull)

        if i < len(bear_arguments):
            bear = bear_arguments[i]
            sections.append(f"\n[BÄR - Runde {round_num}]")
            sections.append(bear[:1500] if len(bear) > 1500 else bear)

    return "\n".join(sections)
