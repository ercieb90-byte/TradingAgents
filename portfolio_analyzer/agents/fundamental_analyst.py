"""
Fundamental / Valuation Analyst Agent.

Evaluates a stock's valuation, profitability, financial health, and
quality metrics — comparing to rough sector benchmarks where possible.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# Rough sector average PE ratios for context (approximate 2024 market data)
_SECTOR_PE_AVERAGES: Dict[str, float] = {
    "Technology":              28.0,
    "Communication Services":  22.0,
    "Consumer Discretionary":  24.0,
    "Consumer Staples":        20.0,
    "Health Care":             20.0,
    "Financials":              14.0,
    "Industrials":             22.0,
    "Materials":               16.0,
    "Energy":                  11.0,
    "Utilities":               17.0,
    "Real Estate":             30.0,
}

_SECTOR_PB_AVERAGES: Dict[str, float] = {
    "Technology":              7.0,
    "Communication Services":  3.5,
    "Consumer Discretionary":  4.5,
    "Consumer Staples":        5.0,
    "Health Care":             4.0,
    "Financials":              1.4,
    "Industrials":             3.5,
    "Materials":               2.5,
    "Energy":                  1.8,
    "Utilities":               1.7,
    "Real Estate":             2.0,
}


def analyze(ticker: str, data_bundle: Dict[str, Any], llm_client: Any) -> str:
    """
    Perform fundamental and valuation analysis for *ticker*.

    Args:
        ticker:      Stock symbol.
        data_bundle: Full data dict from the orchestrator.
        llm_client:  LLMClient instance.

    Returns:
        A string report with valuation assessment and quality score.
    """
    yf_data   = data_bundle.get("yfinance", {})
    meta      = yf_data.get("meta", {})
    funda     = yf_data.get("fundamentals", {})
    financials = yf_data.get("financials", {})
    analyst   = yf_data.get("analyst", {})
    language  = data_bundle.get("_language", "German")

    company_name = meta.get("name", ticker)
    sector       = meta.get("sector", "")
    industry     = meta.get("industry", "")

    fundamentals_summary = _format_fundamentals(
        ticker, company_name, sector, industry, funda, financials, analyst
    )

    system_prompt = (
        "Du bist ein erfahrener Value-Investor und Fundamentalanalyst mit Fokus auf Bewertung "
        "und Unternehmensqualität. Du bewertest Aktien anhand von Kennzahlen, Geschäftsmodell-Stärke "
        f"und finanzieller Gesundheit. Antworte ausschließlich auf {language}."
    ) if language == "German" else (
        "You are an experienced value investor and fundamental analyst focused on valuation "
        "and business quality. You evaluate stocks based on metrics, business model strength "
        f"and financial health. Always respond in {language}."
    )

    user_prompt = f"""Analysiere die folgenden Fundamentaldaten für {ticker} ({company_name})
im Sektor: {sector} | Branche: {industry}

{fundamentals_summary}

Erstelle einen fundamentalen Analysebericht mit:

1. **Bewertungsanalyse (Valuation)**:
   - KGV (trailing & forward): Fair, günstig oder teuer? Vergleich mit Sektordurchschnitt
   - PEG-Ratio: Wachstum eingepreist?
   - KBV und KUV: Substanzbewertung
   - EV/EBITDA: Enterprise Value Bewertung

2. **Qualitäts-Score** (1-10 skala):
   - Profitabilität (Margen, ROE, ROA)
   - Bilanzstärke (Schulden, Current Ratio)
   - Wachstum (Umsatz & Gewinnwachstum)
   - Dividende (falls vorhanden)

3. **Finanzielle Gesundheit**:
   - Verschuldungssituation: Problematisch oder gesund?
   - Liquidität: Kurzfristige Zahlungsfähigkeit
   - Free Cash Flow Qualität

4. **Quartalsentwicklung**:
   - Umsatz- und Gewinntrend (QoQ)
   - Qualität des Gewinnwachstums

5. **Analysten-Konsens**:
   - Kurszielvergleich (Upside/Downside zum aktuellen Kurs)
   - Empfehlungsverteilung

6. **Gesamturteil**:
   - Qualitätsscore (1-10)
   - Bewertungsurteil (günstig/fair/teuer)
   - 2-3 Satz Zusammenfassung

Sei konkret und quantitativ. Nutze Sektorvergleiche wo angegeben.
""" if language == "German" else f"""Analyze the following fundamental data for {ticker} ({company_name})
in sector: {sector} | Industry: {industry}

{fundamentals_summary}

Create a fundamental analysis report covering:

1. **Valuation Analysis**:
   - PE ratio (trailing & forward): Fair, cheap or expensive? Compare to sector average
   - PEG ratio: Is growth priced in?
   - P/B and P/S: Asset-based valuation
   - EV/EBITDA: Enterprise value assessment

2. **Quality Score** (1-10 scale):
   - Profitability (margins, ROE, ROA)
   - Balance sheet strength (debt, current ratio)
   - Growth (revenue & earnings growth)
   - Dividend (if applicable)

3. **Financial Health**:
   - Debt situation: Problematic or healthy?
   - Liquidity: Short-term payment ability
   - Free Cash Flow quality

4. **Quarterly Trends**:
   - Revenue and earnings trend (QoQ)
   - Quality of earnings growth

5. **Analyst Consensus**:
   - Price target comparison (upside/downside vs current price)
   - Recommendation distribution

6. **Overall Verdict**:
   - Quality score (1-10)
   - Valuation verdict (cheap/fair/expensive)
   - 2-3 sentence summary

Be concrete and quantitative. Use sector comparisons where provided.
"""

    return llm_client.complete(system_prompt, user_prompt, deep=False)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_fundamentals(
    ticker: str,
    name: str,
    sector: str,
    industry: str,
    funda: dict,
    financials: dict,
    analyst: dict,
) -> str:
    lines = []

    # Market cap
    mktcap = funda.get("market_cap")
    if mktcap:
        lines.append(f"Marktkapitalisierung: ${_fmt_large(mktcap)}")

    lines.append("")
    lines.append("--- BEWERTUNGSKENNZAHLEN ---")

    # Valuation ratios with sector comparison
    sector_pe = _SECTOR_PE_AVERAGES.get(sector)
    sector_pb = _SECTOR_PB_AVERAGES.get(sector)

    pe_trail = funda.get("pe_trailing")
    pe_fwd   = funda.get("pe_forward")
    peg      = funda.get("peg_ratio")
    pb       = funda.get("pb_ratio")
    ps       = funda.get("ps_ratio")
    ev_ebit  = funda.get("ev_ebitda")

    if pe_trail is not None:
        sector_note = ""
        if sector_pe:
            diff = ((pe_trail / sector_pe) - 1) * 100
            sector_note = f" (Sektordurchschnitt: {sector_pe}, {diff:+.0f}%)"
        lines.append(f"KGV (trailing): {pe_trail}{sector_note}")

    if pe_fwd is not None:
        lines.append(f"KGV (forward): {pe_fwd}")

    if peg is not None:
        peg_note = " → Wachstum fair bewertet" if 1.0 <= peg <= 1.5 else (
            " → Teuer relativ zum Wachstum" if peg > 2.0 else
            " → Günstig relativ zum Wachstum"
        )
        lines.append(f"PEG-Ratio: {peg}{peg_note}")

    if pb is not None:
        sector_note = ""
        if sector_pb:
            diff = ((pb / sector_pb) - 1) * 100
            sector_note = f" (Sektordurchschnitt: {sector_pb}, {diff:+.0f}%)"
        lines.append(f"KBV (P/B): {pb}{sector_note}")

    if ps is not None:
        lines.append(f"KUV (P/S): {ps}")
    if ev_ebit is not None:
        lines.append(f"EV/EBITDA: {ev_ebit}")

    lines.append("")
    lines.append("--- PROFITABILITÄT ---")

    profit_margin = funda.get("profit_margin")
    roe           = funda.get("roe")
    roa           = funda.get("roa")
    rev_growth    = funda.get("revenue_growth")
    earn_growth   = funda.get("earnings_growth")

    if profit_margin is not None:
        lines.append(f"Nettomarge: {profit_margin}%")
    if roe is not None:
        lines.append(f"Eigenkapitalrendite (ROE): {roe}%")
    if roa is not None:
        lines.append(f"Gesamtkapitalrendite (ROA): {roa}%")
    if rev_growth is not None:
        lines.append(f"Umsatzwachstum (YoY): {rev_growth}%")
    if earn_growth is not None:
        lines.append(f"Gewinnwachstum (YoY): {earn_growth}%")

    eps_trail = funda.get("eps_trailing")
    eps_fwd   = funda.get("eps_forward")
    if eps_trail is not None:
        lines.append(f"EPS (trailing): ${eps_trail}")
    if eps_fwd is not None:
        lines.append(f"EPS (forward): ${eps_fwd}")

    div_yield = funda.get("dividend_yield")
    if div_yield is not None:
        lines.append(f"Dividendenrendite: {div_yield}%")

    lines.append("")
    lines.append("--- BILANZSTÄRKE ---")

    d_to_e   = funda.get("debt_to_equity")
    curr_r   = funda.get("current_ratio")
    quick_r  = funda.get("quick_ratio")
    beta_val = funda.get("beta")

    if d_to_e is not None:
        health = "Hoch" if d_to_e > 100 else ("Moderat" if d_to_e > 50 else "Niedrig")
        lines.append(f"Verschuldungsgrad (D/E): {d_to_e} → {health}")
    if curr_r is not None:
        liq = "Gut" if curr_r >= 1.5 else ("Ausreichend" if curr_r >= 1.0 else "Kritisch")
        lines.append(f"Current Ratio: {curr_r} → {liq}")
    if quick_r is not None:
        lines.append(f"Quick Ratio: {quick_r}")
    if beta_val is not None:
        lines.append(f"Beta: {beta_val}")

    # Financial statements
    rev_q = financials.get("revenue_quarterly")
    if rev_q:
        formatted = [f"${_fmt_large(v)}" for v in rev_q]
        lines.append("")
        lines.append("--- QUARTALSDATEN ---")
        lines.append(f"Umsatz (letzte 4 Quartale, aktuell zuerst): {' | '.join(formatted)}")
        qoq = financials.get("revenue_qoq_growth_pct")
        if qoq is not None:
            lines.append(f"  QoQ Umsatzwachstum: {qoq:+.1f}%")

    ni_q = financials.get("net_income_quarterly")
    if ni_q:
        formatted = [f"${_fmt_large(v)}" for v in ni_q]
        lines.append(f"Nettogewinn (letzte 4Q): {' | '.join(formatted)}")

    fcf = financials.get("fcf_latest_quarter")
    if fcf is not None:
        lines.append(f"Free Cash Flow (letztes Quartal): ${_fmt_large(fcf)}")

    debt = financials.get("total_debt_latest")
    if debt is not None:
        lines.append(f"Gesamtverschuldung: ${_fmt_large(debt)}")

    # Analyst data
    lines.append("")
    lines.append("--- ANALYSTEN ---")

    target_mean = analyst.get("target_mean")
    target_high = analyst.get("target_high")
    target_low  = analyst.get("target_low")
    rec_key     = analyst.get("recommendation_key", "")
    n_analysts  = analyst.get("number_of_analyst_opinions")

    if target_mean:
        lines.append(f"Kursziel Mittelwert: ${target_mean} | Hoch: ${target_high} | Tief: ${target_low}")
    if rec_key:
        lines.append(f"Empfehlung: {rec_key.upper()}")
    if n_analysts:
        lines.append(f"Anzahl Analysten: {n_analysts}")

    strong_buy  = analyst.get("rec_strong_buy", 0)
    buy_cnt     = analyst.get("rec_buy", 0)
    hold_cnt    = analyst.get("rec_hold", 0)
    sell_cnt    = analyst.get("rec_sell", 0)
    s_sell      = analyst.get("rec_strong_sell", 0)
    if any([strong_buy, buy_cnt, hold_cnt, sell_cnt, s_sell]):
        lines.append(
            f"Empfehlungen: StrongBuy={strong_buy} Buy={buy_cnt} Hold={hold_cnt} "
            f"Sell={sell_cnt} StrongSell={s_sell}"
        )

    return "\n".join(lines)


def _fmt_large(val: Optional[float]) -> str:
    """Format large numbers: billions (B) / millions (M)."""
    if val is None:
        return "k.A."
    try:
        v = float(val)
        if abs(v) >= 1e12:
            return f"{v/1e12:.2f}T"
        if abs(v) >= 1e9:
            return f"{v/1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"{v/1e6:.1f}M"
        return f"{v:.0f}"
    except Exception:
        return str(val)
