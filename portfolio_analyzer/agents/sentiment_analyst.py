"""
Sentiment / News Analyst Agent.

Analyzes news articles, insider transactions, and analyst consensus
data from Finnhub to assess market sentiment around a stock.
"""

from __future__ import annotations

from typing import Any, Dict, List


def analyze(ticker: str, data_bundle: Dict[str, Any], llm_client: Any) -> str:
    """
    Analyze sentiment, news, and insider activity for *ticker*.

    Args:
        ticker:      Stock symbol.
        data_bundle: Full data dict from the orchestrator.
        llm_client:  LLMClient instance.

    Returns:
        A string report on sentiment direction, news themes, and insider activity.
    """
    finnhub_data = data_bundle.get("finnhub", {})
    yf_data      = data_bundle.get("yfinance", {})
    meta         = yf_data.get("meta", {})
    language     = data_bundle.get("_language", "German")

    company_name = meta.get("name", ticker)
    sector       = meta.get("sector", "")

    if not finnhub_data.get("available", False):
        reason = finnhub_data.get("reason", "Finnhub-Daten nicht verfügbar.")
        return (
            f"## Sentiment-Analyse – {ticker}\n\n"
            f"**Hinweis:** {reason}\n\n"
            "Ohne Finnhub-Daten ist keine vollständige Sentiment-Analyse möglich. "
            "Bitte FINNHUB_API_KEY in der .env konfigurieren (kostenlos auf finnhub.io)."
        )

    sentiment_summary = _format_sentiment_data(ticker, company_name, finnhub_data)

    system_prompt = (
        "Du bist ein erfahrener Sentimentanalyst und Nachrichtenanalyst für Finanzmärkte. "
        "Du interpretierst Nachrichten, Insidertransaktionen und Analystenmeinungen, um "
        "die Stimmungslage und das Narrativ rund um eine Aktie zu verstehen. "
        f"Antworte ausschließlich auf {language}."
    ) if language == "German" else (
        "You are an experienced sentiment analyst and financial news analyst. "
        "You interpret news, insider transactions, and analyst opinions to understand "
        "the mood and narrative around a stock. "
        f"Always respond in {language}."
    )

    user_prompt = f"""Analysiere die folgenden Sentiment- und Nachrichtendaten für {ticker} ({company_name})
im Sektor: {sector}

{sentiment_summary}

Erstelle einen Sentiment-Analysebericht mit:

1. **Nachrichtenthemen**: Was sind die dominanten Themen in den aktuellen Nachrichten?
   Sind diese positiv, negativ oder neutral für die Aktie?

2. **Earnings Überraschungen**: Verlauf der Gewinnüberraschungen – schlägt das Unternehmen
   regelmäßig die Erwartungen? Was sagt das über die Qualität des Managements?

3. **Insideraktivität**: Interpretation der Insider-Transaktionen. Wer kauft/verkauft?
   Wie ist das im Kontext zu bewerten (Diversifikation vs. Überzeugung)?

4. **Analysten-Konsens**: Aktuelle Empfehlungsverteilung und Trends. Ändern sich Meinungen?

5. **Gesamtstimmung**: Ist das Sentiment rund um die Aktie aktuell:
   - Sehr bullisch / Bullisch / Neutral / Bearish / Sehr bearisch?
   - Begründung in 2-3 Sätzen

6. **Risiken aus Sentiment**: Welche Sentiment-Risiken (Überoptimismus, Short-Squeeze, etc.)
   sind zu berücksichtigen?

Sei konkret – zitiere spezifische Headlines oder Datenpunkte aus den bereitgestellten Daten.
""" if language == "German" else f"""Analyze the following sentiment and news data for {ticker} ({company_name})
in sector: {sector}

{sentiment_summary}

Create a sentiment analysis report covering:

1. **News Themes**: What are the dominant themes in current news?
   Are they positive, negative, or neutral for the stock?

2. **Earnings Surprises**: Pattern of earnings surprises – does the company consistently
   beat expectations? What does this say about management quality?

3. **Insider Activity**: Interpretation of insider transactions. Who is buying/selling?
   Context: diversification vs. conviction?

4. **Analyst Consensus**: Current recommendation distribution and trends. Are opinions changing?

5. **Overall Sentiment**: Is sentiment around the stock currently:
   - Very bullish / Bullish / Neutral / Bearish / Very bearish?
   - Reasoning in 2-3 sentences.

6. **Sentiment Risks**: What sentiment risks (over-optimism, short squeeze, etc.)
   should be considered?

Be specific – cite particular headlines or data points from the provided data.
"""

    return llm_client.complete(system_prompt, user_prompt, deep=False)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_sentiment_data(ticker: str, name: str, finnhub_data: dict) -> str:
    lines = []

    # News
    news = finnhub_data.get("news", [])
    if news:
        lines.append("=== AKTUELLE NACHRICHTEN (letzte 7 Tage) ===")
        for i, art in enumerate(news[:10], 1):
            headline = art.get("headline", "")
            source   = art.get("source", "")
            summary  = art.get("summary", "")
            lines.append(f"\n{i}. [{source}] {headline}")
            if summary:
                lines.append(f"   {summary[:250]}...")
    else:
        lines.append("Keine aktuellen Nachrichten verfügbar.")

    lines.append("")

    # Earnings surprises
    earnings = finnhub_data.get("earnings", [])
    if earnings:
        lines.append("=== EARNINGS ÜBERRASCHUNGEN (letzte 4 Quartale) ===")
        for q in earnings:
            period   = q.get("period", "")
            actual   = q.get("actual_eps")
            estimate = q.get("estimate_eps")
            surprise = q.get("surprise_pct")
            if actual is not None:
                surprise_str = f"{surprise:+.1f}%" if surprise is not None else "k.A."
                beat_miss = "BEAT" if (surprise or 0) > 0 else ("MISS" if (surprise or 0) < 0 else "IN-LINE")
                lines.append(
                    f"  {period}: Actual={actual} | Estimate={estimate} | "
                    f"Überraschung={surprise_str} [{beat_miss}]"
                )
    else:
        lines.append("Keine Earnings-Daten verfügbar.")

    lines.append("")

    # Insider transactions
    insiders = finnhub_data.get("insiders", {})
    if insiders:
        lines.append("=== INSIDERTRANSAKTIONEN (letzte 90 Tage) ===")
        sentiment_label = insiders.get("insider_sentiment", "")
        buy_cnt   = insiders.get("buy_count", 0)
        sell_cnt  = insiders.get("sell_count", 0)
        buy_shs   = insiders.get("total_buy_shares", 0)
        sell_shs  = insiders.get("total_sell_shares", 0)
        net_shs   = insiders.get("net_shares", 0)
        lines.append(f"  Gesamturteil: {sentiment_label}")
        lines.append(f"  Käufe: {buy_cnt} Transaktionen ({_fmt_num(buy_shs)} Aktien)")
        lines.append(f"  Verkäufe: {sell_cnt} Transaktionen ({_fmt_num(sell_shs)} Aktien)")
        lines.append(f"  Netto: {_fmt_num(net_shs)} Aktien")

        recent_tx = insiders.get("recent_transactions", [])
        if recent_tx:
            lines.append("\n  Jüngste Transaktionen:")
            for tx in recent_tx[:5]:
                name_tx = tx.get("name", "")
                title   = tx.get("title", "")
                tx_type = tx.get("transaction_type", "")
                shares  = tx.get("shares")
                price   = tx.get("price")
                tx_date = tx.get("date", "")
                tx_label = "KAUF" if tx_type in ("P", "A") else ("VERKAUF" if tx_type in ("S", "D") else tx_type)
                price_str = f"@ ${price}" if price else ""
                lines.append(
                    f"    {tx_date}: {name_tx} ({title}) – {tx_label} "
                    f"{_fmt_num(shares)} Aktien {price_str}"
                )
    else:
        lines.append("Keine Insider-Transaktionsdaten verfügbar.")

    lines.append("")

    # Analyst recommendation
    rec = finnhub_data.get("recommendation", {})
    if rec:
        lines.append("=== ANALYSTEN-EMPFEHLUNG ===")
        period     = rec.get("period", "")
        s_buy      = rec.get("strong_buy", 0)
        buy        = rec.get("buy", 0)
        hold       = rec.get("hold", 0)
        sell       = rec.get("sell", 0)
        s_sell     = rec.get("strong_sell", 0)
        total      = rec.get("total_analysts", 0)
        score      = rec.get("consensus_score")
        label      = rec.get("consensus_label", "")
        lines.append(f"  Zeitraum: {period} | Konsens: {label} (Score: {score}/5.0)")
        lines.append(f"  StrongBuy={s_buy} | Buy={buy} | Hold={hold} | Sell={sell} | StrongSell={s_sell}")
        lines.append(f"  Gesamte Analysten: {total}")
    else:
        lines.append("Keine Analysten-Empfehlungsdaten verfügbar.")

    # Selected Finnhub metrics
    metrics = finnhub_data.get("metrics", {})
    useful_metric_keys = [
        ("52WeekPriceReturnDaily", "52W Performance"),
        ("priceRelativeToS&P50013Week", "13W Perf. rel. zu S&P500"),
        ("priceRelativeToS&P500YTD", "YTD Perf. rel. zu S&P500"),
        ("rsi14D", "RSI(14) – Finnhub"),
        ("revenueGrowthTTMYoy", "Umsatzwachstum TTM YoY"),
        ("grossMarginTTM", "Bruttomarge TTM"),
        ("netProfitMarginTTM", "Nettomarge TTM"),
    ]
    relevant = [(label, metrics[key]) for key, label in useful_metric_keys if key in metrics and metrics[key] is not None]
    if relevant:
        lines.append("")
        lines.append("=== ZUSÄTZLICHE FINNHUB-METRIKEN ===")
        for label, val in relevant:
            lines.append(f"  {label}: {val}")

    return "\n".join(lines)


def _fmt_num(val) -> str:
    """Format a number with comma separators."""
    try:
        v = int(val)
        return f"{v:,}"
    except (TypeError, ValueError):
        return str(val) if val is not None else "k.A."
