"""
Bull and Bear Researcher Agents.

These two functions simulate a structured debate between a bull and a bear,
each building on the previous round's opponent arguments to create a
dynamic, multi-round research dialogue.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def make_bull_case(
    ticker: str,
    all_analyst_reports: Dict[str, str],
    previous_bear_argument: Optional[str],
    llm_client: Any,
    language: str = "German",
) -> str:
    """
    Generate the bull (buy) case for *ticker*.

    Args:
        ticker:                 Stock symbol.
        all_analyst_reports:    Dict with keys "technical", "macro", "fundamental", "sentiment".
        previous_bear_argument: The bear's previous argument (None in round 1).
        llm_client:             LLMClient instance.
        language:               Output language.

    Returns:
        A string making the bull case for the stock.
    """
    reports_text = _format_reports(all_analyst_reports)
    round_label  = "Eröffnungsargument" if previous_bear_argument is None else "Rebuttal"

    bear_section = ""
    if previous_bear_argument:
        bear_section = f"""
=== ARGUMENT DES BÄREN (das du widerlegen oder relativieren musst) ===
{previous_bear_argument}
=== ENDE DES BÄREN-ARGUMENTS ===
"""

    system_prompt = (
        f"Du bist ein überzeugter Bulle (Long-Investor) und Research-Analyst. "
        f"Deine Aufgabe ist es, die stärkste mögliche Kaufthese für eine Aktie zu entwickeln. "
        f"Du nutzt alle verfügbaren Analysedaten, um eine überzeugende, faktenbasierte "
        f"Argumentation für eine Investition aufzubauen. "
        f"Du bist optimistisch, aber nicht naiv – du erkennst Risiken, relativierst sie aber. "
        f"Antworte ausschließlich auf {language}."
    ) if language == "German" else (
        f"You are a convinced bull (long investor) and research analyst. "
        f"Your task is to develop the strongest possible buy thesis for a stock. "
        f"You use all available analysis data to build a compelling, fact-based "
        f"investment argument. "
        f"You are optimistic but not naive – you acknowledge risks but put them in perspective. "
        f"Always respond in {language}."
    )

    user_prompt = f"""**{round_label}: BULLEN-CASE für {ticker}**

Hier sind die Berichte unserer Analysten:

{reports_text}
{bear_section}

{'Erstelle dein Eröffnungsargument' if not previous_bear_argument else 'Reagiere auf das Bären-Argument und stärke deine Position'} für {ticker}.

Dein Bullen-Case soll folgendes abdecken:

1. **Kernthese** (1-2 Sätze): Warum ist {ticker} eine starke Kaufgelegenheit?

2. **Stärkste Argumente** (mindestens 4):
   - Bewertungsargument: Ist die Aktie günstig bewertet?
   - Wachstumsargument: Was treibt zukünftiges Wachstum?
   - Qualitätsargument: Warum ist dieses Unternehmen besser als Wettbewerber?
   - Technisches/Momentum-Argument: Unterstützt der Chart die Kaufthese?
   - Makro-Katalysator: Welche wirtschaftlichen Faktoren begünstigen die Aktie?

3. **Kursziel & Zeithorizont**: Konservatives und optimistisches Kursziel mit Begründung

4. **Widerlegung der Risiken** (falls Runde 2+): Warum sind die Bären-Argumente übertrieben oder falsch?

5. **Fazit**: Klares Kaufvotum mit Überzeugungslevel (Hoch/Mittel/Niedrig)

Sei überzeugend, präzise und nutze konkrete Zahlen aus den Analyseberichten.
""" if language == "German" else f"""**{round_label}: BULL CASE for {ticker}**

Here are our analyst reports:

{reports_text}
{bear_section}

{'Create your opening argument' if not previous_bear_argument else 'Respond to the bear argument and strengthen your position'} for {ticker}.

Your bull case should cover:

1. **Core Thesis** (1-2 sentences): Why is {ticker} a strong buying opportunity?

2. **Strongest Arguments** (at least 4):
   - Valuation argument: Is the stock cheaply valued?
   - Growth argument: What drives future growth?
   - Quality argument: Why is this company better than competitors?
   - Technical/Momentum argument: Does the chart support the buy thesis?
   - Macro catalyst: What economic factors favor the stock?

3. **Price Target & Time Horizon**: Conservative and optimistic price targets with rationale

4. **Rebuttal of Risks** (round 2+): Why are bear arguments exaggerated or wrong?

5. **Conclusion**: Clear buy recommendation with conviction level (High/Medium/Low)

Be persuasive, precise, and use concrete numbers from the analysis reports.
"""

    return llm_client.complete(system_prompt, user_prompt, deep=True)


def make_bear_case(
    ticker: str,
    all_analyst_reports: Dict[str, str],
    previous_bull_argument: Optional[str],
    llm_client: Any,
    language: str = "German",
) -> str:
    """
    Generate the bear (sell/avoid) case for *ticker*.

    Args:
        ticker:                 Stock symbol.
        all_analyst_reports:    Dict with keys "technical", "macro", "fundamental", "sentiment".
        previous_bull_argument: The bull's previous argument (None in round 1).
        llm_client:             LLMClient instance.
        language:               Output language.

    Returns:
        A string making the bear case against the stock.
    """
    reports_text = _format_reports(all_analyst_reports)
    round_label  = "Eröffnungsargument" if previous_bull_argument is None else "Rebuttal"

    bull_section = ""
    if previous_bull_argument:
        bull_section = f"""
=== ARGUMENT DES BULLEN (das du widerlegen oder relativieren musst) ===
{previous_bull_argument}
=== ENDE DES BULLEN-ARGUMENTS ===
"""

    system_prompt = (
        f"Du bist ein kritischer Bär (Short-Seller / vorsichtiger Investor) und Research-Analyst. "
        f"Deine Aufgabe ist es, die stärksten möglichen Risiken und Gegenargumente für "
        f"eine Aktie zu identifizieren. Du suchst nach Überberechnungen, versteckten Risiken "
        f"und bearischen Signalen, die optimistische Investoren übersehen. "
        f"Du bist pessimistisch, aber sachlich und faktenbasiert. "
        f"Antworte ausschließlich auf {language}."
    ) if language == "German" else (
        f"You are a critical bear (short seller / cautious investor) and research analyst. "
        f"Your task is to identify the strongest possible risks and counter-arguments for "
        f"a stock. You look for overvaluations, hidden risks, and bearish signals that "
        f"optimistic investors miss. "
        f"You are pessimistic but objective and fact-based. "
        f"Always respond in {language}."
    )

    user_prompt = f"""**{round_label}: BÄREN-CASE für {ticker}**

Hier sind die Berichte unserer Analysten:

{reports_text}
{bull_section}

{'Erstelle dein Eröffnungsargument' if not previous_bull_argument else 'Reagiere auf das Bullen-Argument und stärke deine Position'} gegen {ticker}.

Dein Bären-Case soll folgendes abdecken:

1. **Kernthese** (1-2 Sätze): Warum ist {ticker} riskant oder zu meiden?

2. **Stärkste Argumente** (mindestens 4):
   - Bewertungsrisiko: Ist die Aktie teuer oder falsch bewertet?
   - Wachstumsrisiko: Warum könnten Wachstumserwartungen enttäuscht werden?
   - Fundamentale Schwäche: Qualitätsprobleme, Schulden, Margendruck
   - Technisches Warnsignal: Charttechnische Risiken, Überkauft, Trendbruch?
   - Makro-Gegenwind: Welche wirtschaftlichen Faktoren belasten die Aktie?
   - Sentiment-Risiko: Übertriebener Optimismus, Insiderverkäufe, etc.

3. **Kurszielrisiko**: Wohin könnte die Aktie fallen? Worst-Case-Szenario?

4. **Widerlegung der Bullen** (falls Runde 2+): Warum sind die Bullen-Argumente naiv oder falsch?

5. **Fazit**: Klares Verkaufs-/Meidungsvotum mit Überzeugungslevel (Hoch/Mittel/Niedrig)

Sei kritisch, präzise und nutze konkrete Zahlen aus den Analyseberichten.
""" if language == "German" else f"""**{round_label}: BEAR CASE for {ticker}**

Here are our analyst reports:

{reports_text}
{bull_section}

{'Create your opening argument' if not previous_bull_argument else 'Respond to the bull argument and strengthen your position'} against {ticker}.

Your bear case should cover:

1. **Core Thesis** (1-2 sentences): Why is {ticker} risky or to be avoided?

2. **Strongest Arguments** (at least 4):
   - Valuation risk: Is the stock expensive or mispriced?
   - Growth risk: Why might growth expectations disappoint?
   - Fundamental weakness: Quality issues, debt, margin pressure
   - Technical warning signal: Chart risks, overbought, trend break?
   - Macro headwind: What economic factors burden the stock?
   - Sentiment risk: Excessive optimism, insider selling, etc.

3. **Downside Target**: Where could the stock fall? Worst-case scenario?

4. **Rebuttal of Bulls** (round 2+): Why are bull arguments naive or wrong?

5. **Conclusion**: Clear sell/avoid recommendation with conviction level (High/Medium/Low)

Be critical, precise, and use concrete numbers from the analysis reports.
"""

    return llm_client.complete(system_prompt, user_prompt, deep=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_reports(reports: Dict[str, str]) -> str:
    """Format analyst reports into a readable text block."""
    sections = {
        "technical":    "TECHNISCHE ANALYSE",
        "macro":        "MAKRO-ANALYSE",
        "fundamental":  "FUNDAMENTALANALYSE",
        "sentiment":    "SENTIMENT-ANALYSE",
    }
    lines = []
    for key, title in sections.items():
        report = reports.get(key, "Nicht verfügbar.")
        lines.append(f"--- {title} ---")
        lines.append(report)
        lines.append("")
    return "\n".join(lines)
