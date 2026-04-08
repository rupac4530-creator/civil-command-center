"""
Civil Command Center — Grader: Easy Task
==========================================
Scores Task 1 (Survival — 10 turns) from 0.0 to 1.0.

Scoring:
- 40% Survival (didn't collapse, population maintained)
- 30% Crisis handling (crises averted vs failed)
- 20% Resource management (food + morale above threshold)
- 10% Efficiency (correct actions ratio)
"""

from typing import Dict, Any


def grade_easy(episode_summary: Dict[str, Any]) -> float:
    score = 0.0

    collapse = episode_summary.get("collapse", False)
    pop_final = episode_summary.get("population_final", 0)
    pop_peak = episode_summary.get("population_peak", 100)
    food_final = episode_summary.get("food_final", 0)
    morale_final = episode_summary.get("morale_final", 0)
    crises_averted = episode_summary.get("crises_averted", 0)
    crises_failed = episode_summary.get("crises_failed", 0)
    correct = episode_summary.get("correct_actions", 0)
    handled = episode_summary.get("messages_handled", 0)
    turns = episode_summary.get("turns_played", 1)
    max_turns = episode_summary.get("max_turns", 10)

    # ── 40%: Survival ────────────────────────────────────────
    if not collapse:
        survival = 1.0
        # Bonus for population retention
        if pop_peak > 0:
            retention = pop_final / pop_peak
            survival = 0.6 + 0.4 * min(1.0, retention)
        score += 0.40 * survival
    else:
        # Partial credit for how long they survived
        score += 0.40 * (turns / max_turns) * 0.3

    # ── 30%: Crisis handling ─────────────────────────────────
    total_crises = crises_averted + crises_failed
    if total_crises > 0:
        crisis_score = crises_averted / total_crises
        score += 0.30 * crisis_score
    else:
        score += 0.30  # No crises = full credit

    # ── 20%: Resource management ─────────────────────────────
    food_score = min(1.0, food_final / 100) if food_final > 0 else 0.0
    morale_score = min(1.0, morale_final / 50) if morale_final > 0 else 0.0
    score += 0.20 * (0.5 * food_score + 0.5 * morale_score)

    # ── 10%: Action efficiency ───────────────────────────────
    if handled > 0:
        score += 0.10 * min(1.0, correct / handled)
    else:
        score += 0.05

    return round(min(0.99, max(0.01, score)), 4)
