"""
Civil Command Center — Grader: Medium Task
============================================
Scores Task 2 (Growth — 20 turns) from 0.0 to 1.0.

Scoring:
- 25% Survival & population growth
- 25% Crisis management
- 20% Technology / era advancement
- 15% Resource balance (food, energy, morale all healthy)
- 15% Progress score
"""

from typing import Dict, Any


def grade_medium(episode_summary: Dict[str, Any]) -> float:
    score = 0.0

    collapse = episode_summary.get("collapse", False)
    pop_final = episode_summary.get("population_final", 0)
    pop_peak = episode_summary.get("population_peak", 100)
    food_final = episode_summary.get("food_final", 0)
    energy_final = episode_summary.get("energy_final", 0)
    morale_final = episode_summary.get("morale_final", 0)
    tech = episode_summary.get("technology_level", 1)
    era = episode_summary.get("era_final", "tribal")
    eras = episode_summary.get("eras_reached", ["tribal"])
    crises_averted = episode_summary.get("crises_averted", 0)
    crises_failed = episode_summary.get("crises_failed", 0)
    correct = episode_summary.get("correct_actions", 0)
    handled = episode_summary.get("messages_handled", 0)
    progress = episode_summary.get("progress_score", 0)
    turns = episode_summary.get("turns_played", 1)
    max_turns = episode_summary.get("max_turns", 20)

    # ── 25%: Survival & population growth ────────────────────
    if not collapse:
        pop_growth = max(0, pop_final - 100) / 100  # target: grow by 50+
        retention = pop_final / max(pop_peak, 1)
        survival = 0.5 * min(1.0, retention) + 0.5 * min(1.0, pop_growth)
        score += 0.25 * survival
    else:
        score += 0.25 * (turns / max_turns) * 0.2

    # ── 25%: Crisis management ───────────────────────────────
    total_crises = crises_averted + crises_failed
    if total_crises > 0:
        score += 0.25 * (crises_averted / total_crises)
    else:
        score += 0.25

    # ── 20%: Technology / era ────────────────────────────────
    era_scores = {"tribal": 0.0, "bronze": 0.5, "iron": 0.8, "industrial": 0.95, "modern": 1.0}
    era_val = era_scores.get(era, 0.0)
    tech_val = min(1.0, tech / 5)  # target: reach tech 5
    score += 0.20 * (0.6 * era_val + 0.4 * tech_val)

    # ── 15%: Resource balance ────────────────────────────────
    food_s = min(1.0, food_final / 120)
    energy_s = min(1.0, energy_final / 80)
    morale_s = min(1.0, morale_final / 60)
    score += 0.15 * (food_s + energy_s + morale_s) / 3

    # ── 15%: Progress score ──────────────────────────────────
    score += 0.15 * min(1.0, progress / 80)

    return round(min(0.99, max(0.01, score)), 4)
