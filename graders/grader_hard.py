"""
Civil Command Center — Grader: Hard Task
==========================================
Scores Task 3 (Era Advancement — 30 turns) from 0.0 to 1.0.

Scoring:
- 20% Survival through 30 turns
- 20% Crisis & threat management
- 25% Era advancement (target: Iron or beyond)
- 15% Population and civilization growth
- 10% Resource balance at end
- 10% Overall action quality
"""

from typing import Dict, Any


def grade_hard(episode_summary: Dict[str, Any]) -> float:
    score = 0.0

    collapse = episode_summary.get("collapse", False)
    pop_final = episode_summary.get("population_final", 0)
    pop_peak = episode_summary.get("population_peak", 120)
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
    ignored = episode_summary.get("messages_ignored", 0)
    progress = episode_summary.get("progress_score", 0)
    danger = episode_summary.get("danger_level", 0)
    turns = episode_summary.get("turns_played", 1)
    max_turns = episode_summary.get("max_turns", 30)

    # ── 20%: Survival ────────────────────────────────────────
    if not collapse:
        completion = turns / max_turns
        score += 0.20 * completion
    else:
        score += 0.20 * (turns / max_turns) * 0.25

    # ── 20%: Crisis & threat management ──────────────────────
    total_crises = crises_averted + crises_failed
    if total_crises > 0:
        crisis_ratio = crises_averted / total_crises
        score += 0.20 * crisis_ratio
    else:
        score += 0.20

    # ── 25%: Era advancement ─────────────────────────────────
    era_scores = {"tribal": 0.0, "bronze": 0.3, "iron": 0.7, "industrial": 0.9, "modern": 1.0}
    era_val = era_scores.get(era, 0.0)
    tech_val = min(1.0, tech / 7)  # target: tech 7+
    num_eras = len(set(eras)) - 1  # exclude starting tribal
    era_breadth = min(1.0, num_eras / 3)
    score += 0.25 * (0.4 * era_val + 0.3 * tech_val + 0.3 * era_breadth)

    # ── 15%: Population & growth ─────────────────────────────
    if not collapse and pop_final > 0:
        growth = max(0, pop_final - 120) / 120
        peak_retention = pop_final / max(pop_peak, 1)
        score += 0.15 * (0.5 * min(1.0, growth) + 0.5 * min(1.0, peak_retention))
    else:
        score += 0.15 * 0.1

    # ── 10%: Resource balance ────────────────────────────────
    food_s = min(1.0, food_final / 100) if food_final > 0 else 0.0
    energy_s = min(1.0, energy_final / 60) if energy_final > 0 else 0.0
    morale_s = min(1.0, morale_final / 50) if morale_final > 0 else 0.0
    danger_s = max(0, 1.0 - danger / 80)
    score += 0.10 * (food_s + energy_s + morale_s + danger_s) / 4

    # ── 10%: Action quality ──────────────────────────────────
    total_msgs = handled + ignored
    if total_msgs > 0:
        handle_ratio = handled / total_msgs
        correct_ratio = correct / max(handled, 1)
        score += 0.10 * (0.5 * handle_ratio + 0.5 * correct_ratio)
    else:
        score += 0.05

    return round(min(1.0, max(0.0, score)), 4)
