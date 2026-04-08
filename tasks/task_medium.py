"""
Civil Command Center — Task 2: Growth
=======================================
Build a Thriving Settlement (20 Turns, Medium)

A mid-difficulty task requiring balance between survival
and expansion. The agent must grow population, advance
technology to Bronze Era, and handle trade opportunities.

Evaluation (via grader_medium):
  25% Survival & population growth
  25% Crisis management
  20% Technology / era advancement
  15% Resource balance
  15% Progress score
"""

TASK_CONFIG = {
    "id": "task_medium",
    "name": "Growth — Build a Thriving Settlement",
    "difficulty": "medium",
    "max_turns": 20,
    "description": (
        "Grow your settlement over 20 turns. Balance food, energy, and morale "
        "while handling threats, trades, and research opportunities. "
        "Reach Bronze Era and grow population past 150."
    ),
    "initial_state": {
        "population": 100,
        "food": 180,
        "energy": 80,
        "morale": 55,
        "technology_level": 1,
        "danger_level": 20,
        "available_workers": 25,
        "active_threats": 0,
    },
    "message_sources": ["citizen", "worker", "scientist", "trader", "defense", "event"],
    "threat_frequency": "medium",
    "trade_frequency": "medium",
    "research_frequency": "medium",
    "goals": [
        "Survive all 20 turns without collapse",
        "Grow population past 150",
        "Reach Bronze Era (tech level 3+)",
        "Maintain resource balance (food > 50, morale > 30)",
        "Successfully handle at least 60% of crises",
    ],
}
