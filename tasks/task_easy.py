"""
Civil Command Center — Task 1: Survival
=========================================
Keep the Village Alive (10 Turns, Easy)

The simplest task: a new settlement with basic needs.
The agent must handle citizen requests and maintain
food, morale, and population above collapse levels.

Evaluation (via grader_easy):
  40% Survival (population retention)
  30% Crisis handling
  20% Resource management
  10% Action efficiency
"""

TASK_CONFIG = {
    "id": "task_easy",
    "name": "Survival — Keep the Village Alive",
    "difficulty": "easy",
    "max_turns": 10,
    "description": (
        "Lead a small village for 10 turns. Handle messages from citizens "
        "and workers. Keep food, morale, and population above collapse levels. "
        "Simple decisions, few threats."
    ),
    "initial_state": {
        "population": 100,
        "food": 200,
        "energy": 100,
        "morale": 60,
        "technology_level": 1,
        "danger_level": 10,
        "available_workers": 30,
        "active_threats": 0,
    },
    "message_sources": ["citizen", "worker", "event"],
    "threat_frequency": "low",
    "trade_frequency": "none",
    "research_frequency": "low",
    "goals": [
        "Survive all 10 turns without collapse",
        "Keep population above 80",
        "Keep morale above 20",
        "Handle at least 50% of messages correctly",
    ],
}
