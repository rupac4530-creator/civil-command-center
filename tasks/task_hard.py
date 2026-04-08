"""
Civil Command Center — Task 3: Era Advancement
================================================
Rise of a Civilization (30 Turns, Hard)

The hardest task: a hostile world with wars, disasters,
disease, and political unrest. The agent must survive
while advancing through technology eras.

Evaluation (via grader_hard):
  20% Survival through 30 turns
  20% Crisis & threat management
  25% Era advancement (target: Iron+)
  15% Population and civilization growth
  10% Resource balance at end
  10% Overall action quality
"""

TASK_CONFIG = {
    "id": "task_hard",
    "name": "Era Advancement — Rise of a Civilization",
    "difficulty": "hard",
    "max_turns": 30,
    "description": (
        "Guide your civilization through 30 turns of crises, wars, "
        "disasters, and opportunities. Reach the Iron Era or beyond. "
        "Survive invasions, manage disease, unlock technologies, "
        "and keep your people alive and hopeful."
    ),
    "initial_state": {
        "population": 120,
        "food": 150,
        "energy": 70,
        "morale": 45,
        "technology_level": 1,
        "danger_level": 35,
        "available_workers": 30,
        "active_threats": 1,
    },
    "message_sources": [
        "citizen", "worker", "scientist", "trader",
        "defense", "event", "advisor", "diplomat",
    ],
    "threat_frequency": "high",
    "trade_frequency": "high",
    "research_frequency": "high",
    "goals": [
        "Survive all 30 turns (or as long as possible)",
        "Reach Iron Era or beyond (tech level 5+)",
        "Keep population above 100",
        "Avert more crises than you fail",
        "Manage resource balance under constant pressure",
        "Demonstrate adaptive strategy over time",
    ],
}
