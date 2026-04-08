"""
Civil Command Center — Pydantic Models
========================================
Type-safe Action, Observation, State, and Message definitions
for the AI Civilization Leader environment built on OpenEnv.

The agent is the LEADER of a growing civilization.
It receives messages from citizens, scientists, workers, defense,
traders, and events — and must decide how to act.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════

class MessageSource(str, Enum):
    """Where the message comes from."""
    CITIZEN = "citizen"
    SCIENTIST = "scientist"
    WORKER = "worker"
    DEFENSE = "defense"
    TRADER = "trader"
    EVENT = "event"
    ADVISOR = "advisor"
    DIPLOMAT = "diplomat"


class MessageUrgency(str, Enum):
    """How urgent the message is."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionType(str, Enum):
    """Actions the leader can take."""
    ALLOCATE_FOOD = "allocate_food"
    ALLOCATE_WORKERS = "allocate_workers"
    APPROVE_RESEARCH = "approve_research"
    DEFEND = "defend"
    CALM_CITIZENS = "calm_citizens"
    ACCEPT_TRADE = "accept_trade"
    REJECT_TRADE = "reject_trade"
    INVEST_GROWTH = "invest_growth"
    EMERGENCY_RESPONSE = "emergency_response"
    IGNORE = "ignore"


class Era(str, Enum):
    """Civilization eras — unlocked through tech progression."""
    TRIBAL = "tribal"          # Era 1: tech_level 0-2
    BRONZE = "bronze"          # Era 2: tech_level 3-4
    IRON = "iron"              # Era 3: tech_level 5-6
    INDUSTRIAL = "industrial"  # Era 4: tech_level 7-8
    MODERN = "modern"          # Era 5: tech_level 9-10


# ═══════════════════════════════════════════════════════════════
# Message Model
# ═══════════════════════════════════════════════════════════════

class Message(BaseModel):
    """A message received by the leader from a civilization source."""
    id: str
    source: str = Field(..., description="Who sent the message: citizen, scientist, worker, defense, trader, event, advisor, diplomat")
    sender_name: str = Field(..., description="Name of the sender")
    subject: str = Field(..., description="Short headline of the message")
    body: str = Field(..., description="Full message content with context and requests")
    urgency: str = Field(default="medium", description="critical, high, medium, low")
    turn_received: int = Field(default=0, description="Turn when this message arrived")

    # Hidden ground truth (not shown to agent)
    best_action: str = Field(default="ignore", description="The ideal action for this message")
    consequences: Dict[str, float] = Field(
        default_factory=dict,
        description="State changes if handled correctly: {population: +5, morale: +10, ...}"
    )
    ignore_penalty: Dict[str, float] = Field(
        default_factory=dict,
        description="State changes if ignored: {morale: -15, danger: +20, ...}"
    )
    decay_turns: int = Field(default=3, description="Turns before consequences auto-apply if ignored")


# ═══════════════════════════════════════════════════════════════
# Action Model
# ═══════════════════════════════════════════════════════════════

class CivAction(BaseModel):
    """Action the leader takes each turn."""
    action_type: str = Field(
        ...,
        description=(
            "One of: allocate_food, allocate_workers, approve_research, "
            "defend, calm_citizens, accept_trade, reject_trade, "
            "invest_growth, emergency_response, ignore"
        ),
    )
    target_message_id: Optional[str] = Field(
        None,
        description="ID of the message this action addresses (optional but recommended)"
    )
    reason: Optional[str] = Field(
        None,
        description="Brief reasoning for the decision (used for quality scoring)"
    )


# ═══════════════════════════════════════════════════════════════
# Observation Model
# ═══════════════════════════════════════════════════════════════

class CivObservation(BaseModel):
    """What the leader sees each turn."""
    done: bool = False
    reward: Optional[float] = None

    # Current turn info
    turn: int = 0
    max_turns: int = 0
    era: str = "tribal"

    # Civilization state (visible to agent)
    population: int = 100
    food: int = 200
    energy: int = 100
    morale: int = 60
    technology_level: int = 1
    danger_level: int = 10
    available_workers: int = 30
    active_threats: int = 0
    progress_score: float = 0.0

    # Messages for this turn
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    num_messages: int = 0

    # Feedback from last action
    message: str = ""
    last_action_effective: Optional[bool] = None

    # Task info
    task_name: str = ""
    task_description: str = ""

    # Summary stats
    total_reward: float = 0.0

    # Memory & personality (visible to agent)
    personality: str = "balanced"
    personality_strength: float = 0.0
    citizen_trust: float = 50.0
    military_trust: float = 50.0
    narrative_summary: str = ""
    pending_consequences: int = 0
    stability_score: float = 0.65
    stability_trend: str = "stable"
    predictions: List[str] = Field(default_factory=list)
    active_chains: List[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# State Model (full internal state)
# ═══════════════════════════════════════════════════════════════

class CivState(BaseModel):
    """Full internal state of the civilization."""
    episode_id: Optional[str] = None
    turn: int = 0

    # Core resources
    population: int = 100
    food: int = 200
    energy: int = 100
    morale: int = 60
    technology_level: int = 1
    danger_level: int = 10
    available_workers: int = 30
    active_threats: int = 0
    progress_score: float = 0.0
    era: str = "tribal"

    # Tracking
    total_reward: float = 0.0
    messages_handled: int = 0
    messages_ignored: int = 0
    correct_actions: int = 0
    crises_averted: int = 0
    crises_failed: int = 0
    tech_unlocks: int = 0
    eras_reached: List[str] = Field(default_factory=lambda: ["tribal"])
    population_peak: int = 100
    collapse: bool = False
    collapse_reason: str = ""

    # Pending messages that haven't been resolved
    pending_messages: List[Dict[str, Any]] = Field(default_factory=list)

    # Task config
    task_id: str = ""
    max_turns: int = 20
    difficulty: str = "easy"
