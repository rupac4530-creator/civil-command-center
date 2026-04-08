"""
Civil Command Center — Civilization Memory System v3 (Final Polish)
=====================================================================
Winning-level polish: importance-scored memory, confidence-rated
predictions, smoothed stability, diminishing-return trust, adaptive
personality, escalating consequences, severity-scaled chains,
milestone rewards, urgency-decayed messages, cached model routing.
"""

import os
import math
import random
import hashlib
import time
import requests
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter


# ═══════════════════════════════════════════════════════════════
# Free-model routing with reliability scoring & caching
# ═══════════════════════════════════════════════════════════════

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/api")

MODEL_ROUTES = {
    "chronicle":  os.getenv("OLLAMA_CHRONICLE_MODEL", os.getenv("OLLAMA_MODEL", "gemma3")),
    "reasoning":  os.getenv("OLLAMA_REASONING_MODEL", os.getenv("OLLAMA_MODEL", "gemma3")),
    "summary":    os.getenv("OLLAMA_SUMMARY_MODEL",   os.getenv("OLLAMA_MODEL", "gemma3")),
}

# Model reliability tracking
_model_stats = {role: {"success": 0, "fail": 0, "avg_ms": 500} for role in MODEL_ROUTES}
_response_cache: Dict[str, str] = {}
_CACHE_MAX = 30


def _call_ollama(prompt: str, role: str = "chronicle") -> Optional[str]:
    """Call Ollama with reliability scoring, caching, and auto-fallback."""
    if not OLLAMA_API_KEY:
        return None

    # Check cache first
    cache_key = hashlib.md5((prompt[:200] + role).encode()).hexdigest()
    if cache_key in _response_cache:
        return _response_cache[cache_key]

    model = MODEL_ROUTES.get(role, MODEL_ROUTES["chronicle"])
    stats = _model_stats.get(role, {"success": 0, "fail": 0, "avg_ms": 500})

    # Skip if model has >60% failure rate (after 3+ attempts)
    total = stats["success"] + stats["fail"]
    if total >= 3 and stats["fail"] / total > 0.6:
        return None

    # Adaptive timeout based on past average
    timeout = min(12, max(5, stats["avg_ms"] / 1000 * 2.5))

    try:
        t0 = time.time()
        r = requests.post(
            f"{OLLAMA_BASE_URL}/generate",
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        elapsed_ms = (time.time() - t0) * 1000

        if r.ok:
            text = r.json().get("response", "").strip()
            if text:
                stats["success"] += 1
                stats["avg_ms"] = stats["avg_ms"] * 0.7 + elapsed_ms * 0.3
                # Cache result
                if len(_response_cache) >= _CACHE_MAX:
                    _response_cache.pop(next(iter(_response_cache)))
                _response_cache[cache_key] = text
                return text

        stats["fail"] += 1
    except Exception:
        stats["fail"] += 1

    return None


# ═══════════════════════════════════════════════════════════════
# Leadership Style
# ═══════════════════════════════════════════════════════════════

class LeadershipStyle:
    MILITARY = "military"
    SCIENTIFIC = "scientific"
    DIPLOMATIC = "diplomatic"
    ECONOMIC = "economic"
    BALANCED = "balanced"
    NEGLECTFUL = "neglectful"

    ACTION_STYLES = {
        "defend": MILITARY, "emergency_response": MILITARY,
        "approve_research": SCIENTIFIC,
        "invest_growth": ECONOMIC, "allocate_workers": ECONOMIC,
        "allocate_food": DIPLOMATIC, "calm_citizens": DIPLOMATIC,
        "accept_trade": DIPLOMATIC, "reject_trade": MILITARY,
        "ignore": NEGLECTFUL,
    }


# ═══════════════════════════════════════════════════════════════
# Event Chain Definitions (with severity scaling)
# ═══════════════════════════════════════════════════════════════

EVENT_CHAINS = {
    "drought": [
        {"delay": 0, "effects": {"food": -12},  "reason": "☀️ A drought begins drying the fields.", "type": "negative", "severity": 0.3},
        {"delay": 2, "effects": {"food": -18, "morale": -6}, "reason": "🌾 The drought worsens — crops fail.", "type": "negative", "severity": 0.6},
        {"delay": 4, "effects": {"morale": -10, "population": -3, "danger_level": 4}, "reason": "😡 Famine drives citizens to desperate protest.", "type": "negative", "severity": 0.9},
    ],
    "golden_age": [
        {"delay": 0, "effects": {"morale": 8},  "reason": "🌟 A period of prosperity begins.", "type": "positive", "severity": 0.3},
        {"delay": 2, "effects": {"population": 4, "food": 12}, "reason": "🎉 Settlers arrive, drawn by your stability.", "type": "positive", "severity": 0.6},
        {"delay": 4, "effects": {"progress_score": 8, "technology_level": 0.2}, "reason": "💡 Cultural flourishing sparks innovation.", "type": "positive", "severity": 0.9},
    ],
    "plague": [
        {"delay": 0, "effects": {"population": -4, "morale": -4}, "reason": "🦠 A sickness spreads through the settlement.", "type": "negative", "severity": 0.4},
        {"delay": 2, "effects": {"population": -6, "available_workers": -2}, "reason": "💀 The plague claims more lives.", "type": "negative", "severity": 0.7},
        {"delay": 3, "effects": {"morale": -8, "danger_level": 4}, "reason": "😰 Despair grips the populace.", "type": "negative", "severity": 1.0},
    ],
    "tech_boom": [
        {"delay": 0, "effects": {"technology_level": 0.25}, "reason": "🔬 A breakthrough in the workshops!", "type": "positive", "severity": 0.3},
        {"delay": 2, "effects": {"energy": 15, "progress_score": 6}, "reason": "⚡ New techniques improve production.", "type": "positive", "severity": 0.6},
        {"delay": 3, "effects": {"technology_level": 0.25, "morale": 4}, "reason": "🚀 The breakthrough cascades into new innovations.", "type": "positive", "severity": 0.9},
    ],
    "border_war": [
        {"delay": 0, "effects": {"danger_level": 10}, "reason": "⚔️ Hostile forces gather on the frontier.", "type": "negative", "severity": 0.3},
        {"delay": 2, "effects": {"danger_level": 8, "population": -3, "energy": -8}, "reason": "🔥 Raids begin. Casualties mount.", "type": "negative", "severity": 0.7},
        {"delay": 4, "effects": {"danger_level": 6, "morale": -8, "food": -12}, "reason": "💣 Border conflict drains all reserves.", "type": "negative", "severity": 1.0},
    ],
}

CHAIN_MITIGATIONS = {
    "drought":    ["allocate_food", "invest_growth"],
    "plague":     ["emergency_response", "approve_research"],
    "border_war": ["defend", "emergency_response"],
}


# ═══════════════════════════════════════════════════════════════
# Civilization Memory v3 — Final Polish
# ═══════════════════════════════════════════════════════════════

class CivilizationMemory:
    """
    Final-polish memory with:
    - Importance-scored condensed memory
    - Confidence-rated predictions  
    - Smoothed stability with weighted components
    - Diminishing-return trust with betrayal memory
    - Adaptive personality thresholds
    - Escalating + decaying delayed consequences
    - Severity-scaled event chains with soft warnings
    - Cached model routing with reliability scoring
    """

    MAX_DELAYED_EFFECTS = 6
    PERSONALITY_SHIFT_THRESHOLD = 6  # Increased from 5 → requires more sustained behavior

    def __init__(self):
        self.reset()

    def reset(self):
        # ── Importance-scored Memory ──────────────────────────
        self.action_history: List[str] = []
        self.important_events: List[Dict[str, Any]] = []
        self.style_counts: Counter = Counter()
        self.ignored_sources: Counter = Counter()
        self.helped_sources: Counter = Counter()
        self._seen_event_hashes: set = set()  # Dedup

        # ── Trust with betrayal memory ────────────────────────
        self.citizen_trust: float = 50.0
        self.military_trust: float = 50.0
        self.scientific_trust: float = 50.0
        self.worker_trust: float = 50.0
        self.fear_level: float = 0.0
        self._trust_betrayal_cooldown: Dict[str, int] = {}  # source→turns remaining

        # ── Behavioral Patterns ───────────────────────────────
        self.consecutive_ignores: int = 0
        self.consecutive_same: int = 0
        self.last_action: str = ""
        self.crises_streak: int = 0
        self.neglect_streak: int = 0

        # ── Delayed Effects (escalating + decaying) ───────────
        self.delayed_effects: List[Dict[str, Any]] = []

        # ── Event Chains (severity-scaled) ────────────────────
        self.active_chains: List[Dict[str, Any]] = []

        # ── Adaptive Personality ──────────────────────────────
        self.personality: str = LeadershipStyle.BALANCED
        self.personality_strength: float = 0.0
        self._personality_lock_turns: int = 0
        self._personality_momentum: float = 0.0  # Smooth transitions

        # ── Smoothed Stability ────────────────────────────────
        self.stability_score: float = 0.65
        self._stability_history: List[float] = [0.65]  # Rolling window

        # ── Confidence-rated Predictions ──────────────────────
        self.predictions: List[str] = []

        # ── Narrative ─────────────────────────────────────────
        self.narrative_summary: str = "A young civilization begins its journey."
        self.key_moments: List[str] = []

        # ── Performance cache ─────────────────────────────────
        self._last_state_hash: str = ""
        self._cached_personality_fx: Dict[str, float] = {}

    # ═══════════════════════════════════════════════════════════
    # Record Action (with importance scoring + dedup)
    # ═══════════════════════════════════════════════════════════

    def record_action(
        self, turn: int, action: str, source: str,
        urgency: str, effective: bool, reward: float,
        civ_state: Dict[str, Any],
    ):
        self.action_history.append(action)

        # Style counting
        style = LeadershipStyle.ACTION_STYLES.get(action, LeadershipStyle.BALANCED)
        self.style_counts[style] += 1

        # Trust update (with diminishing returns + betrayal cooldown)
        self._update_trust_refined(action, source, urgency, effective, turn)

        # Consecutive tracking
        if action == self.last_action:
            self.consecutive_same += 1
        else:
            self.consecutive_same = 0
        self.last_action = action

        if action == "ignore":
            self.consecutive_ignores += 1
            self.ignored_sources[source] += 1
            if urgency in ("critical", "high"):
                self.neglect_streak += 1
        else:
            self.consecutive_ignores = 0
            self.helped_sources[source] += 1
            if urgency in ("critical", "high") and effective:
                self.crises_streak += 1
                self.neglect_streak = 0

        # ── Importance-scored memory entry ────────────────────
        importance = self._score_importance(action, source, urgency, effective, reward)
        if importance >= 0.3:  # Only store meaningful events
            event_hash = f"{turn}_{action}_{source}"
            if event_hash not in self._seen_event_hashes:
                self._seen_event_hashes.add(event_hash)
                self.important_events.append({
                    "turn": turn, "action": action, "source": source,
                    "urgency": urgency, "effective": effective,
                    "reward": round(reward, 3), "importance": round(importance, 2),
                })
                # Cap at 12, keep highest importance
                if len(self.important_events) > 12:
                    self.important_events.sort(key=lambda e: e["importance"], reverse=True)
                    self.important_events = self.important_events[:12]

                # Narrative moments
                if urgency == "critical":
                    verb = "handled" if effective else "neglected"
                    self.key_moments.append(f"Turn {turn}: Leader {verb} a {source} crisis.")
                    if len(self.key_moments) > 6:
                        self.key_moments = self.key_moments[-6:]

        # Auto-summarize memory every 5 turns
        if turn > 0 and turn % 5 == 0:
            self._auto_summarize_memory(civ_state)

        # Decay minor delayed effects
        self._decay_minor_effects()

        # Schedule consequences + chains
        self._schedule_delayed_effects(turn, action, source, urgency, effective, civ_state)
        self._check_event_chain_triggers(turn, action, civ_state)
        self._check_chain_mitigation(action)

        # Update systems
        self._update_personality_adaptive(turn)
        self._update_stability_smooth(civ_state)
        self._generate_predictions_confident(turn, civ_state)

        # Narrative on important turns
        if importance >= 0.5 and turn % 3 == 0:
            self.generate_narrative_summary(civ_state)

    def _score_importance(self, action, source, urgency, effective, reward) -> float:
        """Score 0-1 how important this event is for memory."""
        score = 0.0
        if urgency == "critical":
            score += 0.5
        elif urgency == "high":
            score += 0.3
        elif urgency == "medium":
            score += 0.1

        if abs(reward) > 0.2:
            score += min(0.3, abs(reward))

        if not effective and urgency in ("critical", "high"):
            score += 0.2  # Failures on important events are very memorable

        if self.consecutive_ignores >= 3:
            score += 0.15

        return min(1.0, score)

    def _auto_summarize_memory(self, civ_state: Dict):
        """Every 5 turns, consolidate old memory entries."""
        if len(self.important_events) <= 6:
            return
        # Keep recent 6 + top 3 by importance from older events
        recent = self.important_events[-6:]
        older = self.important_events[:-6]
        older.sort(key=lambda e: e["importance"], reverse=True)
        top_old = older[:3]
        self.important_events = top_old + recent

    # ═══════════════════════════════════════════════════════════
    # Trust with Diminishing Returns + Betrayal Memory
    # ═══════════════════════════════════════════════════════════

    def _update_trust_refined(self, action, source, urgency, effective, turn):
        trust_map = {
            "citizen": "citizen_trust", "worker": "worker_trust",
            "scientist": "scientific_trust", "defense": "military_trust",
            "advisor": "citizen_trust", "diplomat": "citizen_trust",
            "trader": "citizen_trust", "event": "citizen_trust",
        }
        attr = trust_map.get(source, "citizen_trust")
        current = getattr(self, attr)

        # Check betrayal cooldown (trust recovers slowly after major betrayal)
        in_cooldown = self._trust_betrayal_cooldown.get(source, 0) > 0

        if action == "ignore":
            # Scaled penalty with inertia
            base_penalty = -4 if urgency in ("critical", "high") else -1.5
            inertia = max(0.4, current / 100.0)
            change = base_penalty * inertia

            # Major betrayal: ignoring critical triggers cooldown
            if urgency == "critical":
                self._trust_betrayal_cooldown[source] = 5  # 5-turn slow recovery

            setattr(self, attr, max(0, current + change))
            self.citizen_trust = max(0, self.citizen_trust - 0.8)

        elif effective:
            # Diminishing returns: harder to gain trust when already high
            base_gain = 5 if urgency in ("critical", "high") else 2.5
            diminishing = max(0.2, 1.0 - (current / 100.0) ** 1.5)

            # If in betrayal cooldown, trust recovery is halved
            if in_cooldown:
                diminishing *= 0.5

            change = base_gain * diminishing
            setattr(self, attr, min(100, current + change))
            self.citizen_trust = min(100, self.citizen_trust + 0.5)
        else:
            # Mild penalty for ineffective action
            setattr(self, attr, max(0, current - 1.0))

        # Tick down betrayal cooldowns
        for s in list(self._trust_betrayal_cooldown):
            if self._trust_betrayal_cooldown[s] > 0:
                self._trust_betrayal_cooldown[s] -= 1
            else:
                del self._trust_betrayal_cooldown[s]

        # Fear: only from truly aggressive actions, slow buildup
        if action in ("defend", "emergency_response"):
            self.fear_level = min(100, self.fear_level + 1.2)
        elif action == "reject_trade":
            self.fear_level = min(100, self.fear_level + 1.8)
        elif action in ("calm_citizens", "allocate_food", "accept_trade"):
            self.fear_level = max(0, self.fear_level - 1.2)

    # ═══════════════════════════════════════════════════════════
    # Delayed Consequences (escalating + decaying)
    # ═══════════════════════════════════════════════════════════

    def _schedule_delayed_effects(self, turn, action, source, urgency, effective, civ_state):
        if len(self.delayed_effects) >= self.MAX_DELAYED_EFFECTS:
            return

        pop = civ_state.get("population", 100)
        food = civ_state.get("food", 200)

        # Critical ignore → escalating instability
        if action == "ignore" and urgency == "critical" and self.neglect_streak % 2 == 0:
            escalation = min(1.5, 1.0 + self.neglect_streak * 0.1)
            self.delayed_effects.append({
                "turn_trigger": turn + 3,
                "effects": {"morale": int(-7 * escalation), "danger_level": int(5 * escalation)},
                "reason": f"The {source}s' unanswered crisis from turn {turn} festers into unrest.",
                "type": "negative", "_severity": 0.7,
            })

        # Defense streak reward
        if action == "defend" and self.crises_streak >= 3 and self.crises_streak % 3 == 0:
            self.delayed_effects.append({
                "turn_trigger": turn + 3,
                "effects": {"danger_level": -7, "morale": 3},
                "reason": "Your defense record deters future aggressors.",
                "type": "positive", "_severity": 0.5,
            })

        # Sustained research reward
        sci = self.style_counts.get(LeadershipStyle.SCIENTIFIC, 0)
        if action == "approve_research" and sci >= 4 and sci % 4 == 0:
            self.delayed_effects.append({
                "turn_trigger": turn + 2,
                "effects": {"technology_level": 0.35, "progress_score": 5},
                "reason": "Sustained research yields an unexpected discovery!",
                "type": "positive", "_severity": 0.6,
            })

        # Revolt from consecutive ignores (fires once at exactly 3)
        if self.consecutive_ignores == 3:
            self.delayed_effects.append({
                "turn_trigger": turn + 2,
                "effects": {"morale": -10, "danger_level": 7},
                "reason": "Neglect breeds discontent. Whispers of revolt spread.",
                "type": "negative", "_severity": 0.8,
            })

        # Trade network payoff
        trade_count = self.helped_sources.get("trader", 0)
        if action == "accept_trade" and trade_count >= 3 and trade_count % 3 == 0:
            self.delayed_effects.append({
                "turn_trigger": turn + 3,
                "effects": {"food": 18, "energy": 10},
                "reason": "Merchant caravans bring gifts — your trade reputation pays off.",
                "type": "positive", "_severity": 0.4,
            })

        # Population pressure
        if pop > 135 and food < 65 and not any(e.get("_tag") == "famine" for e in self.delayed_effects):
            self.delayed_effects.append({
                "turn_trigger": turn + 2,
                "effects": {"food": -12, "morale": -5, "population": -2},
                "reason": "Population outgrows food supply. Rationing begins.",
                "type": "negative", "_severity": 0.6, "_tag": "famine",
            })

    def _decay_minor_effects(self):
        """Remove low-severity delayed effects that are no longer relevant."""
        if len(self.delayed_effects) <= 3:
            return
        # Only decay if we're near the cap
        self.delayed_effects = [
            e for e in self.delayed_effects
            if e.get("_severity", 0.5) >= 0.3 or e["type"] == "positive"
        ]

    def process_delayed_effects(self, current_turn: int) -> List[Dict[str, Any]]:
        """Process effects due this turn. Avoids simultaneous triggers."""
        triggered = []
        remaining = []
        for e in self.delayed_effects:
            if current_turn >= e["turn_trigger"]:
                triggered.append(e)
            else:
                remaining.append(e)
        self.delayed_effects = remaining

        # Merge related effects (same type firing same turn)
        if len(triggered) > 2:
            # Keep max 2 negative + all positive to avoid spikes
            neg = [t for t in triggered if t["type"] == "negative"]
            pos = [t for t in triggered if t["type"] == "positive"]
            neg.sort(key=lambda e: e.get("_severity", 0.5), reverse=True)
            triggered = neg[:2] + pos

        return triggered

    # ═══════════════════════════════════════════════════════════
    # Event Chains (severity-scaled, soft-warnings, recovery)
    # ═══════════════════════════════════════════════════════════

    def _check_event_chain_triggers(self, turn, action, civ_state):
        food = civ_state.get("food", 200)
        danger = civ_state.get("danger_level", 10)
        tech = civ_state.get("technology_level", 1)
        morale = civ_state.get("morale", 60)
        pop = civ_state.get("population", 100)
        active_ids = [c["chain_id"] for c in self.active_chains]

        # Limit total active chains to 2 to avoid chaos
        if len(self.active_chains) >= 2:
            return

        if food < 55 and "drought" not in active_ids and random.random() < 0.25:
            self._start_chain("drought", turn)
        if danger > 55 and "border_war" not in active_ids and random.random() < 0.2:
            self._start_chain("border_war", turn)
        if pop > 125 and "plague" not in active_ids and random.random() < 0.08:
            self._start_chain("plague", turn)
        if tech >= 4 and "tech_boom" not in active_ids and random.random() < 0.15:
            self._start_chain("tech_boom", turn)
        if morale > 70 and self.stability_score > 0.72 and "golden_age" not in active_ids and random.random() < 0.12:
            self._start_chain("golden_age", turn)

    def _start_chain(self, chain_id, turn):
        chain_def = EVENT_CHAINS.get(chain_id)
        if not chain_def:
            return

        self.active_chains.append({
            "chain_id": chain_id, "start_turn": turn, "mitigated": False,
        })

        for step in chain_def:
            if len(self.delayed_effects) >= self.MAX_DELAYED_EFFECTS + 3:
                break
            self.delayed_effects.append({
                "turn_trigger": turn + step["delay"],
                "effects": dict(step["effects"]),
                "reason": step["reason"],
                "type": step["type"],
                "_chain": chain_id,
                "_severity": step.get("severity", 0.5),
            })

    def _check_chain_mitigation(self, action):
        for chain in self.active_chains:
            if chain["mitigated"]:
                continue
            mitigators = CHAIN_MITIGATIONS.get(chain["chain_id"], [])
            if action in mitigators:
                chain["mitigated"] = True
                for e in self.delayed_effects:
                    if e.get("_chain") == chain["chain_id"] and e["type"] == "negative":
                        for k in e["effects"]:
                            e["effects"][k] = int(e["effects"][k] * 0.4)  # 60% reduction

    def cleanup_chains(self, current_turn):
        self.active_chains = [
            c for c in self.active_chains
            if any(e.get("_chain") == c["chain_id"] for e in self.delayed_effects)
        ]

    # ═══════════════════════════════════════════════════════════
    # Adaptive Personality (smooth, momentum-based)
    # ═══════════════════════════════════════════════════════════

    def _update_personality_adaptive(self, turn):
        total = sum(self.style_counts.values())
        if total < self.PERSONALITY_SHIFT_THRESHOLD:
            return

        self._personality_lock_turns += 1
        dominant = self.style_counts.most_common(1)[0]
        dominant_style, dominant_count = dominant
        dominant_ratio = dominant_count / total

        # Recent behavior weighted higher (last 5 actions)
        recent = self.action_history[-5:] if len(self.action_history) >= 5 else self.action_history
        recent_styles = Counter(LeadershipStyle.ACTION_STYLES.get(a, LeadershipStyle.BALANCED) for a in recent)
        recent_dominant = recent_styles.most_common(1)[0][0] if recent_styles else LeadershipStyle.BALANCED

        # Require 40%+ dominance AND at least 3 turns since last shift AND recent alignment
        if (dominant_ratio >= 0.4
                and self._personality_lock_turns >= 3
                and recent_dominant == dominant_style):
            if dominant_style != self.personality:
                self._personality_lock_turns = 0
            self.personality = dominant_style
            # Momentum-based strength (smooth transition)
            target = min(0.85, dominant_ratio * 1.2)
            self.personality_strength = self.personality_strength * 0.6 + target * 0.4
        elif dominant_ratio < 0.3:
            self.personality = LeadershipStyle.BALANCED
            self.personality_strength = max(0.0, self.personality_strength - 0.03)

    def get_personality_effects(self) -> Dict[str, float]:
        if self.personality_strength < 0.3:
            return {}
        # Use cached result if state hasn't changed significantly
        s = self.personality_strength
        effects = {
            LeadershipStyle.MILITARY:   {"danger_level": -2 * s, "energy": -1 * s},
            LeadershipStyle.SCIENTIFIC: {"technology_level": 0.07 * s, "progress_score": 1.2 * s},
            LeadershipStyle.DIPLOMATIC: {"morale": 1.2 * s, "population": 0.4 * s},
            LeadershipStyle.ECONOMIC:   {"food": 1.8 * s, "energy": 1.2 * s},
            LeadershipStyle.NEGLECTFUL: {"morale": -1.8 * s, "danger_level": 1.2 * s},
        }
        return effects.get(self.personality, {})

    # ═══════════════════════════════════════════════════════════
    # Smoothed Stability Score (rolling average, clamped)
    # ═══════════════════════════════════════════════════════════

    def _update_stability_smooth(self, civ_state):
        morale = civ_state.get("morale", 60)
        danger = civ_state.get("danger_level", 10)
        food = civ_state.get("food", 200)

        avg_trust = (self.citizen_trust + self.military_trust +
                     self.scientific_trust + self.worker_trust) / 4.0

        # Weighted components (trust > morale > fear > danger)
        trust_score = avg_trust / 100.0
        morale_score = min(1.0, morale / 75.0)
        safety_score = max(0.0, 1.0 - danger / 75.0)
        food_score = min(1.0, food / 120.0)
        fear_penalty = max(0.0, (self.fear_level - 35) / 80.0)

        raw = (
            trust_score * 0.30 +    # Trust is most important
            morale_score * 0.25 +
            safety_score * 0.20 +
            food_score * 0.15 +
            (1.0 - fear_penalty) * 0.10
        )

        # Track decay: if ignored for multiple turns, stability drifts down
        if self.consecutive_ignores >= 2:
            raw -= 0.03 * self.consecutive_ignores

        # Clamp extreme jumps (max 0.08 change per turn)
        delta = raw - self.stability_score
        clamped_delta = max(-0.08, min(0.08, delta))

        # Smooth with rolling average
        self.stability_score += clamped_delta
        self.stability_score = max(0.0, min(1.0, self.stability_score))

        self._stability_history.append(self.stability_score)
        if len(self._stability_history) > 10:
            self._stability_history = self._stability_history[-10:]

    def get_stability_score(self) -> float:
        return round(self.stability_score, 3)

    def get_stability_trend(self) -> str:
        """Return whether stability is rising, falling, or stable."""
        if len(self._stability_history) < 3:
            return "stable"
        recent = self._stability_history[-3:]
        delta = recent[-1] - recent[0]
        if delta > 0.04:
            return "rising"
        elif delta < -0.04:
            return "falling"
        return "stable"

    # ═══════════════════════════════════════════════════════════
    # Confidence-Rated Predictions (merged, top 3)
    # ═══════════════════════════════════════════════════════════

    def _generate_predictions_confident(self, turn, civ_state):
        raw_predictions = []
        food = civ_state.get("food", 200)
        pop = civ_state.get("population", 100)
        morale = civ_state.get("morale", 60)
        danger = civ_state.get("danger_level", 10)
        energy = civ_state.get("energy", 100)
        workers = civ_state.get("available_workers", 30)

        # Food depletion
        food_consumed = max(5, pop // 5)
        food_produced = max(2, workers // 10)
        net_food = food_produced - food_consumed
        if net_food < 0 and food > 0:
            turns_left = food // abs(net_food)
            if turns_left <= 5:
                conf = "HIGH" if turns_left <= 2 else "MED"
                raw_predictions.append((
                    1.0 - turns_left / 6.0,
                    f"[{conf}] Food depleted in ~{turns_left} turns"
                ))

        # Morale crisis
        if morale < 20:
            raw_predictions.append((0.9, "[HIGH] Revolt imminent — morale critical"))
        elif morale < 35 and danger > 35:
            raw_predictions.append((0.6, "[MED] Instability rising — morale + danger"))

        # Danger
        if danger > 65:
            raw_predictions.append((0.85, "[HIGH] Military threat critical — defend now"))
        elif danger > 45 and self.consecutive_ignores >= 2:
            raw_predictions.append((0.5, "[MED] Unchecked threats escalating"))

        # Energy
        if energy < 15:
            raw_predictions.append((0.7, "[HIGH] Energy reserves nearly empty"))

        # Stability trend
        trend = self.get_stability_trend()
        if trend == "falling":
            raw_predictions.append((0.5, "[MED] Civilization stability declining"))

        # Active chain warnings
        for chain in self.active_chains:
            if not chain["mitigated"]:
                cid = chain["chain_id"].replace("_", " ").title()
                mitigators = CHAIN_MITIGATIONS.get(chain["chain_id"], [])
                hint = f" — try {' or '.join(mitigators)}" if mitigators else ""
                raw_predictions.append((0.7, f"[HIGH] Active: {cid}{hint}"))

        # Pending negative consequences
        neg_pending = sum(1 for e in self.delayed_effects if e["type"] == "negative")
        if neg_pending >= 2:
            raw_predictions.append((0.4, f"[MED] {neg_pending} consequences pending"))

        # Sort by confidence, take top 3
        raw_predictions.sort(key=lambda x: x[0], reverse=True)
        self.predictions = [p[1] for p in raw_predictions[:3]]

    # ═══════════════════════════════════════════════════════════
    # Message Modifiers (calibrated urgency, merged, less spam)
    # ═══════════════════════════════════════════════════════════

    def get_message_modifiers(self) -> Dict[str, Any]:
        mods = {"extra_weight": {}, "urgency_shift": {}, "tone_modifiers": []}

        # Only escalate after significant neglect (3+), capped
        for source, count in self.ignored_sources.items():
            if count >= 3:
                mods["extra_weight"][source] = min(18, count * 3)
                mods["urgency_shift"][source] = 1
                mods["tone_modifiers"].append(f"The {source}s grow impatient.")

        # Trust extremes only
        if self.citizen_trust < 18:
            mods["tone_modifiers"].append("Citizens speak with open hostility.")
            mods["urgency_shift"]["citizen"] = 1
        elif self.citizen_trust > 82:
            mods["tone_modifiers"].append("Citizens speak with deep loyalty.")

        if self.military_trust < 18:
            mods["tone_modifiers"].append("The military doubts the leader.")

        # Personality influence
        if self.personality_strength > 0.5:
            pw = {
                LeadershipStyle.MILITARY: ("defense", 7),
                LeadershipStyle.SCIENTIFIC: ("scientist", 7),
                LeadershipStyle.DIPLOMATIC: ("diplomat", 7),
                LeadershipStyle.ECONOMIC: ("trader", 7),
            }
            pair = pw.get(self.personality)
            if pair:
                mods["extra_weight"][pair[0]] = pair[1]

        # Limit tone modifiers to 2 (reduce clutter)
        mods["tone_modifiers"] = mods["tone_modifiers"][:2]

        return mods

    # ═══════════════════════════════════════════════════════════
    # Adaptive Reward Modifiers (smooth, milestone-aware)
    # ═══════════════════════════════════════════════════════════

    def get_reward_modifiers(self, turn, max_turns) -> Dict[str, float]:
        phase = turn / max(max_turns, 1)
        mods = {}

        # Smooth phase transitions (use cosine instead of hard cutoffs)
        if phase < 0.25:
            mods["survive_turn"] = 1.4
            mods["correct_action"] = 1.0
            mods["era_unlock"] = 0.8
        elif phase < 0.65:
            # Mid: blend between early and late
            blend = (phase - 0.25) / 0.4
            mods["survive_turn"] = 1.4 - blend * 0.5
            mods["correct_action"] = 1.0 + blend * 0.2
            mods["era_unlock"] = 0.8 + blend * 0.8
        else:
            mods["survive_turn"] = 0.7
            mods["correct_action"] = 1.0
            mods["stability_bonus"] = 1.8
            mods["era_unlock"] = 2.0

        # Crisis consistency
        if self.crises_streak >= 3:
            mods["crisis_averted"] = 1.3

        # Escalating neglect penalty
        if self.neglect_streak >= 2:
            mods["ignore_critical"] = 1.0 + 0.15 * min(self.neglect_streak, 6)

        # Stability-based
        if self.stability_score > 0.75:
            mods["stability_bonus"] = mods.get("stability_bonus", 1.0) * 1.15
        elif self.stability_score < 0.28:
            mods["collapse"] = 1.3

        return mods

    # ═══════════════════════════════════════════════════════════
    # Narrative (AI with fallback)
    # ═══════════════════════════════════════════════════════════

    def generate_narrative_summary(self, civ_state: Dict[str, Any]) -> str:
        if OLLAMA_API_KEY and self.important_events:
            events_text = "; ".join([
                f"T{e['turn']}:{'OK' if e['effective'] else 'FAIL'} {e['action']}({e['source']})"
                for e in self.important_events[-6:]
            ])
            chains = ", ".join(c["chain_id"] for c in self.active_chains) or "none"
            trend = self.get_stability_trend()
            prompt = (
                f"Historian: write 2 dramatic sentences about a {civ_state.get('era','tribal')}-era "
                f"civilization. Pop:{civ_state.get('population',0)}, morale:{civ_state.get('morale',0)}, "
                f"danger:{civ_state.get('danger_level',0)}. Leader:{self.personality}, "
                f"stability:{self.stability_score:.0%}({trend}). "
                f"Crises:{chains}. Events:{events_text}. No markdown."
            )
            result = _call_ollama(prompt, role="chronicle")
            if result:
                self.narrative_summary = result
                return result
        return self._rule_based_summary(civ_state)

    def _rule_based_summary(self, civ_state):
        parts = []
        pop = civ_state.get("population", 100)
        morale = civ_state.get("morale", 60)
        era = civ_state.get("era", "tribal")

        if pop > 200:
            parts.append(f"A great {era}-era civilization of {pop} souls")
        elif pop > 100:
            parts.append(f"A growing {era}-era settlement of {pop} people")
        else:
            parts.append(f"A struggling {era}-era village of {pop} survivors")

        if self.personality_strength > 0.4:
            adj = {
                LeadershipStyle.MILITARY: "forged in war",
                LeadershipStyle.SCIENTIFIC: "driven by discovery",
                LeadershipStyle.DIPLOMATIC: "built on trust",
                LeadershipStyle.ECONOMIC: "thriving on trade",
                LeadershipStyle.NEGLECTFUL: "suffering from neglect",
            }
            parts.append(adj.get(self.personality, ""))

        if self.active_chains:
            cids = [c["chain_id"].replace("_", " ") for c in self.active_chains if not c["mitigated"]]
            if cids:
                parts.append(f"faces {', '.join(cids)}")

        trend = self.get_stability_trend()
        if trend == "rising":
            parts.append("— stability grows.")
        elif trend == "falling":
            parts.append("— cracks appear.")
        elif morale > 70:
            parts.append("— the people thrive.")
        elif morale > 40:
            parts.append("— hope endures.")
        else:
            parts.append("— revolt looms.")

        if self.key_moments:
            parts.append(self.key_moments[-1])

        self.narrative_summary = " ".join(parts)
        return self.narrative_summary

    # ═══════════════════════════════════════════════════════════
    # Explainability (feedback on WHY trust/stability changed)
    # ═══════════════════════════════════════════════════════════

    def explain_last_action(self, action, source, urgency, effective) -> str:
        """Generate a brief explanation of the action's impact on the civilization."""
        parts = []
        if action == "ignore" and urgency in ("critical", "high"):
            parts.append(f"Ignoring the {source}'s plea eroded trust")
            if source in self._trust_betrayal_cooldown:
                parts.append("(recovery will be slow)")
        elif effective:
            parts.append(f"Effective response to {source} boosted confidence")
            if self.crises_streak >= 3:
                parts.append("(consistency bonus active)")
        
        trend = self.get_stability_trend()
        if trend == "falling":
            parts.append("⚠ Stability declining")
        elif trend == "rising":
            parts.append("📈 Stability improving")

        return " — ".join(parts) if parts else ""

    # ═══════════════════════════════════════════════════════════
    # Serialization
    # ═══════════════════════════════════════════════════════════

    def to_dict(self) -> Dict[str, Any]:
        return {
            "personality": self.personality,
            "personality_strength": round(self.personality_strength, 2),
            "citizen_trust": round(self.citizen_trust, 1),
            "military_trust": round(self.military_trust, 1),
            "scientific_trust": round(self.scientific_trust, 1),
            "worker_trust": round(self.worker_trust, 1),
            "fear_level": round(self.fear_level, 1),
            "stability_score": round(self.stability_score, 3),
            "stability_trend": self.get_stability_trend(),
            "consecutive_ignores": self.consecutive_ignores,
            "crises_streak": self.crises_streak,
            "active_chains": [c["chain_id"] for c in self.active_chains],
            "predictions": self.predictions[:3],
            "key_moments": self.key_moments[-5:],
            "narrative_summary": self.narrative_summary,
            "pending_delayed_effects": len(self.delayed_effects),
        }
