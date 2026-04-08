"""
Civil Command Center — Core Environment Logic
================================================
Implements the OpenEnv standard: reset(), step(), state()

The agent is the leader of a growing civilization.
Each turn it receives messages from citizens, scientists, workers,
defense, traders, and events. It must choose how to act.
The civilization evolves, eras unlock, and consequences compound.
"""

import uuid
import random
from typing import Optional, Dict, Any, List

from models import CivAction, CivObservation, CivState, Era
from data.emails import generate_turn_messages
from server.memory import CivilizationMemory


# ═══════════════════════════════════════════════════════════════
# Reward Constants
# ═══════════════════════════════════════════════════════════════

REWARDS = {
    # Correct handling
    "correct_action": 0.12,
    "crisis_averted": 0.25,
    "good_trade": 0.10,
    "research_approved": 0.15,
    "growth_invested": 0.08,

    # Milestones
    "era_unlock": 0.50,
    "population_milestone": 0.20,  # every +50 pop
    "stability_bonus": 0.10,      # per turn with morale > 60

    # Penalties
    "wrong_action": -0.06,
    "ignore_urgent": -0.15,
    "ignore_critical": -0.30,
    "resource_waste": -0.08,
    "collapse": -1.00,

    # Survival
    "survive_turn": 0.03,
}


# ═══════════════════════════════════════════════════════════════
# Era Thresholds
# ═══════════════════════════════════════════════════════════════

ERA_THRESHOLDS = {
    "tribal": 0,
    "bronze": 3,
    "iron": 5,
    "industrial": 7,
    "modern": 9,
}

ERA_ORDER = ["tribal", "bronze", "iron", "industrial", "modern"]


# ═══════════════════════════════════════════════════════════════
# Task Definitions
# ═══════════════════════════════════════════════════════════════

TASKS = {
    "task_easy": {
        "name": "Survival — Keep the Village Alive",
        "description": (
            "Lead a small village for 10 turns. Handle messages from citizens "
            "and workers. Keep food, morale, and population above collapse levels. "
            "Simple decisions, few threats."
        ),
        "difficulty": "easy",
        "max_turns": 10,
        "initial_state": {
            "population": 100, "food": 200, "energy": 100,
            "morale": 60, "technology_level": 1, "danger_level": 10,
            "available_workers": 30, "active_threats": 0,
        },
    },
    "task_medium": {
        "name": "Growth — Build a Thriving Settlement",
        "description": (
            "Grow your settlement over 20 turns. Balance food, energy, and morale "
            "while handling threats, trades, and research opportunities. "
            "Reach Bronze Era and grow population past 150."
        ),
        "difficulty": "medium",
        "max_turns": 20,
        "initial_state": {
            "population": 100, "food": 180, "energy": 80,
            "morale": 55, "technology_level": 1, "danger_level": 20,
            "available_workers": 25, "active_threats": 0,
        },
    },
    "task_hard": {
        "name": "Era Advancement — Rise of a Civilization",
        "description": (
            "Guide your civilization through 30 turns of crises, wars, "
            "disasters, and opportunities. Reach the Iron Era or beyond. "
            "Survive invasions, manage disease, unlock technologies, "
            "and keep your people alive and hopeful."
        ),
        "difficulty": "hard",
        "max_turns": 30,
        "initial_state": {
            "population": 120, "food": 150, "energy": 70,
            "morale": 45, "technology_level": 1, "danger_level": 35,
            "available_workers": 30, "active_threats": 1,
        },
    },
    "task_demo_5": {
        "name": "Quick Demo (5 Turns)",
        "description": "A rapid 5-turn showcase of the simulation mechanics for judges and quick testing.",
        "difficulty": "easy",
        "max_turns": 5,
        "initial_state": {
            "population": 100, "food": 200, "energy": 100,
            "morale": 60, "technology_level": 2, "danger_level": 10,
            "available_workers": 30, "active_threats": 0,
        },
    },
    "task_demo_10": {
        "name": "Standard Demo (10 Turns)",
        "description": "A 10-turn demonstration showing delayed consequences and mid-term stability.",
        "difficulty": "medium",
        "max_turns": 10,
        "initial_state": {
            "population": 110, "food": 160, "energy": 80,
            "morale": 50, "technology_level": 2, "danger_level": 25,
            "available_workers": 35, "active_threats": 0,
        },
    },
}


# ═══════════════════════════════════════════════════════════════
# Main Environment
# ═══════════════════════════════════════════════════════════════

class CivilCommandCenter:
    """
    Civil Command Center: AI Civilization Leader Environment

    The agent is the leader. Each turn it receives messages
    from different parts of the civilization and must decide
    how to act. The world evolves based on decisions.

    Implements: reset(), step(), state() — OpenEnv standard.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = CivState()
        self._current_messages: List[Dict[str, Any]] = []
        self._pending: List[Dict[str, Any]] = []
        self._actions_log: List[Dict[str, Any]] = []
        self._task_config: Dict[str, Any] = {}
        self._seed: int = 42
        self._memory = CivilizationMemory()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "task_easy",
        **kwargs,
    ) -> CivObservation:
        """Start a new episode."""
        self._seed = seed if seed is not None else random.randint(0, 999999)
        random.seed(self._seed)

        task = TASKS.get(task_id, TASKS["task_easy"])
        self._task_config = task
        init = task["initial_state"]

        self._state = CivState(
            episode_id=episode_id or str(uuid.uuid4()),
            turn=0,
            population=init["population"],
            food=init["food"],
            energy=init["energy"],
            morale=init["morale"],
            technology_level=init["technology_level"],
            danger_level=init["danger_level"],
            available_workers=init["available_workers"],
            active_threats=init["active_threats"],
            progress_score=0.0,
            era="tribal",
            total_reward=0.0,
            messages_handled=0,
            messages_ignored=0,
            correct_actions=0,
            crises_averted=0,
            crises_failed=0,
            tech_unlocks=0,
            eras_reached=["tribal"],
            population_peak=init["population"],
            collapse=False,
            collapse_reason="",
            pending_messages=[],
            task_id=task_id,
            max_turns=task["max_turns"],
            difficulty=task["difficulty"],
        )

        self._pending = []
        self._actions_log = []
        self._memory.reset()

        # Generate first turn messages
        self._current_messages = generate_turn_messages(
            turn=0,
            difficulty=task["difficulty"],
            civ_state=self._get_state_dict(),
            seed=self._seed,
        )

        return self._build_observation(
            reward=None,
            done=False,
            message=f"🏛️ You are the leader of a {task['difficulty']} civilization. {task['name']}",
            effective=None,
        )

    def step(
        self,
        action: CivAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> CivObservation:
        """Process one action from the leader."""
        self._state.turn += 1
        turn_reward = 0.0
        feedback = ""
        reward_mods = self._memory.get_reward_modifiers(self._state.turn, self._state.max_turns)
        effective = None

        # ── 1. Process the agent's action ────────────────────
        action_type = action.action_type.lower().strip()
        target_id = action.target_message_id

        # Find the targeted message (or best match)
        matched_msg = self._find_message(target_id, action_type)

        if matched_msg:
            r, msg, eff = self._process_action(action_type, matched_msg)
            turn_reward += r
            feedback = msg
            effective = eff
            self._state.messages_handled += 1
        elif action_type == "ignore":
            feedback = "🤷 Leader chose to ignore this turn."
            turn_reward += REWARDS["survive_turn"]
            effective = False
        else:
            # Action doesn't match any message — partial credit
            feedback = f"⚙️ Action '{action_type}' taken (no specific message targeted)."
            turn_reward += self._apply_generic_action(action_type)
            effective = True

        # ── 1b. Record action in memory ────────────────────
        source = matched_msg["source"] if matched_msg else "none"
        urgency = matched_msg["urgency"] if matched_msg else "low"
        self._memory.record_action(
            turn=self._state.turn,
            action=action_type,
            source=source,
            urgency=urgency,
            effective=effective or False,
            reward=turn_reward,
            civ_state=self._get_state_dict(),
        )

        # ── 1c. Explainability Feedback ────────────────────
        explanation = self._memory.explain_last_action(action_type, source, urgency, effective or False)
        if explanation:
            feedback += f" 💡 {explanation}"

        # ── 2. Process pending (ignored) messages ────────────
        decay_penalty = self._process_pending()
        turn_reward += decay_penalty
        if decay_penalty < -0.1:
            feedback += f" ⚠️ Ignored messages had consequences!"

        # ── 2b. Process delayed effects from memory ──────────
        triggered = self._memory.process_delayed_effects(self._state.turn)
        for effect in triggered:
            self._apply_consequences(effect["effects"])
            if effect["type"] == "positive":
                turn_reward += 0.08
                feedback += f" 🌟 {effect['reason']}"
            else:
                turn_reward -= 0.08
                feedback += f" ⚡ {effect['reason']}"

        # ── 2c. Cleanup expired event chains ──────────────────
        self._memory.cleanup_chains(self._state.turn)

        # ── 3. Natural decay/growth + personality effects ─────
        self._apply_natural_changes()
        personality_fx = self._memory.get_personality_effects()
        if personality_fx:
            self._apply_consequences(personality_fx)

        # ── 4. Check era advancement ─────────────────────────
        era_bonus = self._check_era_advancement()
        turn_reward += era_bonus

        # ── 5. Milestone checks ──────────────────────────────
        turn_reward += self._check_milestones()

        # ── 6. Survival bonus (adaptive) ──────────────────────
        survive_mod = reward_mods.get("survive_turn", 1.0)
        stability_mod = reward_mods.get("stability_bonus", 1.0)
        turn_reward += REWARDS["survive_turn"] * survive_mod
        if self._state.morale > 60:
            turn_reward += REWARDS["stability_bonus"] * stability_mod

        # ── 7. Check for collapse ────────────────────────────
        collapse, reason = self._check_collapse()
        if collapse:
            self._state.collapse = True
            self._state.collapse_reason = reason
            turn_reward += REWARDS["collapse"]
            feedback += f" 💀 CIVILIZATION COLLAPSED: {reason}"

        # ── 8. Log action ────────────────────────────────────
        self._actions_log.append({
            "turn": self._state.turn,
            "action": action_type,
            "target_message": target_id,
            "reward": turn_reward,
            "effective": effective,
        })

        # ── 9. Update state ──────────────────────────────────
        self._state.total_reward += turn_reward
        self._state.population_peak = max(self._state.population_peak, self._state.population)

        # ── 10. Check if done ────────────────────────────────
        done = (
            collapse
            or self._state.turn >= self._state.max_turns
        )

        # Generate next turn's messages if not done
        if not done:
            memory_mods = self._memory.get_message_modifiers()
            self._current_messages = generate_turn_messages(
                turn=self._state.turn,
                difficulty=self._state.difficulty,
                civ_state=self._get_state_dict(),
                seed=self._seed + self._state.turn,
                memory_modifiers=memory_mods,
            )
        else:
            self._current_messages = []
            if not collapse:
                feedback += " 🏁 Episode complete!"

            # ── Compute graded task score for final reward ────
            # OpenEnv requires the final reward to be the task score
            # strictly between 0 and 1 (exclusive).
            summary = self.get_episode_summary()
            task_id = self._state.task_id
            try:
                if task_id == "task_hard":
                    from graders.grader_hard import grade_hard
                    turn_reward = grade_hard(summary)
                elif task_id == "task_medium" or task_id == "task_demo_10":
                    from graders.grader_medium import grade_medium
                    turn_reward = grade_medium(summary)
                else:
                    from graders.grader_easy import grade_easy
                    turn_reward = grade_easy(summary)
            except Exception:
                # Fallback: clamp accumulated reward to valid range
                turn_reward = round(min(0.99, max(0.01, self._state.total_reward / max(self._state.max_turns, 1))), 4)

        return self._build_observation(
            reward=turn_reward,
            done=done,
            message=feedback,
            effective=effective,
        )

    @property
    def state(self) -> CivState:
        """Return the full internal state."""
        return self._state

    # ═══════════════════════════════════════════════════════════
    # Action Processing
    # ═══════════════════════════════════════════════════════════

    def _find_message(self, target_id: Optional[str], action_type: str) -> Optional[Dict]:
        """Find a message matching the target ID or best-action."""
        if target_id:
            for msg in self._current_messages:
                if msg["id"] == target_id:
                    return msg

        # Auto-match: find message where best_action matches action_type
        for msg in self._current_messages:
            if msg["best_action"] == action_type:
                return msg

        # Return first urgent message as fallback
        for msg in sorted(self._current_messages,
                         key=lambda m: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(m["urgency"], 4)):
            return msg

        return None

    def _process_action(self, action_type: str, msg: Dict) -> tuple:
        """Process an action against a specific message."""
        best = msg["best_action"]
        consequences = msg["consequences"]
        urgency = msg["urgency"]

        if action_type == best:
            # Correct action — apply positive consequences
            self._apply_consequences(consequences)
            self._state.correct_actions += 1

            if urgency in ("critical", "high"):
                self._state.crises_averted += 1
                reward = REWARDS["crisis_averted"]
                return reward, f"✅ Excellent! {msg['subject']} — Handled correctly!", True
            else:
                reward = REWARDS["correct_action"]
                return reward, f"✅ {msg['subject']} — Good decision.", True

        elif action_type == "ignore":
            # Ignored — add to pending for future decay
            self._pending.append(msg)
            self._state.messages_ignored += 1
            if urgency == "critical":
                return REWARDS["ignore_critical"], f"⚠️ CRITICAL ignored: {msg['subject']}", False
            elif urgency == "high":
                return REWARDS["ignore_urgent"], f"⚠️ Urgent ignored: {msg['subject']}", False
            else:
                return -0.02, f"📭 Ignored: {msg['subject']}", False
        else:
            # Wrong action — partial consequences
            partial = {k: v * 0.3 for k, v in consequences.items()}
            self._apply_consequences(partial)
            return REWARDS["wrong_action"], f"❌ Suboptimal action for: {msg['subject']}", False

    def _apply_generic_action(self, action_type: str) -> float:
        """Apply a generic (untargeted) action effect."""
        effects = {
            "allocate_food": {"food": -20, "morale": 5, "population": 2},
            "allocate_workers": {"available_workers": -3, "energy": 10},
            "approve_research": {"technology_level": 0.3, "progress_score": 5},
            "defend": {"danger_level": -8, "morale": 3},
            "calm_citizens": {"morale": 10},
            "accept_trade": {},
            "reject_trade": {},
            "invest_growth": {"progress_score": 5, "food": -10, "energy": -5},
            "emergency_response": {"danger_level": -5, "food": -10, "energy": -5},
        }
        fx = effects.get(action_type, {})
        self._apply_consequences(fx)
        return REWARDS["correct_action"] * 0.3

    # ═══════════════════════════════════════════════════════════
    # World Simulation
    # ═══════════════════════════════════════════════════════════

    def _apply_consequences(self, effects: Dict[str, float]):
        """Apply a dict of state changes."""
        for key, value in effects.items():
            if hasattr(self._state, key):
                current = getattr(self._state, key)
                if isinstance(current, int):
                    setattr(self._state, key, max(0, current + int(value)))
                elif isinstance(current, float):
                    setattr(self._state, key, max(0.0, current + float(value)))

        # Clamp values
        self._state.morale = min(100, max(0, self._state.morale))
        self._state.danger_level = min(100, max(0, self._state.danger_level))
        self._state.technology_level = min(10, max(0, self._state.technology_level))
        self._state.available_workers = max(0, self._state.available_workers)

    def _process_pending(self) -> float:
        """Process pending (ignored) messages — decay consequences."""
        penalty = 0.0
        still_pending = []

        for msg in self._pending:
            msg["decay_turns"] -= 1

            if msg["decay_turns"] <= 0:
                # Time's up — apply ignore penalties
                ignore_penalty = msg.get("ignore_penalty", {})
                if ignore_penalty:
                    self._apply_consequences(ignore_penalty)
                    penalty -= 0.1
                    if msg["urgency"] in ("critical", "high"):
                        self._state.crises_failed += 1
                        penalty -= 0.15
            else:
                still_pending.append(msg)

        self._pending = still_pending
        self._state.pending_messages = [
            {"id": m["id"], "subject": m["subject"], "turns_left": m["decay_turns"]}
            for m in self._pending
        ]
        return penalty

    def _apply_natural_changes(self):
        """Natural per-turn state changes."""
        s = self._state

        # Food consumption: ~1 per 5 population
        food_consumed = max(5, s.population // 5)
        s.food = max(0, s.food - food_consumed)

        # Energy consumption
        energy_consumed = max(2, s.available_workers // 5)
        s.energy = max(0, s.energy - energy_consumed)

        # Small natural food production
        s.food += max(2, s.available_workers // 10)
        s.energy += max(1, s.available_workers // 15)

        # Population growth if food and morale are good
        if s.food > 100 and s.morale > 50:
            s.population += random.randint(1, 3)
            s.available_workers += random.randint(0, 1)

        # Population decline if food is zero
        if s.food <= 0:
            loss = random.randint(3, 8)
            s.population = max(0, s.population - loss)
            s.morale = max(0, s.morale - 10)

        # Morale decay if danger is high
        if s.danger_level > 60:
            s.morale = max(0, s.morale - 3)

        # Danger slowly reduces naturally (patrols)
        s.danger_level = max(0, s.danger_level - 1)

    def _check_era_advancement(self) -> float:
        """Check if civilization has reached a new era."""
        tech = self._state.technology_level
        current_era = self._state.era
        reward = 0.0

        for era_name in reversed(ERA_ORDER):
            threshold = ERA_THRESHOLDS[era_name]
            if tech >= threshold and era_name not in self._state.eras_reached:
                self._state.era = era_name
                self._state.eras_reached.append(era_name)
                self._state.tech_unlocks += 1
                reward += REWARDS["era_unlock"]
                break

        # Update era to highest reached
        for era_name in reversed(ERA_ORDER):
            if tech >= ERA_THRESHOLDS[era_name]:
                self._state.era = era_name
                break

        return reward

    def _check_milestones(self) -> float:
        """Check for population and progress milestones."""
        reward = 0.0
        pop = self._state.population

        # Population milestones at 150, 200, 250, ...
        milestone_threshold = 150
        while pop >= milestone_threshold:
            if milestone_threshold not in getattr(self, '_pop_milestones', set()):
                if not hasattr(self, '_pop_milestones'):
                    self._pop_milestones = set()
                self._pop_milestones.add(milestone_threshold)
                reward += REWARDS["population_milestone"]
            milestone_threshold += 50

        return reward

    def _check_collapse(self) -> tuple:
        """Check if civilization has collapsed."""
        s = self._state

        if s.population <= 0:
            return True, "Population reached zero — your people are gone."
        if s.morale <= 0:
            return True, "Morale collapsed — citizens revolted and overthrew the leader."
        if s.food <= 0 and s.population < 20:
            return True, "Famine and starvation — too few people left to sustain the settlement."
        if s.danger_level >= 100:
            return True, "Overwhelmed by threats — the civilization was destroyed."

        return False, ""

    # ═══════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════

    def _get_state_dict(self) -> Dict[str, Any]:
        """Get state as dict for message generator."""
        return {
            "food": self._state.food,
            "morale": self._state.morale,
            "danger_level": self._state.danger_level,
            "technology_level": self._state.technology_level,
            "population": self._state.population,
            "energy": self._state.energy,
            "era": self._state.era,
        }

    def _build_observation(
        self,
        reward: Optional[float],
        done: bool,
        message: str,
        effective: Optional[bool],
    ) -> CivObservation:
        """Build the observation the agent sees."""
        # Strip hidden fields from messages for agent
        visible_messages = []
        for m in self._current_messages:
            visible_messages.append({
                "id": m["id"],
                "source": m["source"],
                "sender_name": m["sender_name"],
                "subject": m["subject"],
                "body": m["body"],
                "urgency": m["urgency"],
            })

        s = self._state
        task = self._task_config

        return CivObservation(
            done=done,
            reward=reward,
            turn=s.turn,
            max_turns=s.max_turns,
            era=s.era,
            population=s.population,
            food=s.food,
            energy=s.energy,
            morale=s.morale,
            technology_level=s.technology_level,
            danger_level=s.danger_level,
            available_workers=s.available_workers,
            active_threats=s.active_threats,
            progress_score=s.progress_score,
            messages=visible_messages,
            num_messages=len(visible_messages),
            message=message,
            last_action_effective=effective,
            task_name=task.get("name", ""),
            task_description=task.get("description", ""),
            total_reward=s.total_reward,
            # Memory data
            personality=self._memory.personality,
            personality_strength=round(self._memory.personality_strength, 2),
            citizen_trust=round(self._memory.citizen_trust, 1),
            military_trust=round(self._memory.military_trust, 1),
            narrative_summary=self._memory.narrative_summary,
            pending_consequences=len(self._memory.delayed_effects),
            stability_score=round(self._memory.stability_score, 3),
            stability_trend=self._memory.get_stability_trend(),
            predictions=self._memory.predictions[:3],
            active_chains=[c["chain_id"] for c in self._memory.active_chains],
        )

    def get_available_tasks(self) -> List[Dict[str, str]]:
        """Return available tasks."""
        return [
            {
                "task_id": tid,
                "name": t["name"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "max_turns": t["max_turns"],
            }
            for tid, t in TASKS.items()
        ]

    def get_episode_summary(self) -> Dict[str, Any]:
        """Return a summary of the completed episode."""
        s = self._state
        mem = self._memory

        # Generate narrative for the episode
        mem.generate_narrative_summary(self._get_state_dict())

        return {
            "episode_id": s.episode_id,
            "task_id": s.task_id,
            "difficulty": s.difficulty,
            "turns_played": s.turn,
            "total_turns": s.turn,
            "max_turns": s.max_turns,
            "total_reward": round(s.total_reward, 4),
            "survived": not s.collapse,
            "population_final": s.population,
            "population_peak": s.population_peak,
            "food_final": s.food,
            "energy_final": s.energy,
            "morale_final": s.morale,
            "technology_level": s.technology_level,
            "era_final": s.era,
            "final_era": s.era,
            "eras_reached": s.eras_reached,
            "danger_level": s.danger_level,
            "progress_score": round(s.progress_score, 2),
            "messages_handled": s.messages_handled,
            "messages_ignored": s.messages_ignored,
            "correct_actions": s.correct_actions,
            "crises_averted": s.crises_averted,
            "crises_failed": s.crises_failed,
            "tech_unlocks": s.tech_unlocks,
            "collapse": s.collapse,
            "collapse_reason": s.collapse_reason,
            "actions": self._actions_log,
            # Memory-enhanced fields
            "memory": mem.to_dict(),
            "narrative": mem.narrative_summary,
        }
