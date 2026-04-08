"""
Civil Command Center — Message & Event Generator
==================================================
Creates diverse, realistic messages from 8 sources that make the
civilization feel alive. Each message has consequences, urgency,
and a best-action for deterministic grading.
"""

import random
import uuid
from typing import List, Dict, Any


# ═══════════════════════════════════════════════════════════════
# Sender Names by Source
# ═══════════════════════════════════════════════════════════════

SENDERS = {
    "citizen": [
        "Elder Mara", "Farmer Tolek", "Artisan Reva", "Mother Kessa",
        "Youth Leader Dex", "Healer Yuna", "Merchant Voss", "Widow Nila",
    ],
    "scientist": [
        "Chief Scientist Orion", "Researcher Lyra", "Inventor Nyx",
        "Astronomer Vega", "Alchemist Sol", "Scholar Cass",
    ],
    "worker": [
        "Foreman Brak", "Miner Kael", "Builder Thane", "Farmer Gwen",
        "Woodcutter Finn", "Blacksmith Jora",
    ],
    "defense": [
        "General Ash", "Scout Captain Wren", "Guard Commander Rook",
        "Wall Sentinel Bryn", "Border Ranger Kite",
    ],
    "trader": [
        "Merchant Caravan Leader", "Sea Trader Nemo", "Silk Road Emissary",
        "Mountain Clan Trader", "Desert Nomad Merchant",
    ],
    "event": [
        "The Wind", "Nature", "The Stars", "The Earth",
        "Fate", "Fortune",
    ],
    "advisor": [
        "Royal Advisor Sage", "Council Elder Theron", "Minister of Affairs",
        "Strategic Advisor Iris",
    ],
    "diplomat": [
        "Ambassador Kael", "Foreign Envoy", "Tribal Chieftain Mako",
        "Alliance Messenger", "Peace Envoy Lira",
    ],
}


# ═══════════════════════════════════════════════════════════════
# Message Templates by Source
# ═══════════════════════════════════════════════════════════════

MESSAGE_TEMPLATES = {
    "citizen": [
        {
            "subject": "🍞 Food shortage in the eastern quarter",
            "body": "Leader, families in the eastern quarter are running low on food. Children are hungry and morale is dropping. We need an immediate food allocation to prevent unrest. The people are looking to you.",
            "urgency": "high",
            "best_action": "allocate_food",
            "consequences": {"morale": 10, "population": 2},
            "ignore_penalty": {"morale": -15, "population": -5},
            "decay_turns": 2,
        },
        {
            "subject": "🍚 Grain storage depleted",
            "body": "My Lord, the granaries in the lower districts are nearly empty. If we do not dispatch emergency rations soon, starvation will start taking lives. Please allocate food immediately.",
            "urgency": "high",
            "best_action": "allocate_food",
            "consequences": {"morale": 10, "population": 2},
            "ignore_penalty": {"morale": -15, "population": -5},
            "decay_turns": 2,
        },
        {
            "subject": "🌾 Farmers report failed crops",
            "body": "Leader! A blight has hit the recent harvest. Many families have nothing to eat tonight. We must share our central reserves. If we allocate food, they will remember your kindness.",
            "urgency": "high",
            "best_action": "allocate_food",
            "consequences": {"morale": 10, "population": 2},
            "ignore_penalty": {"morale": -15, "population": -5},
            "decay_turns": 2,
        },
        {
            "subject": "😡 Workers are unhappy with conditions",
            "body": "Leader, the workers are grumbling about long hours and poor conditions. A few have started talking about refusing to work. If you address their concerns now, we can avoid a full shutdown. They need to hear from you directly.",
            "urgency": "medium",
            "best_action": "calm_citizens",
            "consequences": {"morale": 15, "available_workers": 5},
            "ignore_penalty": {"morale": -10, "available_workers": -8},
            "decay_turns": 3,
        },
        {
            "subject": "🗣️ Rumors of a strike spreading",
            "body": "There are whispers in the taverns that the laborers plan to lay down their tools tomorrow. Please go out and calm the citizens before this turns into a city-wide strike.",
            "urgency": "medium",
            "best_action": "calm_citizens",
            "consequences": {"morale": 15, "available_workers": 5},
            "ignore_penalty": {"morale": -10, "available_workers": -8},
            "decay_turns": 3,
        },
        {
            "subject": "🎉 Citizens celebrate the harvest festival",
            "body": "Great Leader! The harvest was bountiful this season. The citizens are celebrating and spirits are high. If you allocate some extra food for a grand feast, morale would soar and more families might join our settlement.",
            "urgency": "low",
            "best_action": "allocate_food",
            "consequences": {"morale": 20, "population": 5, "food": -30},
            "ignore_penalty": {"morale": -3},
            "decay_turns": 5,
        },
        {
            "subject": "🏠 Request for new housing",
            "body": "Leader, our population is growing but housing is limited. Families are doubling up and tensions are rising. We need workers assigned to build new shelters before the cold season arrives.",
            "urgency": "medium",
            "best_action": "allocate_workers",
            "consequences": {"population": 8, "morale": 5, "available_workers": -5},
            "ignore_penalty": {"morale": -8, "population": -3},
            "decay_turns": 4,
        },
        {
            "subject": "⚖️ Dispute between two clans",
            "body": "Leader, the River Clan and the Hill Clan are fighting over fishing rights. Elders have tried to mediate but failed. Only your word can settle this before it turns violent. The people need your judgment.",
            "urgency": "high",
            "best_action": "calm_citizens",
            "consequences": {"morale": 12, "danger_level": -5},
            "ignore_penalty": {"morale": -20, "danger_level": 10, "population": -2},
            "decay_turns": 2,
        },
    ],
    "scientist": [
        {
            "subject": "🔬 Breakthrough: Improved irrigation discovered",
            "body": "Leader! We've made a breakthrough in irrigation technology. If you approve the research and allocate workers, we can double our food production within a few turns. This could feed the entire settlement for years.",
            "urgency": "medium",
            "best_action": "approve_research",
            "consequences": {"technology_level": 1, "food": 40, "progress_score": 15},
            "ignore_penalty": {"progress_score": -5},
            "decay_turns": 5,
        },
        {
            "subject": "⚡ Energy source prototype ready",
            "body": "We've built a prototype for a new energy source — a waterwheel system. With your approval and some workers, we can power our forges and double tool production. This is a turning point for our civilization.",
            "urgency": "medium",
            "best_action": "approve_research",
            "consequences": {"technology_level": 1, "energy": 50, "progress_score": 20},
            "ignore_penalty": {"progress_score": -3},
            "decay_turns": 6,
        },
        {
            "subject": "🗺️ New territory mapping complete",
            "body": "Leader, our scouts have finished mapping the northern territories. Rich mineral deposits and fertile land await. If we invest in growth now, we can expand our borders and gain critical resources.",
            "urgency": "low",
            "best_action": "invest_growth",
            "consequences": {"progress_score": 10, "food": 20, "energy": 15},
            "ignore_penalty": {},
            "decay_turns": 8,
        },
        {
            "subject": "🛡️ Military technology advancement",
            "body": "Our researchers have developed stronger defensive structures. With your approval, we can upgrade our walls and train defenders with new weapons. This would significantly reduce our vulnerability to attacks.",
            "urgency": "medium",
            "best_action": "approve_research",
            "consequences": {"technology_level": 1, "danger_level": -15, "progress_score": 10},
            "ignore_penalty": {"danger_level": 5},
            "decay_turns": 4,
        },
    ],
    "worker": [
        {
            "subject": "⛏️ Mines running low on ore",
            "body": "Boss, the mines are nearly depleted. We need to prospect new veins or our tool production will halt within 3 turns. Assign more workers to prospecting or we'll face an energy and production crisis.",
            "urgency": "high",
            "best_action": "allocate_workers",
            "consequences": {"energy": 30, "available_workers": -3},
            "ignore_penalty": {"energy": -40, "available_workers": -5},
            "decay_turns": 3,
        },
        {
            "subject": "🌾 Bumper crop ready for harvest",
            "body": "Great news, Leader! The fields are overflowing. If you assign extra workers to harvest, we can store enough food for two seasons. But we need to act fast before the crop rots.",
            "urgency": "medium",
            "best_action": "allocate_workers",
            "consequences": {"food": 60, "available_workers": -5},
            "ignore_penalty": {"food": -20},
            "decay_turns": 2,
        },
        {
            "subject": "🔧 Infrastructure needs repair",
            "body": "Leader, roads and bridges are crumbling. Trade routes are getting slower and dangerous. We need workers to repair infrastructure before it affects our supply chains and safety.",
            "urgency": "medium",
            "best_action": "allocate_workers",
            "consequences": {"progress_score": 8, "danger_level": -5, "available_workers": -4},
            "ignore_penalty": {"danger_level": 8, "progress_score": -5},
            "decay_turns": 4,
        },
        {
            "subject": "💪 Workers volunteering for extra shifts",
            "body": "Leader, a group of dedicated workers is volunteering for extra shifts. If you provide extra food rations, they'll boost production significantly this turn. A small food investment for a big gain.",
            "urgency": "low",
            "best_action": "allocate_food",
            "consequences": {"energy": 20, "food": -15, "progress_score": 5},
            "ignore_penalty": {},
            "decay_turns": 2,
        },
    ],
    "defense": [
        {
            "subject": "⚔️ Enemy raiders spotted at the border!",
            "body": "URGENT — Leader, a band of raiders has been spotted approaching from the west! They are well-armed and heading straight for our farming villages. We need an immediate defense response or we will lose people and resources!",
            "urgency": "critical",
            "best_action": "defend",
            "consequences": {"danger_level": -20, "morale": 10},
            "ignore_penalty": {"population": -15, "food": -40, "morale": -25, "danger_level": 20},
            "decay_turns": 1,
        },
        {
            "subject": "🐺 Wild beasts threatening livestock",
            "body": "Leader, packs of wolves have been attacking our livestock at night. Farmers are terrified. We need to defend the farming sector or we'll lose significant food supply.",
            "urgency": "high",
            "best_action": "defend",
            "consequences": {"food": 10, "danger_level": -10, "morale": 5},
            "ignore_penalty": {"food": -25, "morale": -10, "danger_level": 10},
            "decay_turns": 2,
        },
        {
            "subject": "🏰 Walls need reinforcement",
            "body": "The settlement walls have weakened after the last storm. If enemies attack now, we're vulnerable. Assign workers to reinforce the walls before we face another threat.",
            "urgency": "medium",
            "best_action": "allocate_workers",
            "consequences": {"danger_level": -15, "available_workers": -4},
            "ignore_penalty": {"danger_level": 15},
            "decay_turns": 4,
        },
        {
            "subject": "🚨 Possible invasion force assembling",
            "body": "CRITICAL — Our scouts report a large force assembling beyond the mountains. This is not a raid — this could be a full invasion. We need emergency preparations: defend our borders, stockpile food, and rally the people.",
            "urgency": "critical",
            "best_action": "emergency_response",
            "consequences": {"danger_level": -25, "morale": 15, "progress_score": 10},
            "ignore_penalty": {"population": -25, "food": -50, "morale": -30, "danger_level": 40},
            "decay_turns": 1,
        },
    ],
    "trader": [
        {
            "subject": "🤝 Trade offer: Food for Energy",
            "body": "Greetings, Leader! My caravan offers 50 units of food in exchange for 30 units of energy. Fair trade for both sides. This deal won't last — we leave at dawn. Accept or reject?",
            "urgency": "medium",
            "best_action": "accept_trade",
            "consequences": {"food": 50, "energy": -30},
            "ignore_penalty": {},
            "decay_turns": 1,
        },
        {
            "subject": "💎 Rare materials available",
            "body": "Leader, a desert trader brings rare crystals that could accelerate our research. They want 40 food and 10 workers for a season. The science team says this could be worth a full technology level advancement.",
            "urgency": "medium",
            "best_action": "accept_trade",
            "consequences": {"technology_level": 1, "food": -40, "available_workers": -3, "progress_score": 15},
            "ignore_penalty": {},
            "decay_turns": 2,
        },
        {
            "subject": "⚠️ Suspicious traders at the gate",
            "body": "Leader, traders have arrived but our scouts noticed hidden weapons in their carts. This could be a trick. If we accept them in, we risk an ambush. If we reject, we might offend genuine traders. Your call.",
            "urgency": "high",
            "best_action": "reject_trade",
            "consequences": {"danger_level": -10, "morale": 5},
            "ignore_penalty": {"danger_level": 15, "population": -5, "food": -20},
            "decay_turns": 1,
        },
        {
            "subject": "🏪 Traveling merchants offer workers",
            "body": "A group of skilled laborers from a distant land seeks refuge. They'll work for food. Accepting them would boost your workforce by 10 for the cost of 25 food per turn.",
            "urgency": "low",
            "best_action": "accept_trade",
            "consequences": {"available_workers": 10, "population": 10, "food": -25},
            "ignore_penalty": {},
            "decay_turns": 3,
        },
    ],
    "event": [
        {
            "subject": "🌊 FLOOD! Rivers overflowing!",
            "body": "DISASTER — Heavy rains have caused the river to burst its banks! Farmlands are flooding and people are trapped. Immediate emergency response is needed to save lives and salvage food supplies!",
            "urgency": "critical",
            "best_action": "emergency_response",
            "consequences": {"population": -2, "food": -15, "morale": 5},
            "ignore_penalty": {"population": -20, "food": -60, "morale": -25, "energy": -20},
            "decay_turns": 1,
        },
        {
            "subject": "🔥 Forest fire spreading toward settlement!",
            "body": "A wildfire is raging through the eastern forest and heading toward our settlement! We need emergency response — evacuate citizens, organize firefighting crews, and protect food stores!",
            "urgency": "critical",
            "best_action": "emergency_response",
            "consequences": {"energy": -10, "danger_level": -5},
            "ignore_penalty": {"population": -15, "food": -30, "energy": -25, "morale": -20, "danger_level": 20},
            "decay_turns": 1,
        },
        {
            "subject": "🌤️ Perfect weather brings prosperity",
            "body": "The skies have blessed us! Perfect weather means excellent growing conditions. If farmers are available, food production will surge naturally. Consider investing in growth to maximize this opportunity.",
            "urgency": "low",
            "best_action": "invest_growth",
            "consequences": {"food": 35, "morale": 8, "progress_score": 5},
            "ignore_penalty": {},
            "decay_turns": 3,
        },
        {
            "subject": "🦠 Disease outbreak in the settlement!",
            "body": "A mysterious illness is spreading! People are falling sick and fear is growing. We need emergency response — quarantine the sick, distribute medicine, and calm the population before panic spreads!",
            "urgency": "critical",
            "best_action": "emergency_response",
            "consequences": {"population": -3, "morale": 5},
            "ignore_penalty": {"population": -25, "morale": -30, "available_workers": -10},
            "decay_turns": 1,
        },
        {
            "subject": "☄️ Strange lights in the sky",
            "body": "Mysterious lights appeared in the night sky. The citizens are either amazed or terrified depending on how you frame it. Scientists want to study the phenomenon. This could boost knowledge or cause panic.",
            "urgency": "low",
            "best_action": "approve_research",
            "consequences": {"technology_level": 1, "morale": 5, "progress_score": 10},
            "ignore_penalty": {"morale": -5},
            "decay_turns": 4,
        },
    ],
    "advisor": [
        {
            "subject": "📊 Quarterly status: resources declining",
            "body": "Leader, I must inform you that our food reserves are trending downward and energy is barely stable. I advise investing in growth and infrastructure now, before we enter a crisis cycle. Prevention is cheaper than recovery.",
            "urgency": "medium",
            "best_action": "invest_growth",
            "consequences": {"food": 20, "energy": 15, "progress_score": 8},
            "ignore_penalty": {"food": -10, "energy": -10},
            "decay_turns": 3,
        },
        {
            "subject": "🎓 Train the next generation",
            "body": "Leader, we should allocate workers to educate our youth. An investment in knowledge today will pay dividends in technology and progress tomorrow. I recommend approving research and training programs.",
            "urgency": "low",
            "best_action": "approve_research",
            "consequences": {"technology_level": 1, "progress_score": 12, "population": 3},
            "ignore_penalty": {},
            "decay_turns": 6,
        },
        {
            "subject": "⚠️ Morale critical — address immediately",
            "body": "URGENT advisory: Citizen morale has dropped to dangerous levels. We are on the verge of civil unrest. You must calm the citizens immediately or risk losing control of the settlement entirely.",
            "urgency": "critical",
            "best_action": "calm_citizens",
            "consequences": {"morale": 25},
            "ignore_penalty": {"morale": -20, "population": -10, "danger_level": 15},
            "decay_turns": 1,
        },
    ],
    "diplomat": [
        {
            "subject": "🕊️ Alliance offer from the Mountain Clans",
            "body": "Leader, the Mountain Clans offer a defensive alliance. They will reduce our danger from northern threats in exchange for food tribute. It costs 20 food per turn but provides lasting safety.",
            "urgency": "medium",
            "best_action": "accept_trade",
            "consequences": {"danger_level": -20, "food": -20, "progress_score": 10},
            "ignore_penalty": {"danger_level": 5},
            "decay_turns": 3,
        },
        {
            "subject": "🚩 Rival civilization threatens war",
            "body": "An envoy from the Iron Coast has delivered an ultimatum: submit to their authority or face invasion within 2 turns. We must defend our borders immediately or find a diplomatic solution. This is not a bluff.",
            "urgency": "critical",
            "best_action": "defend",
            "consequences": {"danger_level": -15, "morale": 10, "progress_score": 5},
            "ignore_penalty": {"population": -20, "food": -30, "morale": -20, "danger_level": 30},
            "decay_turns": 2,
        },
        {
            "subject": "🎭 Cultural exchange opportunity",
            "body": "A delegation from a peaceful eastern civilization offers cultural exchange. They'll teach us new techniques (boosting technology) if we welcome them with food and shelter. A chance for peaceful advancement.",
            "urgency": "low",
            "best_action": "accept_trade",
            "consequences": {"technology_level": 1, "morale": 8, "food": -15, "progress_score": 10},
            "ignore_penalty": {},
            "decay_turns": 4,
        },
    ],
}


# ═══════════════════════════════════════════════════════════════
# Generator Functions
# ═══════════════════════════════════════════════════════════════

def _pick_message(source: str, index: int = 0) -> Dict[str, Any]:
    """Pick a message template from a source."""
    templates = MESSAGE_TEMPLATES.get(source, MESSAGE_TEMPLATES["event"])
    template = templates[index % len(templates)]
    senders = SENDERS.get(source, SENDERS["event"])
    sender = random.choice(senders)

    return {
        "id": f"msg-{uuid.uuid4().hex[:8]}",
        "source": source,
        "sender_name": sender,
        "subject": template["subject"],
        "body": template["body"],
        "urgency": template["urgency"],
        "best_action": template["best_action"],
        "consequences": template["consequences"].copy(),
        "ignore_penalty": template["ignore_penalty"].copy(),
        "decay_turns": template.get("decay_turns", 3),
    }


def generate_turn_messages(
    turn: int,
    difficulty: str,
    civ_state: Dict[str, Any],
    seed: int = None,
    memory_modifiers: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Generate messages for a turn based on difficulty and civilization state.
    More messages and higher urgency at higher difficulty.
    State-reactive: low food → food-related messages, high danger → defense messages.
    Memory-aware: ignored sources escalate, trusted sources are warmer.
    """
    if seed is not None:
        random.seed(seed + turn * 7)

    if memory_modifiers is None:
        memory_modifiers = {}

    messages = []

    # Base message count by difficulty
    if difficulty == "easy":
        count = random.choices([1, 2], weights=[60, 40])[0]
    elif difficulty == "medium":
        count = random.choices([2, 3], weights=[50, 50])[0]
    else:  # hard
        count = random.choices([2, 3, 4], weights=[20, 50, 30])[0]

    # Choose sources & templates based on civilization state + memory
    available_sources = list(MESSAGE_TEMPLATES.keys())
    weights = _compute_source_weights(civ_state, difficulty, memory_modifiers)

    # Tone modifiers from memory
    tone_mods = memory_modifiers.get("tone_modifiers", [])
    urgency_shifts = memory_modifiers.get("urgency_shift", {})

    urgency_escalation = {"low": "medium", "medium": "high", "high": "critical"}

    for _ in range(count):
        source = random.choices(available_sources, weights=weights)[0]
        templates = MESSAGE_TEMPLATES[source]
        idx = random.randint(0, len(templates) - 1)
        msg = _pick_message(source, idx)
        msg["turn_received"] = turn

        # Apply urgency escalation from memory
        shift = urgency_shifts.get(source, 0)
        if shift > 0:
            current_urgency = msg["urgency"]
            msg["urgency"] = urgency_escalation.get(current_urgency, current_urgency)

        # Append memory-aware context to message body
        if tone_mods:
            relevant = [t for t in tone_mods if source in t.lower()]
            if relevant:
                msg["body"] += f" ({relevant[0]})"

        messages.append(msg)

    return messages


def _compute_source_weights(
    civ_state: Dict[str, Any],
    difficulty: str,
    memory_modifiers: Dict[str, Any] = None,
) -> List[float]:
    """
    Weight message sources based on current civilization state + memory.
    Low food → more citizen/worker messages.
    High danger → more defense messages.
    Memory: ignored sources become louder.
    """
    if memory_modifiers is None:
        memory_modifiers = {}

    base_weights = {
        "citizen": 15, "scientist": 10, "worker": 12, "defense": 10,
        "trader": 10, "event": 10, "advisor": 8, "diplomat": 8,
    }

    food = civ_state.get("food", 200)
    morale = civ_state.get("morale", 60)
    danger = civ_state.get("danger_level", 10)
    tech = civ_state.get("technology_level", 1)

    # Adjust weights based on state
    if food < 80:
        base_weights["citizen"] += 15
        base_weights["worker"] += 10
    if morale < 30:
        base_weights["citizen"] += 20
        base_weights["advisor"] += 10
    if danger > 50:
        base_weights["defense"] += 25
        base_weights["diplomat"] += 10
    if tech < 3:
        base_weights["scientist"] += 8

    # Hard mode: more events and crises
    if difficulty == "hard":
        base_weights["event"] += 15
        base_weights["defense"] += 10

    # Memory-based adjustments (ignored sources get louder)
    extra = memory_modifiers.get("extra_weight", {})
    for source, bonus in extra.items():
        if source in base_weights:
            base_weights[source] += bonus

    sources = list(MESSAGE_TEMPLATES.keys())
    return [base_weights.get(s, 10) for s in sources]


# ═══════════════════════════════════════════════════════════════
# Pre-built Task Inboxes (for deterministic grading)
# ═══════════════════════════════════════════════════════════════

def generate_easy_scenario(seed: int = 42) -> List[List[Dict[str, Any]]]:
    """Pre-generate all turn messages for easy task (10 turns)."""
    random.seed(seed)
    turns = []
    state = {"food": 200, "morale": 60, "danger_level": 10, "technology_level": 1}
    for t in range(10):
        msgs = generate_turn_messages(t, "easy", state, seed=seed + t)
        turns.append(msgs)
    return turns


def generate_medium_scenario(seed: int = 42) -> List[List[Dict[str, Any]]]:
    """Pre-generate all turn messages for medium task (20 turns)."""
    random.seed(seed)
    turns = []
    state = {"food": 180, "morale": 55, "danger_level": 20, "technology_level": 1}
    for t in range(20):
        msgs = generate_turn_messages(t, "medium", state, seed=seed + t)
        turns.append(msgs)
    return turns


def generate_hard_scenario(seed: int = 42) -> List[List[Dict[str, Any]]]:
    """Pre-generate all turn messages for hard task (30 turns)."""
    random.seed(seed)
    turns = []
    state = {"food": 150, "morale": 45, "danger_level": 35, "technology_level": 1}
    for t in range(30):
        msgs = generate_turn_messages(t, "hard", state, seed=seed + t)
        turns.append(msgs)
    return turns
