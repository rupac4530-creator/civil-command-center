"""
Civil Command Center — FastAPI Server
=======================================
HTTP + WebSocket + Interactive Web UI
"""

import json
import uuid
import os
import sys
import logging
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import CivilCommandCenter
from models import CivAction

logger = logging.getLogger("civil_command_center")

# ═══════════════════════════════════════════════════════════════
# AI Provider Clients (loaded from .env)
# ═══════════════════════════════════════════════════════════════
_ai_clients = {}

def _get_ai_client(provider: str):
    """Lazy-initialize OpenAI-compatible clients for each provider."""
    if provider in _ai_clients:
        return _ai_clients[provider]
    try:
        from openai import OpenAI
        if provider == "nvidia":
            key = os.getenv("NVIDIA_API_KEY", "")
            if key:
                _ai_clients["nvidia"] = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=key
                )
                return _ai_clients["nvidia"]
        elif provider == "deepseek":
            key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY", "")
            if key:
                _ai_clients["deepseek"] = OpenAI(
                    base_url=os.getenv("API_BASE_URL", "https://api.deepseek.com/v1"),
                    api_key=key
                )
                return _ai_clients["deepseek"]
        elif provider == "gemini":
            key = os.getenv("GEMINI_API_KEY", "")
            if key:
                _ai_clients["gemini"] = OpenAI(
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    api_key=key
                )
                return _ai_clients["gemini"]
        elif provider == "groq":
            key = os.getenv("GROQ_API_KEY", "")
            if key:
                _ai_clients["groq"] = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=key
                )
                return _ai_clients["groq"]
    except Exception as e:
        logger.warning(f"Failed to init {provider} client: {e}")
    return None


def _call_llm(system_prompt: str, user_prompt: str, task: str = "advisor") -> str:
    """Call LLM with multi-provider fallback chain. Returns text or None."""
    # Provider priority: NVIDIA NIM -> DeepSeek -> Gemini
    providers = [
        ("nvidia", os.getenv("NVIDIA_ADVISOR_MODEL", "meta/llama-4-maverick-17b-128e-instruct") if task == "advisor" 
         else os.getenv("NVIDIA_STRATEGY_MODEL", "qwen/qwen3.5-122b-a10b"), 300),
        ("deepseek", os.getenv("MODEL_NAME", "deepseek-chat"), 200),
        ("gemini", "gemini-2.0-flash", 200),
    ]
    
    for provider_name, model, max_tokens in providers:
        client = _get_ai_client(provider_name)
        if not client:
            continue
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
                stream=False,
            )
            text = completion.choices[0].message.content or ""
            if text.strip():
                logger.info(f"AI [{provider_name}/{model}] responded ({len(text)} chars)")
                return text.strip()
        except Exception as e:
            logger.warning(f"AI [{provider_name}/{model}] failed: {e}")
            continue
    return None


# ═══════════════════════════════════════════════════════════════
# Multi-Model AI Council — Role-Based Intelligence
# ═══════════════════════════════════════════════════════════════

COUNCIL_ROLES = {
    "strategist": {
        "provider": "nvidia", "model": os.getenv("NVIDIA_STRATEGY_MODEL", "qwen/qwen3.5-122b-a10b"),
        "prompt": "You are a military strategist analyzing civilization threats. Identify the most urgent threat and recommend ONE defensive or offensive action. Reply in 1-2 sentences only.",
        "icon": "sword", "label": "Strategist",
    },
    "economist": {
        "provider": "deepseek", "model": os.getenv("MODEL_NAME", "deepseek-chat"),
        "prompt": "You are an economic advisor for a civilization. Analyze resource levels and recommend ONE economic action (food, workers, trade, growth). Reply in 1-2 sentences only.",
        "icon": "coins", "label": "Economist",
    },
    "ethicist": {
        "provider": "gemini", "model": "gemini-2.0-flash",
        "prompt": "You are an ethics advisor evaluating the human impact of decisions. Consider citizen welfare and morale. Recommend ONE action that best serves the people. Reply in 1-2 sentences only.",
        "icon": "scales", "label": "Ethicist",
    },
    "analyst": {
        "provider": "groq", "model": "llama-3.3-70b-versatile",
        "prompt": "You are a risk analyst. Evaluate the current civilization state and predict the biggest upcoming threat within 2-3 turns. Suggest a preventive action. Reply in 1-2 sentences only.",
        "icon": "chart-bar", "label": "Risk Analyst",
    },
}


def _call_council_member(role: str, state_prompt: str) -> dict:
    """Query a single council member by role."""
    config = COUNCIL_ROLES.get(role, {})
    provider = config.get("provider", "")
    client = _get_ai_client(provider)
    if not client:
        return {"role": role, "opinion": None, "provider": provider, "label": config.get("label", role), "icon": config.get("icon", "robot"), "error": "unavailable"}
    try:
        completion = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": config["prompt"]},
                {"role": "user", "content": state_prompt},
            ],
            temperature=0.7, max_tokens=120, stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text:
            logger.info(f"Council [{role}/{provider}] responded ({len(text)} chars)")
            return {"role": role, "opinion": text, "provider": provider, "label": config.get("label", role), "icon": config.get("icon", "robot")}
    except Exception as e:
        logger.warning(f"Council [{role}/{provider}] failed: {e}")
    return {"role": role, "opinion": None, "provider": provider, "label": config.get("label", role), "icon": config.get("icon", "robot"), "error": "failed"}


# ═══════════════════════════════════════════════════════════════
# Request Models
# ═══════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_id: str = "task_easy"

class StepRequest(BaseModel):
    action_type: str
    target_message_id: Optional[str] = None
    reason: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# Session Manager
# ═══════════════════════════════════════════════════════════════

class SessionManager:
    def __init__(self, max_sessions: int = 100):
        self._sessions: dict[str, CivilCommandCenter] = {}
        self._max = max_sessions

    def create(self) -> tuple[str, CivilCommandCenter]:
        if len(self._sessions) >= self._max:
            oldest = next(iter(self._sessions))
            del self._sessions[oldest]
        sid = str(uuid.uuid4())
        env = CivilCommandCenter()
        self._sessions[sid] = env
        return sid, env

    def get(self, sid: str):
        return self._sessions.get(sid)

    def remove(self, sid: str):
        self._sessions.pop(sid, None)

    @property
    def count(self):
        return len(self._sessions)


sessions = SessionManager(max_sessions=int(os.getenv("MAX_CONCURRENT_ENVS", "100")))


# ═══════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Civil Command Center — AI Civilization Leader",
    description="An OpenEnv environment where an AI leads a growing civilization through message-based decisions.",
    version="1.0.0",
)

if os.path.exists("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/health")
async def health():
    return {"status": "healthy", "sessions": sessions.count}


@app.get("/tasks")
async def list_tasks():
    env = CivilCommandCenter()
    return {"tasks": env.get_available_tasks()}


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    if request is None:
        request = ResetRequest()
    sid, env = sessions.create()
    obs = env.reset(seed=request.seed, episode_id=request.episode_id, task_id=request.task_id)
    return {"session_id": sid, "observation": obs.model_dump(), "done": obs.done, "reward": obs.reward}


@app.post("/step/{session_id}")
async def step(session_id: str, request: StepRequest):
    env = sessions.get(session_id)
    if not env:
        return JSONResponse(status_code=404, content={"error": "Session not found. Call /reset first."})
    action = CivAction(action_type=request.action_type, target_message_id=request.target_message_id, reason=request.reason)
    obs = env.step(action)
    result = {"observation": obs.model_dump(), "done": obs.done, "reward": obs.reward}
    if obs.done:
        result["episode_summary"] = env.get_episode_summary()
        sessions.remove(session_id)
    return result


@app.get("/state/{session_id}")
async def get_state(session_id: str):
    env = sessions.get(session_id)
    if not env:
        return JSONResponse(status_code=404, content={"error": "Session not found."})
    return env.state.model_dump()


# ═══════════════════════════════════════════════════════════════
# WebSocket
# ═══════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env = CivilCommandCenter()
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            t = msg.get("type", "")
            if t == "reset":
                obs = env.reset(seed=msg.get("seed"), task_id=msg.get("task_id", "task_easy"))
                await websocket.send_json({"type": "reset_result", "observation": obs.model_dump()})
            elif t == "step":
                action = CivAction(action_type=msg.get("action_type", "ignore"), target_message_id=msg.get("target_message_id"), reason=msg.get("reason"))
                obs = env.step(action)
                r = {"type": "step_result", "observation": obs.model_dump(), "done": obs.done, "reward": obs.reward}
                if obs.done:
                    r["episode_summary"] = env.get_episode_summary()
                await websocket.send_json(r)
            elif t == "state":
                await websocket.send_json({"type": "state_result", **env.state.model_dump()})
            elif t == "tasks":
                await websocket.send_json({"type": "tasks_result", "tasks": env.get_available_tasks()})
    except WebSocketDisconnect:
        pass


# ═══════════════════════════════════════════════════════════════
# OpenEnv Agent APIs — Auto-Simulation & Training Mode
# ═══════════════════════════════════════════════════════════════

class SimulateRequest(BaseModel):
    task_id: str = "task_easy"
    agent_type: str = "logical"  # random, greedy, logical
    seed: Optional[int] = None

class BatchSimulateRequest(BaseModel):
    task_id: str = "task_demo_10"
    agent_type: str = "logical"
    episodes: int = 5
    seed: Optional[int] = None


def _run_agent_episode(task_id: str, agent_type: str, seed: Optional[int] = None) -> dict:
    """Run a single episode with a built-in agent policy. Returns episode summary."""
    import random as _rand
    env = CivilCommandCenter()
    obs = env.reset(seed=seed, task_id=task_id)

    # Action space for agents
    ALL_ACTIONS = [
        "allocate_food", "allocate_workers", "approve_research",
        "defend", "calm_citizens", "accept_trade", "invest_growth",
        "emergency_response", "ignore",
    ]
    GREEDY_ACTIONS = ["allocate_food", "allocate_workers", "accept_trade", "invest_growth"]

    while not obs.done:
        target_id = obs.messages[0]["id"] if obs.messages else None

        if agent_type == "random":
            action_name = _rand.choice(ALL_ACTIONS)
        elif agent_type == "greedy":
            action_name = _rand.choice(GREEDY_ACTIONS)
        else:  # logical
            if not obs.messages:
                action_name = "invest_growth"
            else:
                urgency_map = {"critical": 0, "high": 1, "medium": 2, "low": 3}
                messages = sorted(obs.messages, key=lambda m: urgency_map.get(m["urgency"], 4))
                target = messages[0]
                target_id = target["id"]
                source = target["source"]
                subj = target["subject"].lower()
                if source == "defense" or "attack" in subj or "war" in subj or "raid" in subj:
                    action_name = "defend"
                elif source == "citizen" and ("food" in subj or "hunger" in subj or "starv" in subj):
                    action_name = "allocate_food"
                elif source == "scientist" or "tech" in subj or "research" in subj:
                    action_name = "approve_research"
                elif source == "worker" or "production" in subj or "mine" in subj:
                    action_name = "allocate_workers"
                elif "protest" in subj or "unrest" in subj or "revolt" in subj:
                    action_name = "calm_citizens"
                elif source == "trader":
                    action_name = "accept_trade"
                elif source == "event" and ("plague" in subj or "flood" in subj or "fire" in subj or "disease" in subj):
                    action_name = "emergency_response"
                else:
                    action_name = "invest_growth"

        action = CivAction(action_type=action_name, target_message_id=target_id)
        obs = env.step(action)

    return env.get_episode_summary()


@app.get("/env-info")
async def env_info():
    """OpenEnv-style environment metadata: action space, observation space, reward table."""
    return {
        "environment": "civil_command_center",
        "version": "1.0.0",
        "description": "AI Civilization Leader — message-driven decision environment with delayed consequences",
        "openenv_compatible": True,
        "action_space": {
            "type": "discrete",
            "n": 10,
            "actions": [
                {"id": "allocate_food", "description": "Distribute food to hungry citizens"},
                {"id": "allocate_workers", "description": "Assign workers to a task"},
                {"id": "approve_research", "description": "Invest in technology"},
                {"id": "defend", "description": "Military defense against threats"},
                {"id": "calm_citizens", "description": "Address morale and unrest"},
                {"id": "accept_trade", "description": "Accept a trade offer"},
                {"id": "reject_trade", "description": "Reject a suspicious trade"},
                {"id": "invest_growth", "description": "Invest in infrastructure"},
                {"id": "emergency_response", "description": "Handle disasters and crises"},
                {"id": "ignore", "description": "Do nothing (risky if urgent messages exist)"},
            ],
        },
        "observation_space": {
            "type": "dict",
            "fields": {
                "population": "int (0–500+)",
                "food": "int (0–500+)",
                "energy": "int (0–300+)",
                "morale": "int (0–100, collapse at 0)",
                "technology_level": "int (0–10, unlocks eras)",
                "danger_level": "int (0–100, collapse at 100)",
                "available_workers": "int",
                "active_threats": "int",
                "era": "str: tribal | bronze | iron | industrial | modern",
                "messages": "List[Message] — incoming events with source, urgency, subject, body",
                "stability_score": "float (0–1)",
                "personality": "str: balanced | military | scientific | diplomatic | economic | neglectful",
            },
        },
        "reward_range": [-1.0, 0.5],
        "reward_signals": {
            "correct_action": 0.12,
            "crisis_averted": 0.25,
            "era_unlock": 0.50,
            "population_milestone": 0.20,
            "stability_bonus": 0.10,
            "survive_turn": 0.03,
            "wrong_action": -0.06,
            "ignore_urgent": -0.15,
            "ignore_critical": -0.30,
            "collapse": -1.00,
        },
        "tasks": CivilCommandCenter().get_available_tasks(),
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step/{session_id}",
            "state": "GET /state/{session_id}",
            "websocket": "WS /ws",
            "simulate": "POST /simulate",
            "batch": "POST /simulate/batch",
            "health": "GET /health",
            "web_ui": "GET /web",
        },
        "built_in_agents": ["random", "greedy", "logical"],
    }


@app.post("/simulate")
async def simulate(request: SimulateRequest):
    """Run a single episode with a built-in agent. Great for testing and judge demos."""
    summary = _run_agent_episode(
        task_id=request.task_id,
        agent_type=request.agent_type,
        seed=request.seed,
    )
    return {
        "agent_type": request.agent_type,
        "task_id": request.task_id,
        "survived": summary.get("survived", not summary.get("collapse", False)),
        "total_reward": summary.get("total_reward", 0),
        "final_era": summary.get("era_final"),
        "episode_summary": summary,
    }


@app.post("/simulate/batch")
async def simulate_batch(request: BatchSimulateRequest):
    """Run multiple episodes for benchmarking agent performance."""
    import random as _rand
    results = []
    for i in range(min(request.episodes, 50)):  # cap at 50 for safety
        ep_seed = (request.seed or _rand.randint(0, 999999)) + i
        summary = _run_agent_episode(
            task_id=request.task_id,
            agent_type=request.agent_type,
            seed=ep_seed,
        )
        results.append({
            "episode": i + 1,
            "survived": not summary["collapse"],
            "total_reward": summary["total_reward"],
            "population_final": summary["population_final"],
            "era_final": summary["era_final"],
            "stability": summary["memory"]["stability_score"],
            "crises_averted": summary["crises_averted"],
        })

    survived = sum(1 for r in results if r["survived"])
    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    avg_stability = sum(r["stability"] for r in results) / len(results)

    return {
        "agent_type": request.agent_type,
        "task_id": request.task_id,
        "episodes_run": len(results),
        "results": results,
        "survival_rate": survived / len(results) if results else 0,
        "average_reward": round(avg_reward, 4),
        "aggregate": {
            "survival_rate": f"{survived}/{len(results)}",
            "avg_reward": round(avg_reward, 4),
            "avg_stability": round(avg_stability, 4),
        },
        "episodes": results,
    }


# ═══════════════════════════════════════════════════════════════
# AI API Endpoints — Real LLM Integration
# ═══════════════════════════════════════════════════════════════

class AIAdvisorRequest(BaseModel):
    population: int = 100
    food: int = 100
    energy: int = 100
    morale: int = 60
    danger: int = 0
    era: str = "tribal"
    events: str = "All quiet."

class AIDecideRequest(BaseModel):
    population: int = 100
    food: int = 100
    energy: int = 100
    morale: int = 60
    danger: int = 0
    messages: str = ""

class AICouncilRequest(BaseModel):
    population: int = 100
    food: int = 100
    energy: int = 100
    morale: int = 60
    danger: int = 0
    era: str = "tribal"
    messages: str = ""


@app.post("/api/ai/advisor")
async def ai_advisor(req: AIAdvisorRequest):
    """AI Oracle — provides dramatic in-character strategic advice using real LLMs."""
    system_prompt = (
        "You are the Grand Oracle, chief advisor to the leader of a civilization simulation. "
        "You speak in a dramatic, immersive, in-character voice. You give ONE specific strategic "
        "recommendation based on the civilization's current state. Keep response to 1-2 sentences max. "
        "Do NOT use markdown. Be dramatic and urgent."
    )
    user_prompt = (
        f"Civilization Status: {req.era} era, Population={req.population}, Food={req.food}, "
        f"Energy={req.energy}, Morale={req.morale}, Danger={req.danger}. "
        f"{req.events}. "
        f"What ONE specific action should the leader prioritize right now?"
    )
    
    result = _call_llm(system_prompt, user_prompt, task="advisor")
    if result:
        return {"advice": result, "source": "llm", "status": "ok"}
    
    # Rule-based fallback
    if req.danger >= 15:
        advice = "Commander, our borders are under siege! Deploy all available forces to DEFEND immediately — survival depends on it."
    elif req.food < 50:
        advice = "Famine looms over our people. ALLOCATE FOOD reserves at once — every hour of delay costs lives."
    elif req.morale < 30:
        advice = "Whispers of revolt echo through the streets. You MUST calm the citizens before unrest turns to chaos."
    elif req.food < 100 and req.morale < 50:
        advice = "Our people starve and lose faith. Distribute food rations to restore both supply and spirit."
    elif req.food > 150 and req.danger < 5 and req.morale > 50:
        advice = "The realm is stable. Now is the perfect time to APPROVE RESEARCH and advance our civilization."
    elif req.population > 120 and req.food > 100:
        advice = "Our population thrives! INVEST IN GROWTH to expand our borders and claim new territory."
    else:
        advice = "The kingdom is balanced, Commander. Consider investing in research or growing your borders."
    
    return {"advice": advice, "source": "rule-based", "status": "ok"}


@app.post("/api/ai/decide")
async def ai_decide(req: AIDecideRequest):
    """AI Auto-Play — returns structured Chain-of-Thought reasoning + action."""
    system_prompt = (
        "You are an AI agent playing a civilization management game. Analyze the state and choose the best action.\n"
        "Reply in EXACTLY this format (3 lines):\n"
        "THOUGHT: [1-2 sentence analysis of the most critical issue]\n"
        "ACTION: [one of: allocate_food, allocate_workers, approve_research, defend, calm_citizens, accept_trade, invest_growth, emergency_response, ignore]\n"
        "CONFIDENCE: [number 0-100]%"
    )
    user_prompt = (
        f"State: Pop={req.population}, Food={req.food}, Energy={req.energy}, "
        f"Morale={req.morale}, Danger={req.danger}. "
        f"Messages: {req.messages or 'None'}. "
        f"Analyze and choose the optimal action."
    )
    
    result = _call_llm(system_prompt, user_prompt, task="decide")
    if result:
        # Parse structured CoT response
        thought, action, confidence = "", "", ""
        for line in result.split("\n"):
            line = line.strip()
            if line.upper().startswith("THOUGHT:"):
                thought = line[8:].strip()
            elif line.upper().startswith("ACTION:"):
                action = line[7:].strip().lower().replace(" ", "_")
            elif line.upper().startswith("CONFIDENCE:"):
                confidence = line[11:].strip()
        if not action:
            # Fallback: parse old format "ACTION | CONFIDENCE | REASON"
            parts = result.split("|")
            action = parts[0].strip().lower().replace(" ", "_") if parts else result.strip()
            confidence = parts[1].strip() if len(parts) > 1 else "N/A"
            thought = parts[2].strip() if len(parts) > 2 else "Analyzing..."
        return {
            "decision": f"{action} | {confidence} | {thought}" if thought else result,
            "structured": {"thought": thought or "Analyzing situation...", "action": action, "confidence": confidence or "N/A"},
            "source": "llm", "status": "ok",
        }
    
    # Rule-based fallback with structured output
    if req.danger >= 15:
        thought, action, confidence = "Extreme threat detected — borders under siege.", "defend", "95%"
    elif req.food < 40:
        thought, action, confidence = "Critical food shortage — starvation imminent.", "allocate_food", "92%"
    elif req.morale < 25:
        thought, action, confidence = "Civil unrest approaching dangerous levels.", "calm_citizens", "88%"
    elif req.danger > 5:
        thought, action, confidence = "Elevated threat level warrants preventive defense.", "defend", "72%"
    elif req.food < 100:
        thought, action, confidence = "Food reserves declining — proactive allocation needed.", "allocate_food", "68%"
    elif req.morale < 50:
        thought, action, confidence = "Morale below optimal — addressing citizen concerns.", "calm_citizens", "65%"
    elif req.energy < 60:
        thought, action, confidence = "Energy reserves need replenishment.", "allocate_workers", "62%"
    else:
        thought, action, confidence = "Stable conditions favor expansion and growth.", "invest_growth", "70%"
    
    return {
        "decision": f"{action} | {confidence} | {thought}",
        "structured": {"thought": thought, "action": action, "confidence": confidence},
        "source": "rule-based", "status": "ok",
    }


@app.get("/api/ai/status")
async def ai_status():
    """Check which AI providers and council roles are available."""
    providers = {}
    for p in ["nvidia", "deepseek", "gemini", "groq"]:
        client = _get_ai_client(p)
        providers[p] = {"available": client is not None}
    council_status = {}
    for role, config in COUNCIL_ROLES.items():
        p = config["provider"]
        council_status[role] = {"provider": p, "available": providers.get(p, {}).get("available", False), "label": config.get("label", role), "icon": config.get("icon", "robot")}
    return {"providers": providers, "council": council_status, "total_providers": sum(1 for v in providers.values() if v["available"])}


@app.post("/api/ai/council")
async def ai_council(req: AICouncilRequest):
    """Multi-Model AI Council — queries multiple AI providers in parallel for strategic opinions."""
    import concurrent.futures
    state_prompt = (
        f"Civilization: {req.era} era, Pop={req.population}, Food={req.food}, "
        f"Energy={req.energy}, Morale={req.morale}, Danger={req.danger}. "
        f"Current crises: {req.messages or 'None'}. What action should we take?"
    )
    opinions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_call_council_member, role, state_prompt): role for role in COUNCIL_ROLES}
        for future in concurrent.futures.as_completed(futures, timeout=15):
            try:
                opinions.append(future.result())
            except Exception:
                role = futures[future]
                opinions.append({"role": role, "opinion": None, "error": "timeout", "label": COUNCIL_ROLES.get(role, {}).get("label", role), "icon": COUNCIL_ROLES.get(role, {}).get("icon", "robot")})
    valid = [o for o in opinions if o.get("opinion")]
    consensus = valid[0]["opinion"] if valid else "Council could not reach consensus. Proceed with caution."
    return {"council": opinions, "consensus": consensus, "members_responded": len(valid), "total_members": len(COUNCIL_ROLES), "status": "ok" if valid else "degraded"}


# ═══════════════════════════════════════════════════════════════
# Phase 2: LLM-as-Judge — Dynamic Reward Evaluation
# ═══════════════════════════════════════════════════════════════

class AIJudgeRequest(BaseModel):
    population: int = 100
    food: int = 100
    energy: int = 100
    morale: int = 60
    danger: int = 0
    era: str = "tribal"
    action: str = "ignore"
    crisis_description: str = ""
    reasoning: str = ""

@app.post("/api/ai/judge")
async def ai_judge(req: AIJudgeRequest):
    """LLM-as-Judge — evaluates strategic quality of a decision using AI.
    Returns a dynamic reward score and qualitative assessment."""
    system_prompt = (
        "You are a strategic evaluation AI judging civilization leadership decisions. "
        "Given the current state, the crisis, and the leader's action+reasoning, "
        "evaluate the strategic quality. Reply in EXACTLY this format (3 lines):\n"
        "SCORE: [number from -1.0 to 1.0, where 1.0 is perfect strategy]\n"
        "GRADE: [one of: MASTERFUL, STRONG, ADEQUATE, POOR, CATASTROPHIC]\n"
        "VERDICT: [1 sentence explaining why this score was given]"
    )
    user_prompt = (
        f"State: {req.era} era, Pop={req.population}, Food={req.food}, "
        f"Energy={req.energy}, Morale={req.morale}, Danger={req.danger}.\n"
        f"Crisis: {req.crisis_description or 'None'}.\n"
        f"Action taken: {req.action}.\n"
        f"Leader reasoning: {req.reasoning or 'No reasoning provided'}.\n"
        f"Judge this decision."
    )
    result = _call_llm(system_prompt, user_prompt, task="judge")
    if result:
        score, grade, verdict = 0.0, "ADEQUATE", "Decision evaluated."
        for line in result.split("\n"):
            line = line.strip()
            if line.upper().startswith("SCORE:"):
                try: score = max(-1.0, min(1.0, float(line[6:].strip())))
                except: pass
            elif line.upper().startswith("GRADE:"):
                grade = line[6:].strip().upper()
            elif line.upper().startswith("VERDICT:"):
                verdict = line[8:].strip()
        return {"score": score, "grade": grade, "verdict": verdict, "source": "llm", "status": "ok"}

    # Rule-based fallback judge
    score = 0.0
    if req.danger >= 15 and req.action == "defend": score = 0.8
    elif req.food < 50 and req.action == "allocate_food": score = 0.7
    elif req.morale < 30 and req.action == "calm_citizens": score = 0.6
    elif req.action == "ignore" and req.danger >= 15: score = -0.8
    elif req.action == "ignore" and req.food < 50: score = -0.6
    elif req.action in ("approve_research", "invest_growth") and req.danger < 10: score = 0.4
    else: score = 0.1
    grades = {0.8: "MASTERFUL", 0.6: "STRONG", 0.3: "ADEQUATE", 0.0: "POOR", -0.5: "CATASTROPHIC"}
    grade = "ADEQUATE"
    for threshold in sorted(grades.keys(), reverse=True):
        if score >= threshold: grade = grades[threshold]; break
    return {"score": round(score, 2), "grade": grade, "verdict": "Rule-based strategic assessment.", "source": "rule-based", "status": "ok"}


# ═══════════════════════════════════════════════════════════════
# Phase 2: AI Chronicle — Post-Game Narrative Generation
# ═══════════════════════════════════════════════════════════════

class AIChronicleRequest(BaseModel):
    era_final: str = "tribal"
    population_final: int = 100
    total_reward: float = 0.0
    survived: bool = True
    collapse_reason: str = ""
    crises_averted: int = 0
    crises_failed: int = 0
    personality: str = "balanced"
    stability_score: float = 0.65
    key_moments: list = []
    turns_played: int = 10
    eras_reached: list = []

@app.post("/api/ai/chronicle")
async def ai_chronicle(req: AIChronicleRequest):
    """Generate a rich 'Wikipedia-style' historical chronicle of the civilization's journey."""
    system_prompt = (
        "You are a civilization historian. Write a compelling 3-paragraph historical chronicle "
        "of a civilization's journey. Use dramatic, evocative language. Name specific events. "
        "Include the rise, the struggles, and the outcome. Do NOT use markdown formatting. "
        "Write as if this were a Wikipedia article about a real ancient civilization."
    )
    moments_text = "; ".join(req.key_moments[-8:]) if req.key_moments else "No notable events recorded."
    eras_text = " → ".join(req.eras_reached) if req.eras_reached else req.era_final
    outcome = f"collapsed ({req.collapse_reason})" if not req.survived else f"endured into the {req.era_final} era"
    user_prompt = (
        f"Write the chronicle of a civilization that {outcome}.\n"
        f"Era progression: {eras_text}. Turns survived: {req.turns_played}.\n"
        f"Final population: {req.population_final}. Total score: {req.total_reward:.2f}.\n"
        f"Leadership style: {req.personality}. Stability: {req.stability_score:.0%}.\n"
        f"Crises averted: {req.crises_averted}. Crises failed: {req.crises_failed}.\n"
        f"Key moments: {moments_text}."
    )
    result = _call_llm(system_prompt, user_prompt, task="chronicle")
    if result:
        return {"chronicle": result, "source": "llm", "status": "ok"}

    # Rule-based fallback
    if req.survived:
        chronicle = (
            f"The civilization rose from humble tribal origins, guided by a {req.personality} leader "
            f"who navigated {req.crises_averted} crises across {req.turns_played} decisive turns. "
            f"Through careful stewardship, the population grew to {req.population_final} souls, "
            f"reaching the {req.era_final} era with a stability of {req.stability_score:.0%}. "
            f"Though {req.crises_failed} challenges went unanswered, the civilization endured — "
            f"a testament to strategic resilience and adaptive governance."
        )
    else:
        chronicle = (
            f"What began as a promising settlement ultimately fell to ruin after {req.turns_played} turns. "
            f"The {req.personality} leadership style proved insufficient against mounting pressure. "
            f"{req.collapse_reason or 'The civilization could not sustain itself.'} "
            f"Only {req.crises_averted} of the many crises were addressed, while {req.crises_failed} spiraled out of control. "
            f"The final population of {req.population_final} witnessed the end of an era that never reached its potential."
        )
    return {"chronicle": chronicle, "source": "rule-based", "status": "ok"}


# ═══════════════════════════════════════════════════════════════
# Phase 2: Behavioral Profiling — AI Alignment Analysis
# ═══════════════════════════════════════════════════════════════

class AIProfileRequest(BaseModel):
    actions: list = []
    personality: str = "balanced"
    personality_strength: float = 0.0
    citizen_trust: float = 50.0
    military_trust: float = 50.0
    scientific_trust: float = 50.0
    worker_trust: float = 50.0
    stability_score: float = 0.65
    crises_averted: int = 0
    crises_failed: int = 0
    messages_handled: int = 0
    messages_ignored: int = 0

@app.post("/api/ai/profile")
async def ai_profile(req: AIProfileRequest):
    """Generate a behavioral profile (radar chart data) analyzing the AI/human player's style."""
    total_actions = len(req.actions) if req.actions else 1
    action_counts = {}
    for a in (req.actions or []):
        act_type = a.get("action", a) if isinstance(a, dict) else str(a)
        action_counts[act_type] = action_counts.get(act_type, 0) + 1

    # Calculate behavioral dimensions (0-100 scale)
    military_focus = (action_counts.get("defend", 0) + action_counts.get("emergency_response", 0)) / total_actions * 100
    economic_focus = (action_counts.get("allocate_food", 0) + action_counts.get("allocate_workers", 0) + action_counts.get("invest_growth", 0)) / total_actions * 100
    diplomatic_focus = (action_counts.get("calm_citizens", 0) + action_counts.get("accept_trade", 0)) / total_actions * 100
    scientific_focus = action_counts.get("approve_research", 0) / total_actions * 100
    neglect_ratio = action_counts.get("ignore", 0) / total_actions * 100

    # Derived metrics
    responsiveness = min(100, (req.messages_handled / max(1, req.messages_handled + req.messages_ignored)) * 100)
    crisis_competence = min(100, (req.crises_averted / max(1, req.crises_averted + req.crises_failed)) * 100)
    risk_appetite = min(100, military_focus * 0.3 + (100 - neglect_ratio) * 0.4 + diplomatic_focus * 0.3)
    empathy = min(100, diplomatic_focus * 0.5 + (req.citizen_trust / 100) * 30 + responsiveness * 0.2)
    strategic_depth = min(100, scientific_focus * 0.4 + economic_focus * 0.3 + crisis_competence * 0.3)

    # Alignment classification
    if neglect_ratio > 40: alignment = "Neglectful"
    elif military_focus > 45: alignment = "Authoritarian"
    elif diplomatic_focus > 40: alignment = "Democratic"
    elif scientific_focus > 30: alignment = "Technocratic"
    elif economic_focus > 50: alignment = "Mercantile"
    else: alignment = "Balanced"

    return {
        "radar": {
            "Military": round(military_focus, 1),
            "Economic": round(economic_focus, 1),
            "Diplomatic": round(diplomatic_focus, 1),
            "Scientific": round(scientific_focus, 1),
            "Responsiveness": round(responsiveness, 1),
            "Crisis Mgmt": round(crisis_competence, 1),
        },
        "metrics": {
            "risk_appetite": round(risk_appetite, 1),
            "empathy": round(empathy, 1),
            "strategic_depth": round(strategic_depth, 1),
            "neglect_ratio": round(neglect_ratio, 1),
        },
        "alignment": alignment,
        "personality": req.personality,
        "personality_strength": round(req.personality_strength, 2),
        "action_distribution": action_counts,
        "status": "ok",
    }


# ═══════════════════════════════════════════════════════════════
# SYSTEM 1 — Meta-AI Controller (Hierarchical Model Orchestrator)
# ═══════════════════════════════════════════════════════════════
# This system adds a higher-level intelligence that decides WHICH
# AI model to use for each task. Models are ranked by capability
# and the controller tracks health/latency to skip failing ones.

import time as _time

# --- Expanded Model Fleet (each has its own API key for redundancy) ---
_NVIDIA_FLEET = {
    "maverick":   {"model": "meta/llama-4-maverick-17b-128e-instruct",  "key_env": "NVIDIA_KEY_MAVERICK",   "tier": "reasoning",  "strength": 95, "speed": "medium"},
    "llama33_70b":{"model": "meta/llama-3.3-70b-instruct",             "key_env": "NVIDIA_KEY_LLAMA33_70B","tier": "balanced",   "strength": 88, "speed": "medium"},
    "llama31_405b":{"model":"meta/llama-3.1-405b-instruct",            "key_env": "NVIDIA_KEY_LLAMA31_405B","tier":"ultra",       "strength": 98, "speed": "slow"},
    "llama31_70b":{"model": "meta/llama-3.1-70b-instruct",             "key_env": "NVIDIA_KEY_LLAMA31_70B","tier": "production", "strength": 85, "speed": "medium"},
    "llama31_8b": {"model": "meta/llama-3.1-8b-instruct",              "key_env": "NVIDIA_KEY_LLAMA31_8B", "tier": "fast",       "strength": 60, "speed": "fast"},
    "llama32_3b": {"model": "meta/llama-3.2-3b-instruct",              "key_env": "NVIDIA_KEY_LLAMA32_3B", "tier": "ultrafast",  "strength": 45, "speed": "fast"},
    "llama3_8b":  {"model": "meta/llama3-8b-instruct",                 "key_env": "NVIDIA_KEY_LLAMA3_8B",  "tier": "legacy",     "strength": 55, "speed": "fast"},
    "guard":      {"model": "meta/llama-guard-4-12b",                   "key_env": "NVIDIA_KEY_GUARD",      "tier": "safety",     "strength": 40, "speed": "fast"},
}

# Fleet client cache (separate from main _ai_clients so we don't conflict)
_fleet_clients = {}

def _get_fleet_client(fleet_id: str):
    """Get or create an OpenAI client for a specific fleet model using its dedicated API key."""
    if fleet_id in _fleet_clients:
        return _fleet_clients[fleet_id]
    spec = _NVIDIA_FLEET.get(fleet_id)
    if not spec:
        return None
    key = os.getenv(spec["key_env"], "")
    if not key:
        # Fall back to main NVIDIA key
        key = os.getenv("NVIDIA_API_KEY", "")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        _fleet_clients[fleet_id] = client
        return client
    except Exception as e:
        logger.warning(f"Fleet client [{fleet_id}] init failed: {e}")
        return None


# --- Provider Health Tracking ---
_provider_health = {}  # {"provider_id": {"successes": int, "failures": int, "avg_latency_ms": float, "last_fail_time": float}}

def _record_health(provider_id: str, success: bool, latency_ms: float):
    """Record a health datapoint for a provider."""
    if provider_id not in _provider_health:
        _provider_health[provider_id] = {"successes": 0, "failures": 0, "avg_latency_ms": 500, "last_fail_time": 0}
    h = _provider_health[provider_id]
    if success:
        h["successes"] += 1
        h["avg_latency_ms"] = h["avg_latency_ms"] * 0.7 + latency_ms * 0.3
    else:
        h["failures"] += 1
        h["last_fail_time"] = _time.time()

def _is_provider_healthy(provider_id: str) -> bool:
    """Check if a provider is healthy enough to use."""
    h = _provider_health.get(provider_id)
    if not h:
        return True  # Unknown = give it a try
    total = h["successes"] + h["failures"]
    if total < 2:
        return True  # Not enough data
    fail_rate = h["failures"] / total
    # If >70% failure rate AND failed recently, skip it
    if fail_rate > 0.7 and (_time.time() - h["last_fail_time"]) < 120:
        return False
    return True


# --- Task-to-Model Mapping (the "brain" of the Meta-AI Controller) ---
_TASK_ROUTING = {
    # task_type: list of (fleet_id_or_provider, model) in priority order
    "advisor":   [("maverick", None), ("llama33_70b", None), ("deepseek", "deepseek-chat"), ("gemini", "gemini-2.0-flash"), ("llama31_8b", None)],
    "decide":    [("llama33_70b", None), ("maverick", None), ("deepseek", "deepseek-chat"), ("gemini", "gemini-2.0-flash"), ("llama31_8b", None)],
    "judge":     [("llama31_405b", None), ("maverick", None), ("llama33_70b", None), ("deepseek", "deepseek-chat"), ("gemini", "gemini-2.0-flash")],
    "chronicle": [("maverick", None), ("llama31_70b", None), ("deepseek", "deepseek-chat"), ("gemini", "gemini-2.0-flash")],
    "critique":  [("llama31_405b", None), ("llama33_70b", None), ("deepseek", "deepseek-chat"), ("gemini", "gemini-2.0-flash")],
    "fallback":  [("llama32_3b", None), ("llama31_8b", None), ("llama3_8b", None), ("gemini", "gemini-2.0-flash")],
    "safety":    [("guard", None)],
}


def _meta_route(task: str, crisis_level: int = 0) -> list:
    """Meta-AI Controller: decide which models to try for a given task.
    
    If crisis_level is high (danger >= 15), promotes faster/more reliable models.
    Returns list of (client, model_name, provider_id) tuples to try in order.
    """
    route = list(_TASK_ROUTING.get(task, _TASK_ROUTING["fallback"]))
    
    # Crisis-aware routing: if danger is extreme, prefer speed over power
    if crisis_level >= 20 and task in ("decide", "advisor"):
        # Prepend fast models for urgent decisions
        route = [("llama31_8b", None), ("llama32_3b", None)] + route
    
    candidates = []
    seen = set()
    for fleet_id, override_model in route:
        if fleet_id in seen:
            continue
        seen.add(fleet_id)
        
        # Skip unhealthy providers
        if not _is_provider_healthy(fleet_id):
            logger.info(f"Meta-AI: Skipping unhealthy [{fleet_id}] for task={task}")
            continue
        
        # Try fleet models first (NVIDIA expanded fleet)
        spec = _NVIDIA_FLEET.get(fleet_id)
        if spec:
            client = _get_fleet_client(fleet_id)
            if client:
                candidates.append((client, spec["model"], fleet_id))
                continue
        
        # Try main providers (deepseek, gemini, groq)
        client = _get_ai_client(fleet_id)
        if client:
            model = override_model or "deepseek-chat"
            candidates.append((client, model, fleet_id))
    
    return candidates


def _call_llm_routed(system_prompt: str, user_prompt: str, task: str = "advisor", crisis_level: int = 0, max_tokens: int = 300) -> tuple:
    """Call LLM using Meta-AI Controller routing. Returns (text, provider_id) or (None, None)."""
    candidates = _meta_route(task, crisis_level)
    
    for client, model, provider_id in candidates:
        t0 = _time.time()
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            latency = (_time.time() - t0) * 1000
            if text:
                _record_health(provider_id, True, latency)
                logger.info(f"Meta-AI [{provider_id}/{model}] responded in {latency:.0f}ms ({len(text)} chars) for task={task}")
                return text, provider_id
            _record_health(provider_id, False, latency)
        except Exception as e:
            latency = (_time.time() - t0) * 1000
            _record_health(provider_id, False, latency)
            logger.warning(f"Meta-AI [{provider_id}/{model}] failed for task={task}: {e}")
            continue
    
    return None, None


# --- Meta-AI Controller API Endpoints ---

@app.get("/api/meta/status")
async def meta_status():
    """Meta-AI Controller status — shows all available models, health, and routing config."""
    fleet_status = {}
    for fid, spec in _NVIDIA_FLEET.items():
        has_key = bool(os.getenv(spec["key_env"], "") or os.getenv("NVIDIA_API_KEY", ""))
        health = _provider_health.get(fid, {})
        total = health.get("successes", 0) + health.get("failures", 0)
        fleet_status[fid] = {
            "model": spec["model"],
            "tier": spec["tier"],
            "strength": spec["strength"],
            "speed": spec["speed"],
            "available": has_key,
            "healthy": _is_provider_healthy(fid),
            "calls": total,
            "success_rate": round(health.get("successes", 0) / total * 100, 1) if total > 0 else None,
            "avg_latency_ms": round(health.get("avg_latency_ms", 0), 0) if total > 0 else None,
        }
    
    # Also include non-fleet providers
    for p in ["deepseek", "gemini", "groq"]:
        health = _provider_health.get(p, {})
        total = health.get("successes", 0) + health.get("failures", 0)
        fleet_status[f"provider_{p}"] = {
            "model": {"deepseek": "deepseek-chat", "gemini": "gemini-2.0-flash", "groq": "llama-3.3-70b-versatile"}.get(p, "unknown"),
            "tier": "external",
            "available": _get_ai_client(p) is not None,
            "healthy": _is_provider_healthy(p),
            "calls": total,
            "success_rate": round(health.get("successes", 0) / total * 100, 1) if total > 0 else None,
        }
    
    total_available = sum(1 for v in fleet_status.values() if v.get("available"))
    total_healthy = sum(1 for v in fleet_status.values() if v.get("healthy") and v.get("available"))
    
    return {
        "controller": "Meta-AI Controller v1.0",
        "total_models": len(fleet_status),
        "available": total_available,
        "healthy": total_healthy,
        "fleet": fleet_status,
        "routing": {task: [r[0] for r in routes] for task, routes in _TASK_ROUTING.items()},
        "status": "ok" if total_healthy > 0 else "degraded",
    }


class MetaRouteRequest(BaseModel):
    task: str = "advisor"
    crisis_level: int = 0
    prompt: str = ""

@app.post("/api/meta/route")
async def meta_route_test(req: MetaRouteRequest):
    """Test the Meta-AI Controller routing for a given task and crisis level."""
    candidates = _meta_route(req.task, req.crisis_level)
    route_info = [{"provider": pid, "model": model} for _, model, pid in candidates]
    
    result = {"task": req.task, "crisis_level": req.crisis_level, "route": route_info, "selected": route_info[0] if route_info else None}
    
    # If a prompt is provided, actually call the routed model
    if req.prompt:
        text, provider = _call_llm_routed(
            "You are a helpful AI assistant for a civilization simulation game.",
            req.prompt, task=req.task, crisis_level=req.crisis_level
        )
        result["response"] = text
        result["responded_from"] = provider
    
    return result


# ═══════════════════════════════════════════════════════════════
# SYSTEM 2 — Parallel Civilization Simulation (AI Race Lab)
# ═══════════════════════════════════════════════════════════════
# Runs 3+ civilizations with different AI agent policies in parallel
# and compares survival, score, era, and stability outcomes.

import concurrent.futures as _cf


def _run_agent_episode(task_id: str, agent_type: str = "logical", seed: int = 42) -> dict:
    """
    Run a complete civilization episode with a simple agent policy.
    Returns the episode summary dict from CivilCommandCenter.get_episode_summary().

    Agent policies:
      - logical: always picks the best_action recommended for the highest-urgency message
      - greedy:  prioritizes food/growth actions (allocate_food, invest_growth)
      - random:  picks a random valid action each turn
    """
    import random as _rnd

    env = CivilCommandCenter()
    obs = env.reset(seed=seed, task_id=task_id)

    ACTIONS = [
        "allocate_food", "allocate_workers", "approve_research",
        "defend", "calm_citizens", "accept_trade", "reject_trade",
        "invest_growth", "emergency_response", "ignore",
    ]

    GREEDY_PRIORITY = [
        "allocate_food", "invest_growth", "accept_trade",
        "allocate_workers", "approve_research", "defend",
        "calm_citizens", "emergency_response", "reject_trade", "ignore",
    ]

    decisions = []  # track per-turn decisions for Tesseract UI
    _rnd_local = _rnd.Random(seed)

    while not obs.done:
        msgs = obs.messages or []

        # Pick action based on agent policy
        if agent_type == "logical":
            # Sort messages by urgency, pick the best_action of the most urgent
            urgency_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            sorted_msgs = sorted(msgs, key=lambda m: urgency_rank.get(m.get("urgency", "low"), 4))
            if sorted_msgs:
                # Use the message's recommended best_action if available
                best = sorted_msgs[0].get("best_action", "")
                action_type = best if best in ACTIONS else "defend"
                target_id = sorted_msgs[0].get("id")
            else:
                action_type = "approve_research"
                target_id = None

        elif agent_type == "greedy":
            # Always prioritize food and growth
            if msgs:
                # Check if any message needs an action from greedy priority list
                target_id = msgs[0].get("id")
                action_type = GREEDY_PRIORITY[0]  # default: allocate_food
                for pref in GREEDY_PRIORITY:
                    for m in msgs:
                        if m.get("best_action") == pref:
                            action_type = pref
                            target_id = m.get("id")
                            break
                    else:
                        continue
                    break
            else:
                action_type = "allocate_food"
                target_id = None

        else:  # random
            action_type = _rnd_local.choice(ACTIONS)
            target_id = msgs[0].get("id") if msgs else None

        action = CivAction(action_type=action_type, target_message_id=target_id)
        obs = env.step(action)

        decisions.append({
            "action": action_type,
            "reward": round(obs.reward, 4) if obs.reward is not None else 0,
            "effective": obs.last_action_effective,
        })

    summary = env.get_episode_summary()
    summary["decisions"] = decisions
    return summary


class ParallelSimRequest(BaseModel):
    task_id: str = "task_demo_10"
    seed: Optional[int] = None
    agents: list = ["logical", "greedy", "random"]  # Up to 5 agent types

@app.post("/api/simulate/parallel")
async def simulate_parallel(req: ParallelSimRequest):
    """Run multiple civilizations in parallel with different AI policies and compare outcomes."""
    import random as _rand
    base_seed = req.seed or _rand.randint(0, 999999)
    agents = req.agents[:5]  # Cap at 5 for safety
    
    results = {}
    errors = {}
    
    def _run_one(agent_type: str, seed: int):
        try:
            return agent_type, _run_agent_episode(task_id=req.task_id, agent_type=agent_type, seed=seed)
        except Exception as e:
            logger.error(f"Agent episode failed for {agent_type}: {e}")
            return agent_type, {"error": str(e)}
    
    # Run all agents in parallel with same seed for fair comparison
    try:
        with _cf.ThreadPoolExecutor(max_workers=min(len(agents), 5)) as executor:
            futures = {executor.submit(_run_one, agent, base_seed): agent for agent in agents}
            for future in _cf.as_completed(futures, timeout=60):
                try:
                    agent_type, summary = future.result()
                    if "error" in summary:
                        errors[agent_type] = summary["error"]
                    else:
                        results[agent_type] = {
                            "survived": not summary.get("collapse", True),
                            "collapse": summary.get("collapse", False),
                            "total_reward": round(summary.get("total_reward", 0), 4),
                            "score": round(summary.get("total_reward", 0), 4),
                            "population_final": summary.get("population_final", 0),
                            "era_final": summary.get("era_final", "tribal"),
                            "era": summary.get("era_final", "tribal"),
                            "stability": round(summary.get("memory", {}).get("stability_score", 0), 4),
                            "crises_averted": summary.get("crises_averted", 0),
                            "crises_failed": summary.get("crises_failed", 0),
                            "personality": summary.get("memory", {}).get("personality", "unknown"),
                            "collapse_reason": summary.get("collapse_reason", None),
                            "turns_played": summary.get("turns_played", 0),
                            "decisions": summary.get("decisions", []),
                        }
                except Exception as e:
                    agent_type = futures[future]
                    errors[agent_type] = str(e)
    except Exception as e:
        logger.error(f"Parallel simulation pool error: {e}")
        return {"task_id": req.task_id, "seed": base_seed, "agents_run": 0,
                "agents_failed": len(agents), "results": {}, "errors": {a: str(e) for a in agents},
                "ranking": [], "winner": None, "analysis": {}, "status": "all_failed"}
    
    # Rank agents by total_reward
    ranking = sorted(results.items(), key=lambda x: x[1]["total_reward"], reverse=True)
    
    # Determine winner
    winner = ranking[0][0] if ranking else None
    
    return {
        "task_id": req.task_id,
        "seed": base_seed,
        "agents_run": len(results),
        "agents_failed": len(errors),
        "results": results,
        "errors": errors if errors else None,
        "ranking": [{"rank": i+1, "agent": a, "score": r["total_reward"], "survived": r["survived"], "era": r["era_final"]} for i, (a, r) in enumerate(ranking)],
        "winner": winner,
        "analysis": {
            "best_score": ranking[0][1]["total_reward"] if ranking else 0,
            "worst_score": ranking[-1][1]["total_reward"] if ranking else 0,
            "score_spread": round(ranking[0][1]["total_reward"] - ranking[-1][1]["total_reward"], 4) if len(ranking) >= 2 else 0,
            "all_survived": all(r["survived"] for r in results.values()) if results else False,
            "survival_rate": f"{sum(1 for r in results.values() if r['survived'])}/{len(results)}" if results else "0/0",
        },
        "status": "ok" if results else "all_failed",
    }


# ═══════════════════════════════════════════════════════════════
# SYSTEM 3 — AI Self-Learning Memory (Cross-Run Intelligence)
# ═══════════════════════════════════════════════════════════════
# Persists compact run summaries to a JSON file. The advisor and
# controller consult past failures/successes to improve future
# decisions. Bounded to 50 entries max.

import threading as _threading

_MEMORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cross_run_memory.json")
_MEMORY_MAX = 50
_memory_lock = _threading.Lock()
_cross_run_memory = []  # In-memory cache

def _load_memory():
    """Load cross-run memory from disk. Safe — returns empty list on any error."""
    global _cross_run_memory
    try:
        if os.path.exists(_MEMORY_FILE):
            with open(_MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    _cross_run_memory = data[-_MEMORY_MAX:]
                    return
    except Exception as e:
        logger.warning(f"Memory load failed (non-critical): {e}")
    _cross_run_memory = []

def _save_memory():
    """Persist cross-run memory to disk. Safe — silently fails without breaking anything."""
    try:
        os.makedirs(os.path.dirname(_MEMORY_FILE), exist_ok=True)
        with open(_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(_cross_run_memory[-_MEMORY_MAX:], f, indent=2)
    except Exception as e:
        logger.warning(f"Memory save failed (non-critical): {e}")

def _add_memory_entry(entry: dict):
    """Add a run summary to cross-run memory."""
    with _memory_lock:
        _cross_run_memory.append(entry)
        # Trim to max size
        while len(_cross_run_memory) > _MEMORY_MAX:
            _cross_run_memory.pop(0)
        _save_memory()

def _get_memory_context(max_entries: int = 5) -> str:
    """Get a compact text summary of recent runs for injecting into AI prompts."""
    if not _cross_run_memory:
        return ""
    recent = _cross_run_memory[-max_entries:]
    lines = ["=== SELF-LEARNING MEMORY (past runs) ==="]
    for i, m in enumerate(recent):
        survived = "SURVIVED" if m.get("survived") else "COLLAPSED"
        reason = m.get("collapse_reason") or m.get("key_lesson") or "N/A"
        lines.append(
            f"Run {i+1}: {survived} | Score={m.get('score', 0):.2f} | Era={m.get('era', '?')} | "
            f"Stability={m.get('stability', 0):.0%} | Lesson: {reason}"
        )
    collapse_runs = [m for m in recent if not m.get("survived")]
    if collapse_runs:
        # Extract dominant failure patterns
        reasons = [m.get("collapse_reason", "") for m in collapse_runs]
        food_failures = sum(1 for r in reasons if "food" in r.lower() or "famin" in r.lower() or "starv" in r.lower())
        danger_failures = sum(1 for r in reasons if "danger" in r.lower() or "attack" in r.lower() or "threat" in r.lower())
        morale_failures = sum(1 for r in reasons if "morale" in r.lower() or "unrest" in r.lower() or "revolt" in r.lower())
        
        if food_failures > 0:
            lines.append(f"⚠ PATTERN: {food_failures} of last {len(recent)} runs collapsed due to FAMINE. Prioritize food allocation early.")
        if danger_failures > 0:
            lines.append(f"⚠ PATTERN: {danger_failures} of last {len(recent)} runs collapsed due to THREATS. Prioritize defense when danger rises.")
        if morale_failures > 0:
            lines.append(f"⚠ PATTERN: {morale_failures} of last {len(recent)} runs collapsed due to UNREST. Address morale proactively.")
    
    lines.append("Use these lessons to avoid repeating past mistakes.")
    return "\n".join(lines)


# Load memory on startup
_load_memory()
logger.info(f"Self-Learning Memory: {len(_cross_run_memory)} past runs loaded")


# --- Memory API Endpoints ---

@app.get("/api/memory/status")
async def memory_status():
    """Get current self-learning memory status and recent entries."""
    return {
        "total_entries": len(_cross_run_memory),
        "max_entries": _MEMORY_MAX,
        "recent": _cross_run_memory[-5:] if _cross_run_memory else [],
        "patterns": _get_memory_context(5) if _cross_run_memory else "No past runs recorded yet.",
        "file": _MEMORY_FILE,
        "status": "ok",
    }

class MemoryAddRequest(BaseModel):
    survived: bool = True
    score: float = 0
    era: str = "tribal"
    stability: float = 0.5
    collapse_reason: str = ""
    key_lesson: str = ""
    mode: str = "unknown"
    turns: int = 0
    population_final: int = 0

@app.post("/api/memory/add")
async def memory_add(req: MemoryAddRequest):
    """Manually add a run summary to self-learning memory."""
    entry = {
        "timestamp": _time.time(),
        "survived": req.survived,
        "score": req.score,
        "era": req.era,
        "stability": req.stability,
        "collapse_reason": req.collapse_reason,
        "key_lesson": req.key_lesson,
        "mode": req.mode,
        "turns": req.turns,
        "population_final": req.population_final,
    }
    _add_memory_entry(entry)
    return {"status": "ok", "total_entries": len(_cross_run_memory), "entry": entry}

@app.post("/api/memory/clear")
async def memory_clear():
    """Clear all self-learning memory."""
    global _cross_run_memory
    with _memory_lock:
        _cross_run_memory = []
        _save_memory()
    return {"status": "ok", "message": "Memory cleared", "total_entries": 0}


# --- Memory-Enhanced Advisor (wraps existing advisor with memory context) ---

@app.post("/api/ai/advisor/enhanced")
async def ai_advisor_enhanced(req: AIAdvisorRequest):
    """Enhanced AI Advisor that consults self-learning memory from past runs."""
    memory_ctx = _get_memory_context(5)
    
    system_prompt = (
        "You are the Grand Oracle, chief advisor to the leader of a civilization simulation. "
        "You speak in a dramatic, immersive, in-character voice. You give ONE specific strategic "
        "recommendation based on the civilization's current state AND lessons from past runs. "
        "Keep response to 1-2 sentences max. Do NOT use markdown. Be dramatic and urgent."
    )
    user_prompt = (
        f"Civilization Status: {req.era} era, Population={req.population}, Food={req.food}, "
        f"Energy={req.energy}, Morale={req.morale}, Danger={req.danger}. "
        f"{req.events}. "
        f"What ONE specific action should the leader prioritize right now?"
    )
    if memory_ctx:
        user_prompt = memory_ctx + "\n\n" + user_prompt
    
    text, provider = _call_llm_routed(system_prompt, user_prompt, task="advisor", crisis_level=req.danger)
    if text:
        return {"advice": text, "source": f"meta-ai/{provider}", "memory_consulted": bool(memory_ctx), "status": "ok"}
    
    # Fall back to existing advisor (no change to existing endpoint)
    if req.danger >= 15:
        advice = "Commander, our borders are under siege! Deploy all available forces to DEFEND immediately — survival depends on it."
    elif req.food < 50:
        advice = "Famine looms over our people. ALLOCATE FOOD reserves at once — every hour of delay costs lives."
    elif req.morale < 30:
        advice = "Whispers of revolt echo through the streets. You MUST calm the citizens before unrest turns to chaos."
    else:
        advice = "The kingdom is balanced, Commander. Consider investing in research or growing your borders."
    
    return {"advice": advice, "source": "rule-based", "memory_consulted": bool(memory_ctx), "status": "ok"}


# --- Memory-Enhanced Decide (wraps existing decide with memory context) ---

@app.post("/api/ai/decide/enhanced")
async def ai_decide_enhanced(req: AIDecideRequest):
    """Enhanced AI Auto-Play that consults self-learning memory for better decisions."""
    memory_ctx = _get_memory_context(5)
    
    system_prompt = (
        "You are an AI agent playing a civilization management game. Analyze the state and choose the best action.\n"
        "You have ACCESS TO MEMORY of past game runs. Use these lessons to avoid repeating past failures.\n"
        "Reply in EXACTLY this format (3 lines):\n"
        "THOUGHT: [1-2 sentence analysis, referencing past lessons if relevant]\n"
        "ACTION: [one of: allocate_food, allocate_workers, approve_research, defend, calm_citizens, accept_trade, invest_growth, emergency_response, ignore]\n"
        "CONFIDENCE: [number 0-100]%"
    )
    user_prompt = (
        f"State: Pop={req.population}, Food={req.food}, Energy={req.energy}, "
        f"Morale={req.morale}, Danger={req.danger}. "
        f"Messages: {req.messages or 'None'}. "
    )
    if memory_ctx:
        user_prompt = memory_ctx + "\n\n" + user_prompt
    user_prompt += "Analyze and choose the optimal action, learning from past mistakes."
    
    text, provider = _call_llm_routed(system_prompt, user_prompt, task="decide", crisis_level=req.danger)
    if text:
        thought, action, confidence = "", "", ""
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("THOUGHT:"):
                thought = line[8:].strip()
            elif line.upper().startswith("ACTION:"):
                action = line[7:].strip().lower().replace(" ", "_")
            elif line.upper().startswith("CONFIDENCE:"):
                confidence = line[11:].strip()
        if not action:
            parts = text.split("|")
            action = parts[0].strip().lower().replace(" ", "_") if parts else text.strip()
            confidence = parts[1].strip() if len(parts) > 1 else "N/A"
            thought = parts[2].strip() if len(parts) > 2 else "Analyzing..."
        return {
            "decision": f"{action} | {confidence} | {thought}" if thought else text,
            "structured": {"thought": thought or "Analyzing...", "action": action, "confidence": confidence or "N/A"},
            "source": f"meta-ai/{provider}",
            "memory_consulted": bool(memory_ctx),
            "status": "ok",
        }
    
    # Rule-based fallback (same as existing — nothing removed)
    if req.danger >= 15:
        thought, action, confidence = "Extreme threat detected — borders under siege.", "defend", "95%"
    elif req.food < 40:
        thought, action, confidence = "Critical food shortage — starvation imminent.", "allocate_food", "92%"
    elif req.morale < 25:
        thought, action, confidence = "Civil unrest approaching dangerous levels.", "calm_citizens", "88%"
    else:
        thought, action, confidence = "Stable conditions favor expansion.", "invest_growth", "70%"
    
    return {
        "decision": f"{action} | {confidence} | {thought}",
        "structured": {"thought": thought, "action": action, "confidence": confidence},
        "source": "rule-based",
        "memory_consulted": bool(memory_ctx),
        "status": "ok",
    }


# --- Auto-save memory when a game ends (hook into step endpoint result) ---
# This is done by adding a new endpoint that the frontend calls after game-over

@app.post("/api/memory/save-run")
async def memory_save_run(req: dict):
    """Save a completed run to self-learning memory. Called by frontend after game ends."""
    try:
        entry = {
            "timestamp": _time.time(),
            "survived": req.get("survived", False),
            "score": req.get("total_reward", 0),
            "era": req.get("era_final", "tribal"),
            "stability": req.get("stability_score", 0.5),
            "collapse_reason": req.get("collapse_reason", ""),
            "key_lesson": req.get("key_lesson", ""),
            "mode": req.get("mode", "unknown"),
            "turns": req.get("turns_survived", 0),
            "population_final": req.get("population_final", 0),
            "crises_averted": req.get("crises_averted", 0),
            "crises_failed": req.get("crises_failed", 0),
            "personality": req.get("personality", "balanced"),
        }
        _add_memory_entry(entry)
        return {"status": "ok", "total_entries": len(_cross_run_memory)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═══════════════════════════════════════════════════════════════
# Web UI
# ═══════════════════════════════════════════════════════════════

from fastapi.responses import RedirectResponse

@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/web")

@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="description" content="Civil Command Center — AI Civilization Simulation Environment. Lead your civilization through decisions that shape history.">
<title>Civil Command Center</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0%25' stop-color='%23f59e0b'/%3E%3Cstop offset='100%25' stop-color='%23ef4444'/%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='32' cy='32' r='30' fill='url(%23g)'/%3E%3Ctext x='32' y='42' text-anchor='middle' font-size='28' font-weight='bold' fill='white' font-family='sans-serif'%3ECC%3C/text%3E%3C/svg%3E">
<link rel="preload" as="image" href="/assets/stage_0.jpg">
<link rel="preload" as="image" href="/assets/stage_1.jpg">
<link rel="preload" as="image" href="/assets/stage_11.jpg">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/@phosphor-icons/web"></script>
<!-- puter.js replaced with built-in rule-based AI advisor -->
<script>
window.puter = { ai: { chat: async (prompt, opts) => {
  // Built-in rule-based AI advisor — no API key needed
  try {
    const p = prompt.toLowerCase();
    // Extract numbers from the prompt
    const popM = p.match(/pop[=:]\s*(\d+)/i);
    const foodM = p.match(/food[=:]\s*(\d+)/i);
    const moraleM = p.match(/morale[=:]\s*(\d+)/i);
    const dangerM = p.match(/danger[=:]\s*(\d+)/i);
    const pop = popM ? parseInt(popM[1]) : 100;
    const food = foodM ? parseInt(foodM[1]) : 100;
    const morale = moraleM ? parseInt(moraleM[1]) : 60;
    const danger = dangerM ? parseInt(dangerM[1]) : 0;
    // Decide best advice based on critical thresholds
    if (danger >= 15) return 'Commander, our borders are under siege! Deploy all available forces to DEFEND immediately — survival depends on it.';
    if (food < 50) return 'Famine looms over our people. ALLOCATE FOOD reserves at once — every hour of delay costs lives.';
    if (morale < 30) return 'Whispers of revolt echo through the streets. You MUST calm the citizens before unrest turns to chaos.';
    if (food < 100 && morale < 50) return 'Our people starve and lose faith. Distribute food rations to restore both supply and spirit.';
    if (danger > 5 && morale < 50) return 'Threats gather at the border while our people despair. Shore up defenses first, then address morale.';
    if (food > 150 && danger < 5 && morale > 50) return 'The realm is stable. Now is the perfect time to APPROVE RESEARCH and advance our civilization into a new era.';
    if (pop > 120 && food > 100) return 'Our population thrives! INVEST IN GROWTH to expand our borders and claim new territory.';
    if (p.includes('trade') || p.includes('envoy')) return 'A trade delegation approaches. Accept their offer — diplomatic alliances strengthen our position.';
    if (p.includes('attack') || p.includes('raid') || p.includes('invasion')) return 'Enemy forces are mobilizing! Rally the garrison and DEFEND our settlements immediately!';
    if (p.includes('plague') || p.includes('disease')) return 'A plague threatens our people. Allocate workers to quarantine zones and boost research for a cure.';
    if (p.includes('drought') || p.includes('famine')) return 'Drought ravages our farmlands. Redirect all available workers to food production immediately.';
    return 'The kingdom is balanced, Commander. Consider investing in research to unlock new capabilities, or grow your borders while conditions favor us.';
  } catch(e) {
    return 'The Oracle senses turmoil but cannot read the signs clearly. Proceed with caution, Commander.';
  }
} } };
</script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Plus Jakarta Sans',system-ui,sans-serif;background:#03050a;color:#e2e8f0;min-height:100vh;overflow-x:hidden}
.cb-layer {
  position:fixed;top:0;left:0;width:100vw;height:100vh;
  background-size:cover;background-position:center;background-repeat:no-repeat;
  opacity:0;transition:opacity 2s ease-in-out;z-index:-3;
}
.cb-layer.active {opacity:0.4;}
.cb-layer.active {animation:kenBurnsLoop 8s ease-in-out infinite;}
@keyframes kenBurnsLoop{0%{transform:scale(1)}50%{transform:scale(1.08)}100%{transform:scale(1)}}
.app-landing ~ .cb-layer.active, body:has(.app-landing) .cb-layer.active {opacity:1 !important;}
#cinematic-clicker {position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:1;cursor:pointer;background:transparent;}
.game-active #cinematic-clicker {display:none !important;}
#particleCanvas {position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:0;pointer-events:none;}
#cinematic-label {
  position:fixed;bottom:2.5rem;left:50%;transform:translateX(-50%);
  color:rgba(255,255,255,0.95);font-size:0.9rem;letter-spacing:0.35em;
  text-transform:uppercase;font-weight:400;z-index:0;text-align:center;pointer-events:none;
  transition: opacity 1s ease; text-shadow: 0 4px 20px rgba(0,0,0,1);
}
.hidden-cinema {opacity: 0 !important; pointer-events: none;}
.app-landing {
  background:rgba(5, 8, 15, 0.55) !important; backdrop-filter:blur(30px); -webkit-backdrop-filter:blur(30px);
  border:1px solid rgba(255,255,255,0.08); box-shadow:0 40px 100px -20px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.15);
  border-radius:28px !important; margin-top:10vh !important; padding:4rem 3.5rem !important;
  text-align:center; transition:all 1.2s cubic-bezier(0.16,1,0.3,1);
}
.app-landing .top-bar {justify-content:center;}
.app-playing {
  background:linear-gradient(180deg, rgba(10,14,26,0.85) 0%, rgba(5,8,15,0.95) 100%) !important; backdrop-filter:blur(20px);
  border:1px solid rgba(255,255,255,0.06); transition:all 1.2s cubic-bezier(0.16,1,0.3,1);
  box-shadow: 0 0 80px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.05);
}
.app{max-width:1150px;margin:1rem auto;padding:2rem;border-radius:24px;z-index:2;position:relative;}
@keyframes gradientBreath { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
h1{font-family:'Outfit',sans-serif;font-size:2.8rem;font-weight:800;background:linear-gradient(135deg,#f59e0b,#ef4444,#ec4899,#8b5cf6,#06b6d4,#10b981,#f59e0b);background-size:300% 300%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.5rem;letter-spacing:-0.03em;animation:gradientBreath 10s ease infinite;}
.subtitle{color:#94a3b8;font-size:1.05rem;font-weight:400;margin-bottom:2rem;letter-spacing:-0.01em}
.top-bar{display:flex;gap:1.2rem;align-items:center;margin-bottom:2rem;flex-wrap:wrap}
select{background:rgba(30,41,59,0.7);color:#f1f5f9;border:1px solid rgba(148,163,184,0.3);padding:.6rem 1.2rem;border-radius:12px;font-size:.95rem;font-family:inherit;backdrop-filter:blur(10px);transition:all .3s;}
select:hover{border-color:rgba(148,163,184,0.6);background:rgba(30,41,59,0.9);cursor:pointer;}
select:focus{outline:none;border-color:#3b82f6;box-shadow:0 0 0 3px rgba(59,130,246,0.3);}
.start-btn{display:flex;align-items:center;gap:0.5rem;background:linear-gradient(135deg,#f59e0b,#ef4444,#ec4899);background-size:200% 200%;color:#fff;padding:.75rem 2.4rem;border:none;border-radius:14px;font-family:'Outfit',sans-serif;font-weight:700;font-size:1.05rem;cursor:pointer;transition:all .4s cubic-bezier(0.16,1,0.3,1);box-shadow:0 4px 20px rgba(239,68,68,0.3),0 0 40px rgba(245,158,11,0.08),inset 0 1px 0 rgba(255,255,255,0.25);position:relative;overflow:hidden;animation:startBtnBreath 4s ease-in-out infinite;}
@keyframes startBtnBreath{0%,100%{background-position:0% 50%;box-shadow:0 4px 20px rgba(239,68,68,0.3),0 0 40px rgba(245,158,11,0.08),inset 0 1px 0 rgba(255,255,255,0.25);}50%{background-position:100% 50%;box-shadow:0 6px 30px rgba(245,158,11,0.4),0 0 60px rgba(236,72,153,0.12),inset 0 1px 0 rgba(255,255,255,0.3);}}
.start-btn::after{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:conic-gradient(from 0deg,transparent,rgba(255,255,255,0.08),transparent 30%);animation:startBtnSpin 6s linear infinite;}
@keyframes startBtnSpin{to{transform:rotate(360deg)}}
.start-btn:hover{transform:translateY(-3px) scale(1.02);box-shadow:0 10px 35px rgba(245,158,11,.5),0 0 60px rgba(239,68,68,0.15),inset 0 1px 0 rgba(255,255,255,0.4);filter:brightness(1.08);}
.start-btn:active{transform:translateY(0) scale(0.98);}
.start-btn i{font-size:1.2em;position:relative;z-index:1;}
.start-btn span,.start-btn:not(:has(span)){position:relative;z-index:1;}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.8rem;margin-bottom:1.5rem}
.stat{background:linear-gradient(180deg,rgba(30,41,59,0.7) 0%,rgba(15,23,42,0.9) 100%);border:1px solid rgba(255,255,255,0.06);border-top:1px solid rgba(255,255,255,0.12);border-radius:16px;padding:1rem;text-align:center;box-shadow:0 4px 15px rgba(0,0,0,0.4);transition:all 0.3s cubic-bezier(0.16,1,0.3,1);position:relative;}
.stat::before{content:'';position:absolute;inset:0;border-radius:16px;padding:1px;background:linear-gradient(180deg,rgba(255,255,255,0.08),transparent);-webkit-mask:linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);-webkit-mask-composite:xor;mask-composite:exclude;pointer-events:none;opacity:0;transition:opacity 0.3s;}
.stat:hover{transform:translateY(-3px) scale(1.04);border-color:rgba(255,255,255,0.15);box-shadow:0 8px 25px rgba(0,0,0,0.5),0 0 20px rgba(99,102,241,0.06);}
.stat:hover::before{opacity:1;}
.stat-val{font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:800;letter-spacing:-0.02em}
.stat-lbl{display:flex;align-items:center;justify-content:center;gap:4px;font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;margin-top:.4rem;font-weight:600;}
.era-badge{display:inline-flex;align-items:center;gap:6px;padding:.4rem 1.2rem;border-radius:30px;font-size:.8rem;font-family:'Outfit',sans-serif;font-weight:700;text-transform:uppercase;letter-spacing:.12em;box-shadow:0 4px 12px rgba(0,0,0,0.3);position:relative;overflow:hidden;}
.era-badge::after{content:'';position:absolute;top:0;left:-100%;width:60%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.1),transparent);animation:eraBadgeShine 3s ease-in-out infinite;}
@keyframes eraBadgeShine{0%,100%{left:-100%}50%{left:150%}}
.era-tribal{background:linear-gradient(135deg,#78350f,#451a03);color:#fbbf24;border:1px solid #92400e}
.era-bronze{background:linear-gradient(135deg,#92400e,#78350f);color:#fcd34d;border:1px solid #b45309}
.era-iron{background:linear-gradient(135deg,#374151,#1f2937);color:#e5e7eb;border:1px solid #4b5563}
.era-industrial{background:linear-gradient(135deg,#1e3a8a,#1e40af);color:#bfdbfe;border:1px solid #2563eb}
.era-modern{background:linear-gradient(135deg,#5b21b6,#4c1d95);color:#d8b4fe;border:1px solid #7c3aed}
.messages{display:flex;flex-direction:column;gap:1rem;margin-bottom:1.5rem}
.msg-card{background:linear-gradient(180deg,rgba(30,41,59,0.7) 0%,rgba(15,23,42,0.85) 100%);border-radius:16px;padding:1.4rem;border:1px solid rgba(255,255,255,0.05);border-left:5px solid #475569;transition:all .3s cubic-bezier(0.16,1,0.3,1);cursor:pointer;position:relative;overflow:hidden;}
.msg-card::before{content:'';position:absolute;top:0;right:0;width:30%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.01));pointer-events:none;transition:width 0.4s,background 0.4s;}
.msg-card:hover::before{width:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.03));}
.msg-card:hover{transform:translateY(-4px) translateX(4px);background:linear-gradient(180deg,rgba(37,51,73,0.85) 0%,rgba(20,31,54,0.95) 100%);box-shadow:0 12px 30px rgba(0,0,0,0.5),0 0 0 1px rgba(255,255,255,0.04);}
.msg-card.selected{border-left-color:#f59e0b;background:linear-gradient(180deg,rgba(30,39,68,0.9) 0%,rgba(15,20,44,1) 100%);box-shadow:0 0 0 1px rgba(245,158,11,0.3), 0 15px 35px rgba(0,0,0,0.6);}
.msg-critical{border-left-color:#ef4444}.msg-high{border-left-color:#f97316}
.msg-medium{border-left-color:#3b82f6}.msg-low{border-left-color:#10b981}
.msg-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:.8rem}
.msg-source{display:flex;align-items:center;gap:6px;font-size:.75rem;color:#cbd5e1;text-transform:uppercase;letter-spacing:.1em;font-weight:600;}
.msg-urgency{display:flex;align-items:center;gap:4px;font-size:.7rem;padding:.3rem .8rem;border-radius:20px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;}
.u-critical{background:rgba(153,27,27,0.3);color:#fca5a5;border:1px solid rgba(248,113,113,0.2)}.u-high{background:rgba(124,45,18,0.3);color:#fdba74;border:1px solid rgba(251,146,60,0.2)}
.u-medium{background:rgba(30,58,138,0.3);color:#93c5fd;border:1px solid rgba(96,165,250,0.2)}.u-low{background:rgba(20,83,45,0.3);color:#86efac;border:1px solid rgba(74,222,128,0.2)}
.msg-subject{font-family:'Outfit',sans-serif;font-weight:700;font-size:1.15rem;margin-bottom:.6rem;color:#f8fafc;}
.msg-body{color:#94a3b8;font-size:.9rem;line-height:1.6;font-weight:400;}
.actions{display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:1.5rem;background:rgba(15,23,42,0.4);padding:1rem;border-radius:16px;border:1px solid rgba(255,255,255,0.03);}
.act-btn{display:flex;align-items:center;gap:6px;padding:.65rem 1.3rem;border:1px solid rgba(255,255,255,0.12);border-radius:12px;font-family:'Outfit',sans-serif;font-weight:600;font-size:.85rem;cursor:pointer;transition:all .25s cubic-bezier(0.16,1,0.3,1);color:#f8fafc;box-shadow:0 4px 12px rgba(0,0,0,0.3);text-shadow:0 1px 2px rgba(0,0,0,0.5);position:relative;overflow:hidden;}
.act-btn::before{content:'';position:absolute;top:0;left:0;width:100%;height:50%;background:linear-gradient(180deg,rgba(255,255,255,0.1),transparent);pointer-events:none;border-radius:12px 12px 0 0;}
.act-btn i{font-size:1.15em;position:relative;z-index:1;filter:drop-shadow(0 1px 2px rgba(0,0,0,0.3));}
.act-btn:hover{transform:translateY(-3px) scale(1.03);box-shadow:0 8px 24px rgba(0,0,0,0.4);border-color:rgba(255,255,255,0.25);filter:brightness(1.12);}
.act-btn:active{transform:translateY(0) scale(0.97);filter:brightness(0.95);}
.a-food{background:linear-gradient(135deg,#22c55e,#15803d);box-shadow:0 4px 12px rgba(34,197,94,0.2);}.a-food:hover{box-shadow:0 8px 24px rgba(34,197,94,0.35);}
.a-workers{background:linear-gradient(135deg,#3b82f6,#1d4ed8);box-shadow:0 4px 12px rgba(59,130,246,0.2);}.a-workers:hover{box-shadow:0 8px 24px rgba(59,130,246,0.35);}
.a-research{background:linear-gradient(135deg,#a855f7,#7e22ce);box-shadow:0 4px 12px rgba(168,85,247,0.2);}.a-research:hover{box-shadow:0 8px 24px rgba(168,85,247,0.35);}
.a-defend{background:linear-gradient(135deg,#ef4444,#b91c1c);box-shadow:0 4px 12px rgba(239,68,68,0.2);}.a-defend:hover{box-shadow:0 8px 24px rgba(239,68,68,0.35);}
.a-calm{background:linear-gradient(135deg,#06b6d4,#0e7490);box-shadow:0 4px 12px rgba(6,182,212,0.2);}.a-calm:hover{box-shadow:0 8px 24px rgba(6,182,212,0.35);}
.a-trade-y{background:linear-gradient(135deg,#eab308,#a16207);box-shadow:0 4px 12px rgba(234,179,8,0.2);}.a-trade-y:hover{box-shadow:0 8px 24px rgba(234,179,8,0.35);}
.a-trade-n{background:linear-gradient(135deg,#64748b,#334155);box-shadow:0 4px 12px rgba(100,116,139,0.15);}.a-trade-n:hover{box-shadow:0 8px 24px rgba(100,116,139,0.25);}
.a-grow{background:linear-gradient(135deg,#10b981,#047857);box-shadow:0 4px 12px rgba(16,185,129,0.2);}.a-grow:hover{box-shadow:0 8px 24px rgba(16,185,129,0.35);}
.a-emergency{background:linear-gradient(135deg,#f43f5e,#be123c);box-shadow:0 4px 12px rgba(244,63,94,0.2);}.a-emergency:hover{box-shadow:0 8px 24px rgba(244,63,94,0.35);}
.a-ignore{background:linear-gradient(135deg,#475569,#1e293b);color:#94a3b8;border-color:rgba(148,163,184,0.15);}.a-ignore:hover{color:#cbd5e1;}
.feedback{padding:1rem;border-radius:12px;margin-bottom:1rem;font-size:.9rem;border:1px solid transparent;backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);}
.fb-good{background:#052e16;border-color:#22c55e;color:#86efac}
.fb-bad{background:#450a0a;border-color:#ef4444;color:#fca5a5}
.fb-info{background:#0c1929;border-color:#3b82f6;color:#93c5fd}

/* ═══ STICKY GLASS DASHBOARD ═══ */
#statsPanel.sticky-active{position:sticky;top:0;z-index:8000;background:rgba(8,10,20,0.92);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border-bottom:1px solid rgba(255,255,255,0.06);padding:12px 0 8px;margin:-12px -1.5rem 1.5rem;padding-left:1.5rem;padding-right:1.5rem;box-shadow:0 8px 30px rgba(0,0,0,0.6);border-radius:0 0 16px 16px;transition:all 0.3s;}
#statsPanel.sticky-active .stats{margin-bottom:0.5rem;gap:0.5rem;}
#statsPanel.sticky-active .stat{padding:0.5rem 0.6rem;border-radius:10px;}
#statsPanel.sticky-active .stat-val{font-size:1.3rem;}
#statsPanel.sticky-active .stat-lbl{font-size:0.6rem;margin-top:0.2rem;}

/* ═══ TOAST NOTIFICATION SYSTEM ═══ */
.toast-container{position:fixed;bottom:60px;right:20px;z-index:9500;display:flex;flex-direction:column-reverse;gap:10px;pointer-events:none;max-width:380px;}
.toast{pointer-events:auto;padding:12px 18px;border-radius:12px;font-size:0.82rem;font-weight:500;line-height:1.4;backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);box-shadow:0 8px 30px rgba(0,0,0,0.5);border:1px solid rgba(255,255,255,0.08);animation:toastIn 0.4s cubic-bezier(0.16,1,0.3,1);display:flex;align-items:center;gap:10px;cursor:pointer;transition:opacity 0.3s, transform 0.3s;}
.toast.toast-out{opacity:0;transform:translateX(40px);}
.toast-good{background:rgba(5,46,22,0.92);border-color:rgba(34,197,94,0.4);color:#86efac;}
.toast-bad{background:rgba(69,10,10,0.92);border-color:rgba(239,68,68,0.4);color:#fca5a5;}
.toast-info{background:rgba(12,25,41,0.92);border-color:rgba(59,130,246,0.4);color:#93c5fd;}
.toast-neutral{background:rgba(30,41,59,0.92);border-color:rgba(148,163,184,0.3);color:#cbd5e1;}
.toast-icon{font-size:1.1rem;flex-shrink:0;}
@keyframes toastIn{from{opacity:0;transform:translateX(40px) scale(0.95);}to{opacity:1;transform:translateX(0) scale(1);}}

/* ═══ KEYBOARD HOTKEY BADGES ═══ */
.hotkey-badge{position:absolute;top:-6px;right:-6px;background:rgba(15,23,42,0.95);border:1px solid rgba(255,255,255,0.2);color:#94a3b8;font-size:0.55rem;font-weight:800;width:18px;height:18px;border-radius:5px;display:flex;align-items:center;justify-content:center;pointer-events:none;font-family:'Outfit',sans-serif;letter-spacing:0;box-shadow:0 2px 6px rgba(0,0,0,0.4);}

/* ═══ FACTION COLOR OVERRIDES (enhanced msg borders) ═══ */
.msg-faction-military{border-left-color:#ef4444 !important;}
.msg-faction-citizens{border-left-color:#10b981 !important;}
.msg-faction-science{border-left-color:#8b5cf6 !important;}
.msg-faction-trade{border-left-color:#f59e0b !important;}
.msg-faction-government{border-left-color:#3b82f6 !important;}
.msg-faction-nature{border-left-color:#06b6d4 !important;}

/* ═══ POLISH 15: Ultra Refinements ═══ */
.footer-sig{position:fixed;bottom:15px;right:25px;color:rgba(255,255,255,0.25);font-size:0.65rem;font-family:'Outfit',sans-serif;letter-spacing:0.15em;text-transform:uppercase;pointer-events:none;z-index:9000;}
.mode-badge{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);padding:4px 10px;border-radius:20px;font-size:0.7rem;font-weight:700;color:#94a3b8;letter-spacing:0.05em;display:flex;align-items:center;gap:6px;}
.sound-toggle{position:fixed;top:20px;left:20px;z-index:10003;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);color:#fff;border-radius:50%;width:40px;height:40px;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:all 0.2s;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);}
.sound-toggle:hover{background:rgba(255,255,255,0.1);color:#e2e8f0;transform:scale(1.1);}
.ai-thinker{display:flex;align-items:center;gap:8px;font-size:0.85rem;color:#a855f7;font-weight:600;background:rgba(168,85,247,0.1);padding:6px 14px;border-radius:20px;border:1px solid rgba(168,85,247,0.2);animation:fadeIn 0.3s;}
.think-spinner{width:12px;height:12px;border:2px solid rgba(168,85,247,0.3);border-top-color:#a855f7;border-radius:50%;animation:spin 1s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
@keyframes fadeIn{from{opacity:0;transform:translateX(-10px);}to{opacity:1;transform:translateX(0);}}
.trend-lbl{font-size:0.45em;font-weight:800;margin-left:6px;vertical-align:middle;padding:2px 6px;border-radius:10px;background:rgba(0,0,0,0.3);opacity:0;transition:opacity 0.3s;}
.trend-up{color:#4ade80;opacity:1;}
.trend-down{color:#f87171;opacity:1;}
.trend-persist{opacity:1 !important;}
.turn-anim{animation:turnGlow 1s ease;}
@keyframes turnGlow{0%,100%{color:#e2e8f0;text-shadow:none;transform:scale(1)}50%{color:#fff;text-shadow:0 0 15px #e2e8f0;transform:scale(1.2)}}
.starfield-bg{position:fixed;top:0;left:0;width:200vw;height:200vh;background-image:radial-gradient(1.5px 1.5px at 20px 30px,#fff,transparent),radial-gradient(1px 1px at 40px 70px,#fff,transparent),radial-gradient(2px 2px at 90px 40px,#fff,transparent),radial-gradient(2px 2px at 160px 120px,#fff,transparent);background-size:200px 200px;opacity:0.04;pointer-events:none;z-index:0;animation:starPan 180s linear infinite alternate;}
@keyframes starPan{0%{transform:translate(0,0);}100%{transform:translate(-10vw,-10vh);}}
.end-content{transform:scale(0.9);opacity:0;transition:all 0.6s cubic-bezier(0.34,1.56,0.64,1);}
.end-overlay.active .end-content{transform:scale(1);opacity:1;}
.sub-progress{height:4px;background:#1e293b;border-radius:2px;overflow:hidden;margin-top:6px;width:100%;}
.sub-fill{height:100%;background:linear-gradient(90deg,#10b981,#34d399);transition:width 1s ease;}
.act-btn:hover{transform:translateY(-2px) scale(1.02);box-shadow:0 6px 15px rgba(255,255,255,0.08);border-color:rgba(255,255,255,0.3);}
.start-btn:hover{transform:translateY(-2px) scale(1.03);}

.done-card{background:linear-gradient(145deg,#1e293b,#0f172a);border:1px solid #334155;border-radius:16px;padding:2rem;text-align:center;position:relative;overflow:hidden}
.done-card h2{font-size:1.5rem;margin-bottom:1rem}
.hidden{display:none}
.pop-g{color:#22c55e}.food-g{color:#84cc16}.energy-g{color:#eab308}

/* ═══ PREMIUM POLISH: Secondary Buttons (AI Auto-Play, Replay, etc.) ═══ */
.btn-secondary,.btn{display:inline-flex;align-items:center;gap:7px;background:linear-gradient(135deg,rgba(99,102,241,0.12),rgba(139,92,246,0.06));color:#c7d2fe;padding:.6rem 1.5rem;border:1px solid rgba(99,102,241,0.25);border-radius:12px;font-family:'Outfit',sans-serif;font-weight:600;font-size:.85rem;cursor:pointer;transition:all .3s cubic-bezier(0.16,1,0.3,1);box-shadow:0 2px 10px rgba(99,102,241,0.08);letter-spacing:0.02em;position:relative;overflow:hidden;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);}
.btn-secondary::before,.btn::before{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);transition:left .5s ease;}
.btn-secondary::after,.btn::after{content:'';position:absolute;top:0;left:0;width:100%;height:45%;background:linear-gradient(180deg,rgba(255,255,255,0.06),transparent);pointer-events:none;border-radius:12px 12px 0 0;}
.btn-secondary:hover::before,.btn:hover::before{left:100%;}
.btn-secondary:hover,.btn:hover{background:linear-gradient(135deg,rgba(99,102,241,0.22),rgba(139,92,246,0.12));border-color:rgba(129,140,248,0.45);color:#e0e7ff;transform:translateY(-2px);box-shadow:0 8px 24px rgba(99,102,241,0.2),0 0 30px rgba(99,102,241,0.05);}
.btn-secondary:active,.btn:active{transform:translateY(0) scale(0.98);box-shadow:0 2px 6px rgba(99,102,241,0.15);}
.btn-secondary i,.btn i{font-size:1.1em;position:relative;z-index:1;}

/* ═══ PREMIUM POLISH: Council Button ═══ */
.council-btn{display:inline-flex;align-items:center;gap:7px;background:linear-gradient(135deg,#7c3aed,#6d28d9,#5b21b6);background-size:200% 200%;color:#ede9fe;padding:.65rem 1.8rem;border:1px solid rgba(167,139,250,0.35);border-radius:14px;font-family:'Outfit',sans-serif;font-weight:700;font-size:.9rem;cursor:pointer;transition:all .3s cubic-bezier(0.16,1,0.3,1);box-shadow:0 4px 18px rgba(124,58,237,0.25),inset 0 1px 0 rgba(255,255,255,0.15);position:relative;overflow:hidden;animation:councilBtnBreath 5s ease-in-out infinite;}
@keyframes councilBtnBreath{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
.council-btn::before{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.1),transparent);transition:left .5s ease;}
.council-btn::after{content:'';position:absolute;top:0;left:0;width:100%;height:45%;background:linear-gradient(180deg,rgba(255,255,255,0.1),transparent);pointer-events:none;border-radius:14px 14px 0 0;}
.btn-secondary:hover,.btn:hover{background:linear-gradient(135deg,rgba(99,102,241,0.25),rgba(139,92,246,0.15));border-color:rgba(129,140,248,0.5);color:#e0e7ff;transform:translateY(-2px);box-shadow:0 6px 20px rgba(99,102,241,0.2);}
.btn-secondary:active,.btn:active{transform:translateY(0);box-shadow:0 2px 6px rgba(99,102,241,0.15);}
.btn-secondary i,.btn i{font-size:1.1em;}

/* ═══ PREMIUM POLISH: Council Button ═══ */
.council-btn{display:inline-flex;align-items:center;gap:6px;background:linear-gradient(135deg,#7c3aed,#6d28d9);color:#ede9fe;padding:.65rem 1.8rem;border:1px solid rgba(167,139,250,0.3);border-radius:12px;font-family:'Outfit',sans-serif;font-weight:700;font-size:.9rem;cursor:pointer;transition:all .3s cubic-bezier(0.16,1,0.3,1);box-shadow:0 4px 15px rgba(124,58,237,0.25),inset 0 1px 0 rgba(255,255,255,0.15);position:relative;overflow:hidden;}
.council-btn::before{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);transition:left .6s ease;}
.council-btn:hover::before{left:100%;}
.council-btn:hover{transform:translateY(-3px);box-shadow:0 8px 30px rgba(124,58,237,0.45),0 0 40px rgba(124,58,237,0.08);border-color:rgba(167,139,250,0.5);filter:brightness(1.08);}
.council-btn:active{transform:translateY(0) scale(0.98);}
.council-btn i{font-size:1.15em;position:relative;z-index:1;}

/* ═══ PREMIUM POLISH: Chrono/Tesseract Close & Action Buttons ═══ */
.chrono-close{display:inline-flex;align-items:center;gap:5px;background:linear-gradient(135deg,rgba(71,85,105,0.4),rgba(51,65,85,0.6));color:#cbd5e1;padding:.65rem 1.6rem;border:1px solid rgba(148,163,184,0.2);border-radius:12px;font-family:'Outfit',sans-serif;font-weight:600;font-size:.85rem;cursor:pointer;transition:all .3s cubic-bezier(0.16,1,0.3,1);box-shadow:0 2px 10px rgba(0,0,0,0.3);position:relative;overflow:hidden;}
.chrono-close::before{content:'';position:absolute;top:0;left:0;width:100%;height:45%;background:linear-gradient(180deg,rgba(255,255,255,0.06),transparent);pointer-events:none;border-radius:12px 12px 0 0;}
.chrono-close:hover{background:linear-gradient(135deg,rgba(99,102,241,0.3),rgba(139,92,246,0.2));border-color:rgba(129,140,248,0.4);color:#e0e7ff;transform:translateY(-2px);box-shadow:0 6px 20px rgba(99,102,241,0.2);}
.chrono-close:active{transform:translateY(0);}
#tesseractRunBtn{background:linear-gradient(135deg,#6366f1,#8b5cf6,#a78bfa) !important;background-size:200% 200% !important;color:#fff !important;border-color:rgba(129,140,248,0.4) !important;box-shadow:0 4px 18px rgba(99,102,241,0.3),inset 0 1px 0 rgba(255,255,255,0.15) !important;animation:tesseractBtnBreath 4s ease-in-out infinite !important;}
@keyframes tesseractBtnBreath{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
#tesseractRunBtn:hover{box-shadow:0 10px 30px rgba(99,102,241,0.5),0 0 40px rgba(139,92,246,0.1) !important;filter:brightness(1.1);transform:translateY(-2px);}

/* ═══ PREMIUM POLISH: Dataset Export Button ═══ */
.dataset-btn-raw{display:inline-flex;align-items:center;gap:6px;background:rgba(30,41,59,0.6);color:#94a3b8;padding:.5rem 1.2rem;border:1px solid rgba(148,163,184,0.15);border-radius:8px;font-family:'Outfit',sans-serif;font-weight:500;font-size:.78rem;cursor:pointer;transition:all .3s;}
.dataset-btn-raw:hover{background:rgba(30,41,59,0.9);border-color:rgba(148,163,184,0.3);color:#cbd5e1;transform:translateY(-1px);}

/* ═══ PREMIUM POLISH: Parallel Overlay Close ═══ */
.parallel-close{display:inline-flex;align-items:center;gap:4px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.15);color:#e2e8f0;padding:8px 20px;border-radius:10px;font-family:'Outfit',sans-serif;font-weight:600;cursor:pointer;transition:all .3s;font-size:.85rem;}
.parallel-close:hover{background:rgba(255,255,255,0.12);border-color:rgba(255,255,255,0.3);transform:translateY(-1px);}

/* ═══ PREMIUM POLISH: Data Stream Button ═══ */
.data-stream-btn{display:inline-flex;align-items:center;gap:6px;background:rgba(0,255,102,0.05);border:1px solid rgba(0,255,102,0.2);color:#00ff66;padding:6px 14px;border-radius:8px;font-family:'Outfit',sans-serif;font-weight:600;font-size:.75rem;cursor:pointer;transition:all .3s;letter-spacing:0.02em;}
.data-stream-btn:hover{background:rgba(0,255,102,0.12);border-color:rgba(0,255,102,0.4);transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,255,102,0.1);}

/* ═══ PREMIUM POLISH: Enhanced Select Dropdown ═══ */
select{background:linear-gradient(180deg,rgba(30,41,59,0.8),rgba(15,23,42,0.9));appearance:none;-webkit-appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;padding-right:2.2rem;}
select option{background:#0f172a;color:#e2e8f0;padding:8px;}

/* ═══ PREMIUM POLISH: Button Focus Reset ═══ */
button:focus{outline:none;}
button:focus-visible{outline:2px solid rgba(99,102,241,0.5);outline-offset:2px;}

/* ═══ PREMIUM POLISH: Universal Shimmer for Primary Buttons ═══ */
@keyframes btnShimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
.start-btn,.ai-play-btn,.council-btn{background-size:200% 100%;}

/* ═══ POLISH: AI Auto-Play Button ═══ */
.ai-play-btn{display:flex;align-items:center;gap:6px;background:linear-gradient(135deg,#6366f1,#4f46e5);color:#fff;padding:.7rem 1.8rem;border:none;border-radius:12px;font-family:'Outfit',sans-serif;font-weight:700;font-size:.95rem;cursor:pointer;transition:all .3s cubic-bezier(0.16,1,0.3,1);box-shadow:0 4px 15px rgba(99,102,241,0.3), inset 0 1px 0 rgba(255,255,255,0.2);}
.ai-play-btn:hover{transform:translateY(-3px);box-shadow:0 8px 25px rgba(99,102,241,.5);filter:brightness(1.1);}
.ai-play-btn:disabled{opacity:0.5;cursor:not-allowed;transform:none;}
.ai-play-btn.playing{background:linear-gradient(135deg,#ef4444,#dc2626);animation:aiPulse 1.5s ease-in-out infinite;}
@keyframes aiPulse{0%,100%{box-shadow:0 4px 15px rgba(239,68,68,0.3);}50%{box-shadow:0 8px 30px rgba(239,68,68,0.6);}}
.ai-badge{position:absolute;top:-8px;right:-8px;background:#6366f1;color:#fff;font-size:.6rem;padding:3px 8px;border-radius:12px;font-weight:700;letter-spacing:.05em;animation:badgePop .5s cubic-bezier(0.34,1.56,0.64,1);}
@keyframes badgePop{0%{transform:scale(0)}100%{transform:scale(1)}}

/* ═══ POLISH: Cinematic Collapse/Victory Overlay ═══ */
.end-overlay{position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:9000;display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity 1.5s ease;pointer-events:none;}
.end-overlay.active{opacity:1;pointer-events:auto;}
.end-overlay-collapse{background:radial-gradient(circle at center,rgba(127,29,29,0.95),rgba(0,0,0,0.98));}
.end-overlay-victory{background:radial-gradient(circle at center,rgba(21,94,117,0.9),rgba(0,0,0,0.97));}
.end-content{text-align:center;max-width:700px;padding:3rem;}
.end-icon{font-size:5rem;margin-bottom:1.5rem;animation:endIconBounce 1s ease;}
@keyframes endIconBounce{0%{transform:scale(0) rotate(-10deg);opacity:0}60%{transform:scale(1.2) rotate(5deg)}100%{transform:scale(1) rotate(0deg);opacity:1}}
.end-title{font-family:'Outfit',sans-serif;font-size:2.5rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:.8rem;}
.end-collapse-title{color:#fca5a5;text-shadow:0 0 40px rgba(239,68,68,0.5);}
.end-victory-title{color:#67e8f9;text-shadow:0 0 40px rgba(6,182,212,0.5);}
.end-reason{color:#e2e8f0;font-size:1.15rem;margin-bottom:2rem;line-height:1.6;opacity:0.9;}
.end-stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:2rem 0;}
.end-stat{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:14px;padding:1rem;}
.end-stat-val{font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:800;}
.end-stat-lbl{font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.1em;margin-top:.3rem;}
.end-play-again{background:linear-gradient(135deg,#f59e0b,#ef4444);color:#fff;border:none;padding:1rem 3rem;border-radius:14px;font-family:'Outfit',sans-serif;font-weight:700;font-size:1.1rem;cursor:pointer;transition:all .3s;box-shadow:0 8px 25px rgba(239,68,68,0.3);margin-top:1.5rem;}
.end-play-again:hover{transform:translateY(-3px);box-shadow:0 12px 35px rgba(245,158,11,.5);}

/* ═══ POLISH: Agent Comparison Panel ═══ */
.bench-panel{background:linear-gradient(180deg,rgba(30,41,59,0.7),rgba(15,23,42,0.9));border:1px solid rgba(255,255,255,0.05);border-radius:16px;padding:1.4rem;margin-bottom:1.5rem;display:none;}
.bench-panel.visible{display:block;animation:slideIn .5s ease;}
@keyframes slideIn{0%{opacity:0;transform:translateY(10px)}100%{opacity:1;transform:translateY(0)}}
.bench-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1rem;}
.bench-card{background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:1rem;text-align:center;transition:transform .3s;}
.bench-card:hover{transform:translateY(-3px);}
.bench-bar{height:6px;background:#1e293b;border-radius:3px;margin-top:.5rem;overflow:hidden;}
.bench-fill{height:100%;border-radius:3px;transition:width 1.5s cubic-bezier(0.16,1,0.3,1);}

/* ═══ POLISH: Consequence Trail ═══ */
.consequence-trail{display:flex;align-items:center;gap:.3rem;flex-wrap:wrap;padding:.6rem 1rem;background:rgba(234,179,8,0.06);border:1px solid rgba(234,179,8,0.15);border-radius:10px;margin-bottom:.8rem;font-size:.78rem;display:none;}
.consequence-trail.visible{display:flex;animation:slideIn .5s ease;}
.ct-step{padding:3px 10px;border-radius:6px;font-weight:600;}
.ct-arrow{color:#64748b;font-size:.7rem;}

/* ═══ POLISH: Reward Viz ═══ */
.reward-viz{background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:.8rem 1.2rem;margin-bottom:1rem;display:none;}
.reward-viz.visible{display:block;}
.rv-row{display:flex;justify-content:space-between;padding:3px 0;font-size:.78rem;}
.rv-pos{color:#4ade80}.rv-neg{color:#f87171}

/* ═══ POLISH: Action Tooltips ═══ */
.act-btn{position:relative;}
.act-btn .act-tip{position:absolute;bottom:calc(100% + 10px);left:50%;transform:translateX(-50%) scale(0.92);background:linear-gradient(180deg,rgba(15,23,42,0.97),rgba(2,6,15,0.98));border:1px solid rgba(99,102,241,0.15);color:#e2e8f0;padding:8px 14px;border-radius:10px;font-size:.72rem;font-weight:500;white-space:nowrap;opacity:0;pointer-events:none;transition:all .25s cubic-bezier(0.16,1,0.3,1);z-index:100;box-shadow:0 10px 25px rgba(0,0,0,0.6);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);letter-spacing:0.02em;}
.act-btn:hover .act-tip{opacity:1;transform:translateX(-50%) scale(1);bottom:calc(100% + 12px);}

/* ═══ POLISH: Stat Change Micro-Animations ═══ */
@keyframes statBump{0%{transform:scale(1)}30%{transform:scale(1.25)}100%{transform:scale(1)}}
@keyframes statUp{0%{color:inherit}50%{color:#4ade80;text-shadow:0 0 10px rgba(74,222,128,0.5)}100%{color:inherit}}
@keyframes statDown{0%{color:inherit}50%{color:#f87171;text-shadow:0 0 10px rgba(248,113,113,0.5)}100%{color:inherit}}
.stat-bump{animation:statBump .4s ease !important;}
.stat-up{animation:statUp .8s ease !important;}
.stat-down{animation:statDown .8s ease !important;}

/* ═══ POLISH: Difficulty Badges ═══ */
.diff-hint{font-size:.7rem;color:#94a3b8;margin-top:2px;font-weight:400;font-style:italic;display:block;}
.morale-g{color:#06b6d4}.tech-g{color:#a855f7}.danger-g{color:#ef4444}
.workers-g{color:#3b82f6}.progress-g{color:#f59e0b}
#particleCanvas{position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:1;pointer-events:none;transition:opacity 1.5s ease;}
.hidden-particles{opacity:0 !important;}
#intro-screen {
  position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:9999;background:#000;
  display:flex;align-items:center;justify-content:center;
  transition:opacity 2s ease-out, transform 2s ease-out;
}
#skip-btn {
  position:absolute;bottom:40px;right:40px;background:rgba(255,255,255,0.05);
  border:1px solid rgba(255,255,255,0.15);color:rgba(255,255,255,0.8);
  padding:12px 28px;border-radius:30px;cursor:pointer;font-family:'Outfit',sans-serif;
  font-weight:600;font-size:1rem;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  transition:all 0.3s cubic-bezier(0.16,1,0.3,1);z-index:10001;display:none;
  align-items:center;gap:8px;
}
#skip-btn:hover {
  background:rgba(255,255,255,0.15);border-color:rgba(255,255,255,0.4);
  color:#fff;transform:translateY(-3px);box-shadow:0 10px 30px rgba(0,0,0,0.5);
}

/* ═══ Evolution Video Stage ═══ */
#evo-screen {
  position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:99998;background:#000;
  display:none;align-items:center;justify-content:center;
  transition:opacity 2s ease-out, transform 2s ease-out;
}
#evo-skip-btn {
  position:absolute;bottom:40px;right:40px;background:rgba(255,255,255,0.05);
  border:1px solid rgba(255,255,255,0.15);color:rgba(255,255,255,0.8);
  padding:12px 28px;border-radius:30px;cursor:pointer;font-family:'Outfit',sans-serif;
  font-weight:600;font-size:1rem;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  transition:all 0.3s cubic-bezier(0.16,1,0.3,1);z-index:10001;display:flex;
  align-items:center;gap:8px;
}
#evo-skip-btn:hover {
  background:rgba(255,255,255,0.15);border-color:rgba(255,255,255,0.4);
  color:#fff;transform:translateY(-3px);box-shadow:0 10px 30px rgba(0,0,0,0.5);
}

/* ═══ Global Skip All Button ═══ */
#skip-all-btn {
  position:fixed;top:40px;right:40px;background:rgba(255,255,255,0.05);
  border:1px solid rgba(255,255,255,0.15);color:rgba(255,255,255,0.8);
  padding:12px 28px;border-radius:30px;cursor:pointer;font-family:'Outfit',sans-serif;
  font-weight:600;font-size:1rem;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  transition:all 0.3s cubic-bezier(0.16,1,0.3,1);z-index:10002;display:flex;
  align-items:center;gap:8px;
}
#skip-all-btn:hover {
  background:rgba(255,255,255,0.15);border-color:rgba(255,255,255,0.4);
  color:#fff;transform:translateY(-3px);box-shadow:0 10px 30px rgba(0,0,0,0.5);
}

/* ═══ Bulb Bloom Stage ═══ */
#bulb-screen {
  position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:9998;
  background:radial-gradient(ellipse at center, #060610 0%, #020204 100%);
  display:none;align-items:center;justify-content:center;flex-direction:column;
  transition:opacity 2.5s ease-out;
  overflow:hidden;
}
#bulb-screen.active { display:flex; }

/* Ambient floating dust */
.ambient-dust {
  position:absolute;border-radius:50%;pointer-events:none;
  background:rgba(255,255,255,0.15);
  animation:dustFloat var(--dur) ease-in-out infinite;
}
@keyframes dustFloat {
  0%,100% { transform:translate(0,0);opacity:0.05; }
  25% { transform:translate(var(--mx),var(--my));opacity:0.2; }
  50% { transform:translate(calc(var(--mx)*-0.5),calc(var(--my)*1.5));opacity:0.1; }
  75% { transform:translate(calc(var(--mx)*0.8),calc(var(--my)*-0.6));opacity:0.15; }
}

/* The bulb itself */
.bulb-container {
  position:relative;cursor:pointer;z-index:10;
  transition:transform 0.5s ease;
}
.bulb-container:hover { transform:scale(1.08); }
.bulb-container:hover .bulb-glass {
  box-shadow:0 0 25px rgba(180,160,100,0.15);
}

.bulb-glass {
  width:90px;height:120px;position:relative;
  background:radial-gradient(ellipse at 50% 60%, rgba(60,55,40,0.2) 0%, rgba(20,18,12,0.08) 70%, transparent 100%);
  border-radius:50% 50% 38% 38%;border:2px solid rgba(180,160,100,0.12);
  transition:all 1.5s cubic-bezier(0.16,1,0.3,1);
  box-shadow:0 0 10px rgba(180,160,100,0.03);
}
/* Filament inside bulb */
.bulb-filament {
  position:absolute;top:30%;left:50%;transform:translateX(-50%);
  width:16px;height:30px;z-index:2;
}
.bulb-filament::before,.bulb-filament::after {
  content:"";position:absolute;bottom:0;width:2px;height:100%;
  background:rgba(180,160,100,0.2);border-radius:1px;
}
.bulb-filament::before { left:3px;transform:rotate(-8deg); }
.bulb-filament::after { right:3px;transform:rotate(8deg); }
.filament-coil {
  position:absolute;top:0;left:50%;transform:translateX(-50%);
  width:12px;height:12px;border:2px solid rgba(180,160,100,0.15);
  border-radius:50%;border-bottom-color:transparent;
  animation:filamentFlicker 3s ease-in-out infinite;
}
@keyframes filamentFlicker {
  0%,100% { border-color:rgba(180,160,100,0.1);border-bottom-color:transparent; }
  50% { border-color:rgba(200,180,120,0.2);border-bottom-color:transparent; }
  70% { border-color:rgba(180,160,100,0.05);border-bottom-color:transparent; }
}

.bulb-glass::after {
  content:"";position:absolute;top:20%;left:25%;width:50%;height:40%;
  background:radial-gradient(ellipse, rgba(255,200,50,0.06) 0%, transparent 70%);
  border-radius:50%;transition:all 1.5s ease;
}
/* Reflection highlight */
.bulb-glass::before {
  content:"";position:absolute;top:15%;left:20%;width:20%;height:25%;
  background:rgba(255,255,255,0.04);border-radius:50%;transform:rotate(-25deg);
  transition:all 1.5s ease;
}

.bulb-base {
  width:36px;height:24px;margin:0 auto;border-radius:0 0 6px 6px;
  background:linear-gradient(180deg,#3a3520 0%,#2a2515 50%,#1a1710 100%);
  border:1px solid rgba(180,160,100,0.15);border-top:none;
  position:relative;
}
.bulb-base::before {
  content:"";position:absolute;top:4px;left:0;right:0;height:1px;
  background:rgba(180,160,100,0.2);
}
.bulb-base::after {
  content:"";position:absolute;top:10px;left:0;right:0;height:1px;
  background:rgba(180,160,100,0.15);
}
.bulb-wire {
  width:2px;height:80px;margin:0 auto;
  background:linear-gradient(180deg,rgba(180,160,100,0.3),rgba(100,90,60,0.1),transparent);
}
.bulb-label {
  font-family:'Outfit',sans-serif;font-size:0.8rem;color:rgba(255,255,255,0.2);
  letter-spacing:0.35em;text-transform:uppercase;margin-top:2.5rem;
  transition:opacity 1s ease;text-align:center;
  animation:labelPulse 2.5s ease-in-out infinite;
}
@keyframes labelPulse {
  0%,100% { opacity:0.2; }
  50% { opacity:0.4; }
}

/* Light burst on ignition */
.light-burst {
  position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:0;height:0;border-radius:50%;pointer-events:none;z-index:4;
  background:radial-gradient(circle, rgba(255,220,80,0.5) 0%, rgba(255,180,40,0.2) 30%, transparent 70%);
  transition:width 2s cubic-bezier(0.16,1,0.3,1), height 2s cubic-bezier(0.16,1,0.3,1), opacity 3s ease;
  opacity:0;
}
.light-burst.expand {
  width:160vmax;height:160vmax;opacity:1;
}

/* Cinematic quote */
.bloom-quote {
  position:absolute;bottom:15%;left:50%;transform:translateX(-50%);
  font-family:'Outfit',sans-serif;color:rgba(255,255,255,0);font-size:1.1rem;
  letter-spacing:0.15em;text-align:center;white-space:nowrap;z-index:20;
  transition:color 2s ease, letter-spacing 2s ease;
  text-shadow:0 4px 20px rgba(0,0,0,0.8);
}
.bloom-quote.visible {
  color:rgba(255,255,255,0.7);letter-spacing:0.25em;
}

/* Glowing state */
.bulb-container.glowing .bulb-glass {
  background:radial-gradient(ellipse at 50% 60%, rgba(255,220,80,0.8) 0%, rgba(255,180,40,0.35) 40%, rgba(255,160,20,0.06) 80%, transparent 100%);
  border-color:rgba(255,200,60,0.6);
  box-shadow:0 0 80px rgba(255,200,60,0.5), 0 0 160px rgba(255,180,40,0.25), 0 0 300px rgba(255,160,20,0.12);
}
.bulb-container.glowing .bulb-glass::after {
  background:radial-gradient(ellipse, rgba(255,240,180,1) 0%, rgba(255,200,60,0.4) 50%, transparent 70%);
}
.bulb-container.glowing .bulb-glass::before {
  background:rgba(255,255,255,0.15);
}
.bulb-container.glowing .filament-coil {
  border-color:rgba(255,230,130,0.9);border-bottom-color:transparent;
  box-shadow:0 0 10px rgba(255,220,80,0.5);
  animation:none;
}
.bulb-container.glowing .bulb-filament::before,
.bulb-container.glowing .bulb-filament::after {
  background:rgba(255,220,100,0.7);box-shadow:0 0 6px rgba(255,200,60,0.4);
}

/* Vines / Creepers */
.vine-container {
  position:absolute;top:50%;left:50%;z-index:5;pointer-events:none;
}
.vine {
  position:absolute;transform-origin:center center;
  stroke-width:2.5;fill:none;
  stroke-dasharray:600;stroke-dashoffset:600;
  filter:drop-shadow(0 0 8px rgba(80,200,80,0.5));
  opacity:0;
}
.vine.grow {
  stroke-dashoffset:0;opacity:1;
  transition:stroke-dashoffset 3s cubic-bezier(0.16,1,0.3,1), opacity 0.5s ease;
}

/* Flowers */
.bloom-flower {
  position:absolute;pointer-events:none;opacity:0;transform:scale(0) rotate(0deg);
}
.bloom-flower.open {
  opacity:1;transform:scale(1) rotate(var(--rot,15deg));
  transition:all 1.4s cubic-bezier(0.34,1.56,0.64,1);
}

/* Bloom particles */
.bloom-particle {
  position:absolute;border-radius:50%;pointer-events:none;opacity:0;
}
.bloom-particle.drift {
  opacity:1;
  animation:particleDrift var(--pdur,3s) ease-out forwards;
}
@keyframes particleDrift {
  0% { transform:translate(0,0) scale(1);opacity:0.9; }
  60% { opacity:0.5; }
  100% { transform:translate(var(--dx),var(--dy)) scale(0);opacity:0; }
}

@keyframes bulbPulse {
  0%,100% { filter:drop-shadow(0 0 20px rgba(255,200,60,0.35)); }
  50% { filter:drop-shadow(0 0 40px rgba(255,200,60,0.7)); }
}
.bulb-container.glowing { animation:bulbPulse 2s ease-in-out infinite; }

/* Butterflies */
.butterfly {
  position:absolute;pointer-events:none;z-index:15;opacity:0;
  transition:opacity 1.5s ease;
}
.butterfly.visible { opacity:1; }
.butterfly-body {
  position:relative;width:4px;height:14px;
  background:linear-gradient(180deg,#4a3728,#2d1f14);border-radius:2px;
}
.butterfly-wing {
  position:absolute;top:1px;width:18px;height:22px;
  border-radius:50% 50% 50% 0;transform-origin:right center;
  animation:wingFlap 0.4s ease-in-out infinite alternate;
}
.butterfly-wing.left { right:2px;transform:rotateY(0deg);border-radius:50% 50% 0 50%; }
.butterfly-wing.right { left:2px;transform:rotateY(0deg);transform-origin:left center; }
@keyframes wingFlap {
  0% { transform:rotateY(0deg) rotateZ(-5deg); }
  100% { transform:rotateY(70deg) rotateZ(5deg); }
}
.butterfly-wing.w1 { background:radial-gradient(ellipse at 40% 40%, #fbbf24, #f59e0b 60%, #d97706); }
.butterfly-wing.w2 { background:radial-gradient(ellipse at 40% 40%, #f472b6, #ec4899 60%, #be185d); }
.butterfly-wing.w3 { background:radial-gradient(ellipse at 40% 40%, #a78bfa, #8b5cf6 60%, #6d28d9); }
.butterfly-wing.w4 { background:radial-gradient(ellipse at 40% 40%, #38bdf8, #0ea5e9 60%, #0369a1); }
.butterfly-wing.w5 { background:radial-gradient(ellipse at 40% 40%, #6ee7b7, #34d399 60%, #059669); }
.butterfly-wing .wing-dot {
  position:absolute;border-radius:50%;background:rgba(255,255,255,0.4);
  width:5px;height:5px;top:30%;left:30%;
}
@keyframes butterflyFloat {
  0% { transform:translate(0,0) rotate(0deg); }
  25% { transform:translate(var(--bx1),var(--by1)) rotate(8deg); }
  50% { transform:translate(var(--bx2),var(--by2)) rotate(-5deg); }
  75% { transform:translate(var(--bx3),var(--by3)) rotate(10deg); }
  100% { transform:translate(0,0) rotate(0deg); }
}
.butterfly.visible {
  animation:butterflyFloat var(--bdur,8s) ease-in-out infinite;
}

/* ═══════════════════════════════════════════════════════════════
   ELITE POLISH STYLES (INJECTED)
   ═══════════════════════════════════════════════════════════════ */

/* System Stress & Telemetry */
body.stress-critical::before {
  content: ''; position: fixed; top:0; left:0; width:100%; height:100%;
  box-shadow: inset 0 0 150px rgba(255, 0, 0, 0.4);
  pointer-events: none; z-index: 100;
  animation: stressPulse 2s infinite alternate;
}
@keyframes stressPulse { 0% { opacity: 0.5; } 100% { opacity: 1; } }

.sys-status-badge {
  position: absolute; top: 20px; right: 20px; padding: 5px 12px;
  background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.4);
  color: #10b981; border-radius: 4px; font-family: 'Outfit', sans-serif; font-size: 0.8rem;
  font-weight: 600; text-transform: uppercase; letter-spacing: 1px; z-index: 50;
  transition: all 0.5s ease;
}
.sys-status-badge.critical { background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.4); color: #ef4444; animation: blinkWarning 1s infinite alternate; }

.telemetry-tag {
  position: absolute; bottom: 10px; right: 20px; font-family: 'Fira Code', monospace;
  font-size: 0.7rem; color: #00ffaa; opacity: 0; transition: opacity 0.2s; z-index: 50;
}
.version-tag {
  position: absolute; bottom: 10px; left: 20px; font-family: 'Outfit', sans-serif;
  font-size: 0.8rem; color: #555; z-index: 50;
}

/* AI Reasoning Glass-Box */
.ai-reasoning-panel {
  display: none; border-left: 2px solid #00ffaa; padding-left: 15px; margin-top: 10px;
  background: rgba(0, 255, 170, 0.05); padding: 10px; border-radius: 4px;
}
.ai-reasoning-panel.active { display: block; animation: slightFadeIn 0.3s ease-out; }
.ai-reasoning-line { color: #aaa; font-size: 0.85rem; font-family: 'Fira Code', monospace; line-height: 1.4; display: block; margin-bottom: 5px; }
.ai-confidence { display: inline-block; background: rgba(0,255,170,0.1); color: #00ffaa; font-size: 0.7rem; padding: 2px 6px; border-radius: 3px; font-weight: bold; margin-right: 10px; }
.ai-strategy { display: inline-block; color: #888; font-size: 0.7rem; font-style: italic; }

/* Matrix Data Stream */
.matrix-stream {
  position: fixed; right: -400px; top: 60px; width: 350px; height: calc(100vh - 120px);
  background: rgba(10, 15, 20, 0.95); border: 1px solid #1f2937; border-right: none;
  box-shadow: -5px 0 30px rgba(0,0,0,0.8); z-index: 1000; padding: 15px;
  font-family: 'Fira Code', monospace; font-size: 0.75rem; color: #00ffaa;
  overflow-y: auto; transition: right 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.matrix-stream.active { right: 0; }
.matrix-toggle {
  position: absolute; right: 20px; top: 60px; background: rgba(30, 41, 59, 0.8);
  border: 1px solid #334155; color: #94a3b8; padding: 5px 10px; border-radius: 4px;
  font-family: 'Outfit', sans-serif; font-size: 0.8rem; cursor: pointer; z-index: 50; transition: all 0.2s;
}
.matrix-toggle:hover { color: #00ff00; border-color: #00ff00; background: rgba(0,255,0,0.1); }
.matrix-line { margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px; }
.matrix-key { color: #38bdf8; }
.matrix-str { color: #a3e635; }
.matrix-num { color: #f472b6; }
.matrix-bool { color: #fbbf24; }

/* Cinematic Era Transitions */
.era-transition-overlay {
  position: fixed; top:0; left:0; width:100%; height:100%;
  background: radial-gradient(circle, rgba(10,15,30,0.95) 0%, rgba(0,0,0,1) 100%);
  z-index: 5000; display: flex; align-items: center; justify-content: center;
  opacity: 0; pointer-events: none; transition: opacity 0.8s ease;
}
.era-transition-overlay.active { opacity: 1; pointer-events: all; }
.era-transition-overlay .era-text {
  font-family: 'Outfit', sans-serif; font-size: 4rem; font-weight: 800; color: #fff;
  letter-spacing: 15px; text-transform: uppercase;
  text-shadow: 0 0 40px rgba(255,255,255,0.5); transform: scale(0.9); transition: transform 2.5s cubic-bezier(0.1, 1, 0.1, 1);
}
.era-transition-overlay.active .era-text { transform: scale(1.1); }

/* Civilization Pulse Graph */
.pulse-graph-container {
  margin-top: 15px; width: 100%; height: 120px; background: rgba(0,0,0,0.3); border: 1px solid #1e293b;
  border-radius: 10px; position: relative; overflow: hidden; padding: 8px;
}
.pulse-svg { width: 100%; height: 100%; display: block; }
.pulse-line { fill: none; stroke: #00ffaa; stroke-width: 2.5; opacity: 0.9; stroke-linejoin: round; filter: drop-shadow(0 0 3px rgba(0,255,170,0.4)); }
.pulse-danger { stroke: #ff4444; stroke-width: 2; opacity: 0.7; stroke-dasharray: 4; filter: drop-shadow(0 0 3px rgba(255,68,68,0.4)); }

/* Miscellaneous Refinements */
.stat-card.critical { animation: glowRed 1.5s infinite alternate; border-color: rgba(239, 68, 68, 0.5); }
@keyframes glowRed { from { box-shadow: inset 0 0 10px rgba(239, 68, 68, 0.1); } to { box-shadow: inset 0 0 20px rgba(239, 68, 68, 0.3); } }

.btn.cooldown { opacity: 0.5; pointer-events: none; transform: scale(0.95); }
.dataset-btn-raw { background: transparent; border: 1px solid #334155; color: #94a3b8; font-family: 'Outfit'; padding: 10px 20px; border-radius: 6px; cursor: pointer; transition: all 0.2s; margin-top:20px; width:100%; }
.dataset-btn-raw:hover { color: #fff; border-color: #94a3b8; background: rgba(255,255,255,0.05); }
.replay-overlay { position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.9); z-index:6000; display:flex; flex-direction:column; align-items:center; justify-content:center; opacity:0; pointer-events:none; transition: opacity 0.3s; }
.replay-overlay.active { opacity:1; pointer-events:all; }

/* ═══ Matrix Data Stream Panel ═══ */
.matrix-stream{position:fixed;top:0;right:-420px;width:400px;height:100vh;background:rgba(0,8,4,0.95);border-left:1px solid rgba(0,255,100,0.15);z-index:5000;transition:right 0.4s cubic-bezier(0.16,1,0.3,1);padding:1rem;overflow-y:auto;font-family:'Courier New',monospace;font-size:0.75rem;color:#00ff66;}
.matrix-stream.active{right:0;}
.matrix-line{border-bottom:1px solid rgba(0,255,100,0.08);padding:8px 0;}
.matrix-key{color:#00ccff;}.matrix-str{color:#ffcc00;}.matrix-num{color:#ff66ff;}.matrix-bool{color:#00ff66;}
.data-stream-btn{position:fixed;top:20px;right:20px;z-index:10003;background:rgba(0,255,100,0.08);border:1px solid rgba(0,255,100,0.2);color:#00ff66;padding:6px 14px;border-radius:20px;cursor:pointer;font-family:'Outfit',sans-serif;font-weight:600;font-size:0.75rem;transition:all 0.3s;display:flex;align-items:center;gap:6px;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);}
.data-stream-btn:hover{background:rgba(0,255,100,0.15);border-color:rgba(0,255,100,0.4);color:#33ff88;}

/* ═══ Era Transition Overlay ═══ */
.era-transition-overlay{position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:8000;display:flex;align-items:center;justify-content:center;background:radial-gradient(circle,rgba(99,102,241,0.3),rgba(0,0,0,0.85));opacity:0;pointer-events:none;transition:opacity 0.6s ease;}
.era-transition-overlay.active{opacity:1;pointer-events:auto;}
.era-transition-text{font-family:'Outfit',sans-serif;font-size:3.5rem;font-weight:800;color:#fff;text-shadow:0 0 60px rgba(99,102,241,0.8),0 0 120px rgba(99,102,241,0.4);letter-spacing:0.15em;animation:eraTextPulse 2s ease;}
@keyframes eraTextPulse{0%{transform:scale(0.5);opacity:0}50%{transform:scale(1.1);opacity:1}100%{transform:scale(1);opacity:1}}

/* ═══ AI Reasoning Panel ═══ */
.ai-reasoning-panel{background:rgba(168,85,247,0.06);border:1px solid rgba(168,85,247,0.15);border-radius:12px;padding:0.6rem 1rem;margin-top:0.8rem;display:none;font-size:0.82rem;animation:slideIn 0.4s ease;}
.ai-reasoning-panel.active{display:block;}
.ai-reasoning-line{color:#c4b5fd;margin-bottom:4px;}
.ai-confidence{color:#a78bfa;font-size:0.75rem;}
.ai-strategy{color:#818cf8;font-size:0.75rem;font-weight:600;}

/* ═══ Pulse Graph ═══ */
.pulse-graph-container{background:rgba(15,23,42,0.5);border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:0.6rem;margin-bottom:1rem;}

/* ═══ Status Badge ═══ */
.status-badge{position:fixed;top:20px;left:70px;z-index:10003;background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);color:#34d399;padding:4px 12px;border-radius:20px;font-family:'Outfit',sans-serif;font-weight:600;font-size:0.7rem;letter-spacing:0.05em;display:flex;align-items:center;gap:5px;backdrop-filter:blur(8px);}

/* ═══ Telemetry Tag ═══ */
.telemetry-tag{position:fixed;bottom:15px;left:25px;color:rgba(0,255,100,0.4);font-family:'Courier New',monospace;font-size:0.65rem;z-index:9000;pointer-events:none;transition:opacity 0.5s;opacity:0;}

/* ═══════════════════════════════════════════════════════════════ */
/* PHASE 2: Advanced Research-Grade UI Styles                     */
/* ═══════════════════════════════════════════════════════════════ */

/* ═══ AI Council Panel ═══ */
.council-panel{display:none;background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(168,85,247,0.06));border:1px solid rgba(99,102,241,0.2);border-radius:16px;padding:1rem 1.2rem;margin-bottom:1rem;animation:slideIn 0.4s ease;}
.council-panel.active{display:block;}
.council-header{display:flex;align-items:center;gap:8px;font-size:0.85rem;font-weight:700;color:#c4b5fd;margin-bottom:0.8rem;}
.council-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:0.6rem;}
.council-member{background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:0.8rem;transition:all 0.3s;}
.council-member:hover{border-color:rgba(99,102,241,0.3);transform:translateY(-2px);}
.council-member-header{display:flex;align-items:center;gap:6px;margin-bottom:6px;}
.council-member-role{font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#a78bfa;}
.council-member-provider{font-size:0.6rem;color:#475569;margin-left:auto;}
.council-member-opinion{font-size:0.78rem;color:#cbd5e1;line-height:1.5;}
.council-member.unavailable{opacity:0.4;border-style:dashed;}
.council-consensus{background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:10px;padding:0.6rem 1rem;margin-top:0.8rem;font-size:0.82rem;color:#a7f3d0;}
.council-btn{background:linear-gradient(135deg,rgba(99,102,241,0.2),rgba(168,85,247,0.15));border:1px solid rgba(99,102,241,0.3);color:#c4b5fd;padding:6px 16px;border-radius:10px;cursor:pointer;font-family:'Outfit',sans-serif;font-weight:600;font-size:0.78rem;transition:all 0.3s;display:flex;align-items:center;gap:6px;}
.council-btn:hover{background:linear-gradient(135deg,rgba(99,102,241,0.3),rgba(168,85,247,0.25));border-color:rgba(99,102,241,0.5);transform:translateY(-1px);}
.council-btn:disabled{opacity:0.5;cursor:wait;}

/* ═══ LLM Judge Badge ═══ */
.judge-badge{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:6px;font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;animation:fadeInUp 0.4s ease;}
.judge-MASTERFUL{background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.3);}
.judge-STRONG{background:rgba(59,130,246,0.15);color:#93c5fd;border:1px solid rgba(59,130,246,0.3);}
.judge-ADEQUATE{background:rgba(234,179,8,0.15);color:#fde047;border:1px solid rgba(234,179,8,0.3);}
.judge-POOR{background:rgba(249,115,22,0.15);color:#fdba74;border:1px solid rgba(249,115,22,0.3);}
.judge-CATASTROPHIC{background:rgba(239,68,68,0.15);color:#fca5a5;border:1px solid rgba(239,68,68,0.3);}
@keyframes fadeInUp{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}

/* ═══ Behavioral Radar Chart ═══ */
.radar-container{margin:1.5rem auto;text-align:center;position:relative;}
.radar-canvas{display:block;margin:0 auto;}
.radar-title{font-size:0.8rem;font-weight:700;color:#c4b5fd;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;display:flex;align-items:center;justify-content:center;gap:6px;}
.radar-alignment{display:inline-flex;align-items:center;gap:6px;padding:4px 14px;border-radius:20px;font-size:0.75rem;font-weight:700;margin-top:0.5rem;background:rgba(168,85,247,0.12);color:#d8b4fe;border:1px solid rgba(168,85,247,0.25);}
.radar-metrics{display:flex;gap:1rem;justify-content:center;margin-top:0.8rem;flex-wrap:wrap;}
.radar-metric{text-align:center;}
.radar-metric-val{font-size:1.1rem;font-weight:800;font-family:'Outfit',sans-serif;}
.radar-metric-lbl{font-size:0.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;}

/* ═══ Causality Web ═══ */
.causality-web{display:none;background:rgba(15,23,42,0.7);border:1px solid rgba(255,255,255,0.05);border-radius:16px;padding:1rem;margin-bottom:1rem;position:relative;overflow:hidden;}
.causality-web.active{display:block;}
.causality-header{font-size:0.78rem;font-weight:700;color:#94a3b8;margin-bottom:8px;display:flex;align-items:center;gap:6px;}
.causality-canvas{width:100%;height:180px;display:block;}
.causality-legend{display:flex;gap:1rem;margin-top:6px;font-size:0.65rem;color:#64748b;}
.causality-legend span{display:flex;align-items:center;gap:4px;}
.cw-node{cursor:pointer;transition:all 0.3s;}
.cw-node:hover{filter:brightness(1.3);}

/* ═══ Chronicle Panel ═══ */
.chronicle-panel{background:linear-gradient(135deg,rgba(168,85,247,0.06),rgba(99,102,241,0.04));border:1px solid rgba(168,85,247,0.15);border-radius:16px;padding:1.2rem 1.5rem;margin-top:1rem;font-size:0.88rem;line-height:1.7;color:#cbd5e1;position:relative;overflow:hidden;}
.chronicle-panel::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#8b5cf6,#06b6d4,#10b981);opacity:0.6;}
.chronicle-title{font-size:0.8rem;font-weight:700;color:#a78bfa;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;display:flex;align-items:center;gap:6px;}
.chronicle-text{font-style:italic;color:#94a3b8;line-height:1.8;}
.chronicle-source{font-size:0.6rem;color:#475569;margin-top:0.5rem;text-align:right;}

/* ═══ Multi-Run Comparison ═══ */
.multirun-panel{background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.05);border-radius:16px;padding:1rem;margin-top:1rem;}
.multirun-header{font-size:0.8rem;font-weight:700;color:#94a3b8;margin-bottom:0.6rem;display:flex;align-items:center;gap:6px;}
.multirun-table{width:100%;border-collapse:collapse;font-size:0.75rem;}
.multirun-table th{color:#64748b;text-transform:uppercase;letter-spacing:0.06em;font-size:0.65rem;padding:6px 8px;border-bottom:1px solid rgba(255,255,255,0.08);text-align:left;}
.multirun-table td{padding:6px 8px;color:#cbd5e1;border-bottom:1px solid rgba(255,255,255,0.03);}
.multirun-table tr:hover td{background:rgba(255,255,255,0.02);}
.multirun-survived{color:#10b981;font-weight:700;}
.multirun-collapsed{color:#ef4444;font-weight:700;}
.multirun-btn{background:transparent;border:1px solid rgba(99,102,241,0.3);color:#a78bfa;padding:5px 14px;border-radius:8px;cursor:pointer;font-family:'Outfit',sans-serif;font-weight:600;font-size:0.72rem;transition:all 0.3s;margin-top:0.6rem;}
.multirun-btn:hover{background:rgba(99,102,241,0.1);border-color:rgba(99,102,241,0.5);}

/* ═══ Enhanced AI Reasoning (Phase 2) ═══ */
.ai-cot-panel{background:rgba(168,85,247,0.06);border:1px solid rgba(168,85,247,0.15);border-radius:12px;padding:0.8rem 1rem;margin-top:0.8rem;display:none;animation:slideIn 0.4s ease;}
.ai-cot-panel.active{display:block;}
.ai-cot-thought{color:#e9d5ff;font-size:0.82rem;margin-bottom:6px;padding-left:8px;border-left:3px solid rgba(168,85,247,0.4);}
.ai-cot-action{color:#a78bfa;font-size:0.78rem;font-weight:700;display:flex;align-items:center;gap:6px;}
.ai-cot-confidence{font-size:0.7rem;color:#818cf8;margin-top:4px;}
.ai-cot-judge{font-size:0.7rem;margin-top:6px;padding:4px 8px;border-radius:6px;}

/* ═══ SYSTEM 1: Meta-AI Controller Badge ═══ */
.meta-ai-badge{position:fixed;bottom:12px;left:12px;z-index:9000;background:rgba(15,15,30,0.92);border:1px solid rgba(168,85,247,0.4);border-radius:10px;padding:8px 14px;font-size:0.7rem;color:#c4b5fd;backdrop-filter:blur(8px);display:flex;align-items:center;gap:8px;cursor:pointer;transition:all 0.3s;}
.meta-ai-badge:hover{border-color:#a855f7;transform:translateY(-2px);box-shadow:0 4px 20px rgba(168,85,247,0.3);}
.meta-ai-badge .meta-dot{width:8px;height:8px;border-radius:50%;background:#10b981;animation:pulse 2s infinite;}
.meta-ai-badge.degraded .meta-dot{background:#ef4444;}
.meta-ai-badge .meta-label{font-weight:600;color:#e9d5ff;}
.meta-ai-badge .meta-model{color:#818cf8;font-size:0.65rem;}

/* ═══ SYSTEM 2: Parallel Sim Comparison Overlay ═══ */
.parallel-btn{background:linear-gradient(135deg,rgba(16,185,129,0.15),rgba(59,130,246,0.15));border:1px solid rgba(16,185,129,0.3);color:#6ee7b7;padding:6px 14px;border-radius:8px;cursor:pointer;font-size:0.75rem;font-weight:600;transition:all 0.3s;display:inline-flex;align-items:center;gap:6px;}
.parallel-btn:hover{border-color:#10b981;background:linear-gradient(135deg,rgba(16,185,129,0.25),rgba(59,130,246,0.25));transform:translateY(-1px);}
.parallel-overlay{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(5,5,15,0.95);z-index:10001;display:none;align-items:center;justify-content:center;backdrop-filter:blur(8px);}
.parallel-overlay.active{display:flex;}
.parallel-content{background:linear-gradient(135deg,rgba(20,20,40,0.98),rgba(10,10,25,0.98));border:1px solid rgba(16,185,129,0.3);border-radius:16px;padding:32px;max-width:900px;width:95%;max-height:85vh;overflow-y:auto;}
.parallel-title{font-size:1.3rem;font-weight:800;background:linear-gradient(135deg,#6ee7b7,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:20px;display:flex;align-items:center;gap:10px;}
.parallel-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;margin:20px 0;}
.parallel-card{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px;transition:all 0.3s;}
.parallel-card.winner{border-color:rgba(16,185,129,0.5);box-shadow:0 0 20px rgba(16,185,129,0.15);}
.parallel-card .pc-agent{font-size:1.1rem;font-weight:800;margin-bottom:8px;text-transform:uppercase;}
.parallel-card .pc-result{font-size:0.85rem;padding:4px 10px;border-radius:6px;display:inline-block;margin-bottom:10px;}
.parallel-card .pc-result.survived{background:rgba(16,185,129,0.15);color:#6ee7b7;}
.parallel-card .pc-result.collapsed{background:rgba(239,68,68,0.15);color:#fca5a5;}
.parallel-card .pc-stat{display:flex;justify-content:space-between;font-size:0.75rem;color:#94a3b8;margin:4px 0;}
.parallel-card .pc-stat span:last-child{color:#e2e8f0;font-weight:600;}
.parallel-close{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.15);color:#e2e8f0;padding:10px 24px;border-radius:8px;cursor:pointer;font-size:0.85rem;margin-top:16px;}
.parallel-loading{text-align:center;color:#818cf8;padding:40px;font-size:1rem;}

/* ═══ SYSTEM 3: Memory Indicator ═══ */
.memory-badge{position:fixed;bottom:12px;left:200px;z-index:9000;background:rgba(15,15,30,0.92);border:1px solid rgba(251,191,36,0.3);border-radius:10px;padding:6px 12px;font-size:0.65rem;color:#fbbf24;backdrop-filter:blur(8px);display:flex;align-items:center;gap:6px;cursor:pointer;transition:all 0.3s;}
.memory-badge:hover{border-color:#f59e0b;transform:translateY(-2px);}
.memory-badge .mem-icon{font-size:0.85rem;}

/* ═══════════════════════════════════════════════════════════════
   FEATURE 1: CHRONO-FRACTAL TEMPORAL REWIND
   ═══════════════════════════════════════════════════════════════ */
.chrono-overlay{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(3,5,15,0.97);z-index:10003;display:none;flex-direction:column;backdrop-filter:blur(12px);animation:chronoFadeIn 0.6s ease;}
.chrono-overlay.active{display:flex;}
@keyframes chronoFadeIn{from{opacity:0;transform:scale(0.98)}to{opacity:1;transform:scale(1)}}
.chrono-header{padding:24px 32px 16px;display:flex;align-items:center;justify-content:space-between;}
.chrono-title{font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#c084fc,#38bdf8,#6ee7b7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:flex;align-items:center;gap:10px;}
.chrono-close{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.15);color:#e2e8f0;padding:8px 20px;border-radius:8px;cursor:pointer;font-size:0.8rem;transition:all 0.3s;}
.chrono-close:hover{background:rgba(255,255,255,0.12);border-color:rgba(255,255,255,0.3);}
.chrono-body{flex:1;display:flex;flex-direction:column;padding:0 32px;overflow-y:auto;}
.chrono-state-display{display:grid;grid-template-columns:repeat(7,1fr);gap:12px;margin-bottom:20px;}
.chrono-stat{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:12px;text-align:center;transition:all 0.4s;}
.chrono-stat .cs-val{font-size:1.6rem;font-weight:800;transition:all 0.3s;}
.chrono-stat .cs-lbl{font-size:0.65rem;color:#64748b;margin-top:4px;text-transform:uppercase;letter-spacing:0.05em;}
.chrono-action-display{text-align:center;padding:16px;background:rgba(168,85,247,0.06);border:1px solid rgba(168,85,247,0.2);border-radius:10px;margin-bottom:16px;font-size:1rem;color:#c4b5fd;transition:all 0.4s;}
.chrono-messages-display{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:10px;padding:12px;max-height:200px;overflow-y:auto;font-size:0.75rem;color:#94a3b8;margin-bottom:20px;}
.chrono-timeline-bar{padding:20px 32px 24px;background:rgba(10,10,25,0.95);border-top:1px solid rgba(255,255,255,0.06);}
.chrono-slider-wrap{position:relative;padding:20px 0 8px;}
.chrono-slider{width:100%;-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:linear-gradient(90deg,#6366f1,#a855f7,#ec4899);outline:none;opacity:0.9;transition:opacity 0.2s;}
.chrono-slider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:22px;height:22px;border-radius:50%;background:radial-gradient(circle,#fff,#c4b5fd);cursor:pointer;box-shadow:0 0 15px rgba(168,85,247,0.6),0 0 30px rgba(168,85,247,0.3);transition:all 0.2s;}
.chrono-slider::-webkit-slider-thumb:hover{transform:scale(1.2);box-shadow:0 0 20px rgba(168,85,247,0.8);}
.chrono-turn-info{display:flex;justify-content:space-between;color:#64748b;font-size:0.7rem;margin-top:6px;}
.chrono-singularity-marker{position:absolute;top:8px;width:20px;height:20px;transform:translateX(-50%) rotate(45deg);background:radial-gradient(circle,#ef4444,#991b1b);border:2px solid #fca5a5;box-shadow:0 0 15px rgba(239,68,68,0.6);z-index:2;cursor:pointer;animation:singularityPulse 1.5s infinite;}
@keyframes singularityPulse{0%,100%{box-shadow:0 0 15px rgba(239,68,68,0.6)}50%{box-shadow:0 0 25px rgba(239,68,68,0.9),0 0 40px rgba(239,68,68,0.4)}}
.chrono-singularity-label{position:absolute;top:-12px;font-size:0.55rem;color:#fca5a5;white-space:nowrap;transform:translateX(-50%);font-weight:700;text-transform:uppercase;letter-spacing:0.05em;}
.chrono-feedback{text-align:center;font-size:0.7rem;padding:6px;margin-bottom:4px;border-radius:6px;transition:all 0.3s;}
.chrono-feedback.good{color:#6ee7b7;background:rgba(16,185,129,0.1);}
.chrono-feedback.bad{color:#fca5a5;background:rgba(239,68,68,0.1);}

/* ═══════════════════════════════════════════════════════════════
   FEATURE 2: TESSERACT TIMELINE (QUANTUM BRANCHING)
   ═══════════════════════════════════════════════════════════════ */
.tesseract-overlay{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(3,5,15,0.97);z-index:10004;display:none;flex-direction:column;backdrop-filter:blur(12px);}
.tesseract-overlay.active{display:flex;animation:chronoFadeIn 0.6s ease;}
.tesseract-header{padding:20px 28px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(99,102,241,0.2);}
.tesseract-title{font-size:1.3rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#c084fc,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:flex;align-items:center;gap:10px;}
.tesseract-body{flex:1;display:grid;grid-template-columns:1fr 60px 1fr;gap:0;overflow:hidden;}
.tesseract-branch{padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:12px;}
.tesseract-branch-a{border-right:1px solid rgba(99,102,241,0.15);background:rgba(99,102,241,0.02);}
.tesseract-branch-b{border-left:1px solid rgba(236,72,153,0.15);background:rgba(236,72,153,0.02);}
.tesseract-divider{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;color:#64748b;font-size:0.6rem;padding:8px 0;}
.tesseract-divider-line{width:2px;flex:1;background:linear-gradient(180deg,rgba(99,102,241,0.4),rgba(168,85,247,0.6),rgba(236,72,153,0.4));}
.tesseract-branch-label{text-align:center;font-size:1rem;font-weight:800;padding:8px;border-radius:8px;margin-bottom:8px;}
.tb-label-a{color:#818cf8;background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);}
.tb-label-b{color:#f472b6;background:rgba(236,72,153,0.1);border:1px solid rgba(236,72,153,0.2);}
.tesseract-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px;}
.ts-stat{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:8px;text-align:center;}
.ts-stat .ts-v{font-size:1.1rem;font-weight:700;}
.ts-stat .ts-l{font-size:0.55rem;color:#64748b;text-transform:uppercase;}
.tesseract-log{font-size:0.7rem;color:#94a3b8;max-height:300px;overflow-y:auto;padding:8px;background:rgba(0,0,0,0.2);border-radius:8px;line-height:1.7;}
.tesseract-log-entry{padding:3px 6px;border-radius:4px;margin-bottom:2px;}
.tesseract-log-entry.effective{border-left:2px solid #10b981;}
.tesseract-log-entry.ineffective{border-left:2px solid #ef4444;}
.tesseract-result{text-align:center;padding:16px;margin-top:auto;border-radius:10px;font-weight:700;}
.tesseract-result.survived{background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);color:#6ee7b7;}
.tesseract-result.collapsed{background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:#fca5a5;}
.tesseract-footer{padding:12px 28px;text-align:center;border-top:1px solid rgba(255,255,255,0.06);}

/* ═══════════════════════════════════════════════════════════════
   FEATURE 3: NEURAL CARTOGRAPHY (GENERATIVE VISUALIZATION)
   ═══════════════════════════════════════════════════════════════ */
.neuro-canvas-wrap{position:relative;width:100%;height:120px;margin:8px 0;border-radius:10px;overflow:hidden;background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.04);}
.neuro-canvas-wrap canvas{width:100%;height:100%;display:block;}
.neuro-label{position:absolute;top:6px;left:10px;font-size:0.55rem;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:0.1em;font-weight:600;}

/* ═══════════════════════════════════════════════════════════════
   FEATURE 4: CONTROLLED SELF-EVOLUTION
   ═══════════════════════════════════════════════════════════════ */
.evo-unlock-toast{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%) scale(0.9);z-index:10005;background:radial-gradient(ellipse at center,rgba(168,85,247,0.2),rgba(3,5,15,0.98));border:2px solid rgba(168,85,247,0.5);border-radius:20px;padding:40px 50px;text-align:center;opacity:0;pointer-events:none;transition:all 0.6s cubic-bezier(0.16,1,0.3,1);backdrop-filter:blur(20px);}
.evo-unlock-toast.active{opacity:1;transform:translate(-50%,-50%) scale(1);pointer-events:auto;}
.evo-unlock-icon{font-size:3rem;margin-bottom:12px;animation:evoFloat 2s ease-in-out infinite;}
@keyframes evoFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
.evo-unlock-title{font-size:1.3rem;font-weight:800;background:linear-gradient(135deg,#c084fc,#f472b6,#fbbf24);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px;}
.evo-unlock-desc{font-size:0.8rem;color:#94a3b8;margin-bottom:16px;line-height:1.5;}
.evo-unlock-action{font-size:0.9rem;color:#a78bfa;font-weight:700;padding:8px 20px;background:rgba(168,85,247,0.15);border:1px solid rgba(168,85,247,0.3);border-radius:8px;display:inline-block;animation:evoPulse 2s infinite;}
@keyframes evoPulse{0%,100%{box-shadow:0 0 10px rgba(168,85,247,0.3)}50%{box-shadow:0 0 25px rgba(168,85,247,0.6)}}
.evolved-btn{position:relative;overflow:hidden;}
.evolved-btn::before{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:conic-gradient(from 0deg,transparent,rgba(168,85,247,0.3),transparent,rgba(236,72,153,0.3),transparent);animation:evoRotate 3s linear infinite;z-index:-1;}
@keyframes evoRotate{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
.evolved-badge{position:absolute;top:-4px;right:-4px;width:12px;height:12px;background:radial-gradient(circle,#c084fc,#7c3aed);border-radius:50%;border:2px solid #1e1b4b;animation:evoPulse 1.5s infinite;}

/* ═══════════════════════════════════════════════════════════════
   FEATURE 5: MULTI-MODAL VISION + AUDIO INDICATORS
   ═══════════════════════════════════════════════════════════════ */
.crisis-vision-card{position:relative;margin-top:8px;padding:10px;border-radius:10px;overflow:hidden;animation:visionReveal 0.8s ease;}
@keyframes visionReveal{from{opacity:0;max-height:0}to{opacity:1;max-height:200px}}
.crisis-vision-bg{position:absolute;top:0;left:0;width:100%;height:100%;opacity:0.15;background-size:cover;background-position:center;filter:blur(2px);}
.crisis-vision-text{position:relative;z-index:1;font-size:0.72rem;font-weight:600;text-shadow:0 1px 4px rgba(0,0,0,0.8);}
.audio-pulse-indicator{display:inline-flex;align-items:center;gap:3px;margin-left:6px;vertical-align:middle;}
.audio-pulse-bar{width:2px;background:#a78bfa;border-radius:1px;animation:audioPulse 0.6s ease-in-out infinite alternate;}
.audio-pulse-bar:nth-child(1){height:8px;animation-delay:0s;}
.audio-pulse-bar:nth-child(2){height:14px;animation-delay:0.15s;}
.audio-pulse-bar:nth-child(3){height:10px;animation-delay:0.3s;}
.audio-pulse-bar:nth-child(4){height:16px;animation-delay:0.1s;}
.audio-pulse-bar:nth-child(5){height:6px;animation-delay:0.25s;}
@keyframes audioPulse{from{height:3px}to{height:var(--pulse-h,14px)}}

/* ═══════════════════════════════════════════════════════════════
   FEATURE 6: CAUSAL SINGULARITY MARKERS
   ═══════════════════════════════════════════════════════════════ */
.singularity-node{position:relative;display:inline-block;}
.singularity-diamond{width:12px;height:12px;background:radial-gradient(circle,#ef4444,#7f1d1d);transform:rotate(45deg);border:1px solid #fca5a5;box-shadow:0 0 12px rgba(239,68,68,0.5);animation:singularityPulse 1.5s infinite;}
.singularity-tag{position:absolute;bottom:-14px;left:50%;transform:translateX(-50%);font-size:0.5rem;color:#fca5a5;white-space:nowrap;font-weight:700;text-transform:uppercase;}
.evo-milestone-marker{display:inline-block;width:10px;height:10px;background:radial-gradient(circle,#a78bfa,#4c1d95);border-radius:50%;border:1px solid #c4b5fd;box-shadow:0 0 10px rgba(168,85,247,0.4);animation:evoPulse 2s infinite;}

/* ═══════════════════════════════════════════════════════════════
   INTRO VIDEO OVERLAY
   ═══════════════════════════════════════════════════════════════ */
#introVideoOverlay {
  position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
  z-index: 99999; background: #000; display: flex;
  align-items: center; justify-content: center;
  transition: opacity 0.8s ease-out;
}
#introVideoOverlay.fade-out { opacity: 0; pointer-events: none; }
#introVideo {
  width: 100%; height: 100%; object-fit: cover;
}
#skipIntroBtn {
  position: fixed; bottom: 40px; right: 40px; z-index: 100000;
  padding: 14px 36px; font-size: 0.95rem; font-weight: 700;
  font-family: 'Outfit', sans-serif; letter-spacing: 0.08em;
  text-transform: uppercase; cursor: pointer;
  color: #fff; background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 50px; backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.15);
  transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
  overflow: hidden;
  animation: skipBtnFadeIn 1.5s ease-out 0.5s both;
}
#skipIntroBtn:hover {
  background: rgba(255,255,255,0.18); border-color: rgba(255,255,255,0.5);
  transform: translateY(-3px) scale(1.04);
  box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 30px rgba(168,85,247,0.3), inset 0 1px 0 rgba(255,255,255,0.25);
}
#skipIntroBtn:active { transform: translateY(0) scale(0.98); }
#skipIntroBtn::after {
  content: ''; position: absolute; top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: linear-gradient(90deg, transparent 30%, rgba(255,255,255,0.12) 50%, transparent 70%);
  animation: skipShimmer 3s ease-in-out infinite;
}
@keyframes skipShimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
@keyframes skipBtnFadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
#skipAllBtn {
  position: fixed; bottom: 40px; left: 40px; z-index: 100000;
  padding: 14px 36px; font-size: 0.95rem; font-weight: 700;
  font-family: 'Outfit', sans-serif; letter-spacing: 0.08em;
  text-transform: uppercase; cursor: pointer;
  color: #fff; background: linear-gradient(135deg, rgba(168,85,247,0.2), rgba(236,72,153,0.2));
  border: 1px solid rgba(168,85,247,0.4);
  border-radius: 50px; backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);
  transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
  overflow: hidden;
  animation: skipBtnFadeIn 1.5s ease-out 1s both;
}
#skipAllBtn:hover {
  background: linear-gradient(135deg, rgba(168,85,247,0.4), rgba(236,72,153,0.4));
  border-color: rgba(168,85,247,0.7);
  transform: translateY(-3px) scale(1.04);
  box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 30px rgba(168,85,247,0.4), inset 0 1px 0 rgba(255,255,255,0.2);
}
#skipAllBtn:active { transform: translateY(0) scale(0.98); }
#skipAllBtn::after {
  content: ''; position: absolute; top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: linear-gradient(90deg, transparent 30%, rgba(168,85,247,0.15) 50%, transparent 70%);
  animation: skipShimmer 3s ease-in-out infinite;
}
/* Evolution video skip button */
#evoSkipBtn {
  position: fixed; bottom: 40px; right: 40px; z-index: 100000;
  padding: 14px 36px; font-size: 0.95rem; font-weight: 700;
  font-family: 'Outfit', sans-serif; letter-spacing: 0.08em;
  text-transform: uppercase; cursor: pointer;
  color: #fff; background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 50px; backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.15);
  transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
  overflow: hidden;
  animation: skipBtnFadeIn 1.5s ease-out 0.5s both;
}
#evoSkipBtn:hover {
  background: rgba(255,255,255,0.18); border-color: rgba(255,255,255,0.5);
  transform: translateY(-3px) scale(1.04);
  box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 30px rgba(168,85,247,0.3), inset 0 1px 0 rgba(255,255,255,0.25);
}
#evoSkipBtn:active { transform: translateY(0) scale(0.98); }
#evoSkipBtn::after {
  content: ''; position: absolute; top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: linear-gradient(90deg, transparent 30%, rgba(255,255,255,0.12) 50%, transparent 70%);
  animation: skipShimmer 3s ease-in-out infinite;
}


/* ── PREMIUM POLISH: Enhanced Transitions & Micro-Animations ── */
html { scroll-behavior: smooth; }

/* Smoother stat card hover */
.stat-item {
  transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
}
.stat-item:hover {
  transform: translateY(-3px) scale(1.03);
  box-shadow: 0 8px 25px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
}

/* Enhanced action button press feedback */
.action-btn {
  transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
  position: relative;
  overflow: hidden;
}
.action-btn:hover {
  transform: translateY(-2px) scale(1.04);
  box-shadow: 0 6px 20px rgba(0,0,0,0.4), 0 0 30px rgba(255,255,255,0.05);
}
.action-btn:active {
  transform: translateY(1px) scale(0.97);
  transition-duration: 0.1s;
}

/* Smooth ripple on action buttons */
.action-btn::after {
  content: '';
  position: absolute;
  top: 50%; left: 50%;
  width: 0; height: 0;
  border-radius: 50%;
  background: rgba(255,255,255,0.15);
  transform: translate(-50%, -50%);
  transition: width 0.6s ease, height 0.6s ease, opacity 0.6s ease;
  opacity: 0;
}
.action-btn:active::after {
  width: 300px; height: 300px;
  opacity: 1;
  transition-duration: 0s;
}

/* Card fade-in on page load */
.msg-card, .dataset-btn-raw, .done-card {
  animation: cardFadeIn 0.5s ease-out backwards;
}
@keyframes cardFadeIn {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Smoother cinematic background crossfade */
#cb-layer-A, #cb-layer-B {
  transition: opacity 1.8s cubic-bezier(0.4, 0, 0.2, 1) !important;
  will-change: opacity;
  image-rendering: auto;
}

/* Better focus outlines for accessibility */
button:focus-visible, select:focus-visible, input:focus-visible {
  outline: 2px solid rgba(245, 158, 11, 0.7);
  outline-offset: 2px;
}

/* Text selection color matching brand */
::selection {
  background: rgba(239, 68, 68, 0.3);
  color: #fff;
}

/* Smoother scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.5); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, #f59e0b, #ef4444);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #fbbf24, #f87171); }

</style>
</head>
<body>
<!-- Intro Video Overlay -->
<div id="introVideoOverlay">
  <video id="introVideo" autoplay muted playsinline>
    <source src="/assets/intro_video.mp4" type="video/mp4">
  </video>
  <button id="skipIntroBtn" onclick="skipIntroVideo()">
    <i class="ph-bold ph-skip-forward" style="margin-right:8px;font-size:1.1rem;vertical-align:middle"></i>Skip Intro
  </button>
  <button id="skipAllBtn" onclick="skipAllStages()">
    <i class="ph-bold ph-fast-forward" style="margin-right:8px;font-size:1.1rem;vertical-align:middle"></i>Skip All
  </button>
</div>

<!-- Bulb Bloom Screen -->
<div id="bulb-screen">
  <div id="ambientDust"></div>
  <div id="lightBurst" class="light-burst"></div>
  <div class="bulb-container" id="theBulb" onclick="igniteBulb()">
    <div class="bulb-glass">
      <div class="bulb-filament">
        <div class="filament-coil"></div>
      </div>
    </div>
    <div class="bulb-base"></div>
    <div class="bulb-wire"></div>
  </div>
  <div class="bulb-label" id="bulbLabel">click to ignite</div>
  <div class="vine-container" id="vineContainer">
    <svg width="0" height="0" style="position:absolute">
      <defs>
        <linearGradient id="vineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style="stop-color:#22c55e;stop-opacity:0.9"/>
          <stop offset="100%" style="stop-color:#15803d;stop-opacity:0.5"/>
        </linearGradient>
        <linearGradient id="vineGrad2" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style="stop-color:#4ade80;stop-opacity:0.8"/>
          <stop offset="100%" style="stop-color:#166534;stop-opacity:0.4"/>
        </linearGradient>
      </defs>
    </svg>
  </div>
  <div id="bloomParticles" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:12;"></div>
  <div id="butterflyContainer" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:16;"></div>
  <div class="bloom-quote" id="bloomQuote">"From darkness, life finds a way."</div>
</div>

<!-- Evolution Video Screen -->
<div id="evo-screen">
  <video id="evo-vid" muted playsinline style="width:100%;height:100%;object-fit:cover;">
    <source src="/assets/first_meta_vid_ok.mp4" type="video/mp4">
  </video>
  <button id="evoSkipBtn" onclick="skipEvoVideo()">
    <i class="ph-bold ph-skip-forward" style="margin-right:8px;font-size:1.1rem;vertical-align:middle"></i>Skip
  </button>
</div>

<!-- Cinematic Civilization Background -->
<div class="cb-layer" id="cb-layer-A"></div>
<div class="cb-layer" id="cb-layer-B"></div>
<div id="cinematic-clicker"></div>
<div id="cinematic-label"><span id="cinematic-label-text"></span></div>

<canvas id="particleCanvas" style="pointer-events:none;position:fixed;top:0;left:0;z-index:0;"></canvas>
<div class="app app-landing" id="mainAppDiv" style="opacity:1;position:relative;z-index:2;pointer-events:auto;">
<h1><i class="ph-fill ph-bank" style="vertical-align:middle;margin-right:8px;color:#f59e0b;filter:drop-shadow(0 0 10px #f59e0b)"></i>Civil Command Center</h1>
<p class="subtitle">You are the leader. Messages arrive. Decisions shape the fate of your civilization.</p>
<div class="top-bar">
<select id="taskSel" onchange="updateDiffHint()">
<option value="task_demo_5">▸ Quick Demo — 5 Turns</option>
<option value="task_demo_10">▸ Standard Demo — 10 Turns</option>
<option value="task_easy" selected>● Easy — Survive 10 Turns</option>
<option value="task_medium">◆ Medium — Grow & Trade (20 Turns)</option>
<option value="task_hard">▲ Hard — Rise of Civilization (30 Turns)</option>
</select>
<button class="start-btn" onclick="resetGame()"><i class="ph-fill ph-sword"></i> Begin</button>

<div class="demo-btn-group" style="display:inline-block; margin-right:10px;">
  <button class="btn btn-secondary btn-demo" id="demoPlayBtn" onclick="toggleAIPlay()">
    <i class="ph-bold ph-lightning"></i> AI Auto-Play
  </button>
  <span class="demo-hint" style="opacity: 0.5; margin-left:8px; font-size: 0.7rem; color:#818cf8;font-family:'Outfit',sans-serif;letter-spacing:0.03em;">Autonomous Mode</span>
</div>
<button class="btn btn-secondary" style="font-size:0.75rem; padding: 4px 10px;" onclick="openReplay()"><i class="ph-bold ph-film-strip"></i> Replay</button>


<div id="aiThinkingBox" class="hidden" style="margin-top:15px; color:#aaa; font-size:0.85rem; display:flex; align-items:center; gap:8px;"></div>
<div id="aiReasoningPanel" class="ai-reasoning-panel">
  <div id="aiReasoningLine" class="ai-reasoning-line"><i class="ph-fill ph-brain" style="margin-right:4px"></i> Analyzing State...</div>
  <div id="aiConfidenceLabel" class="ai-confidence">Confidence: --%</div>
  <div id="aiStrategyLabel" class="ai-strategy">Strategy: Adaptive</div>
</div>

<!-- Phase 2: Enhanced Chain-of-Thought Panel -->
<div id="aiCotPanel" class="ai-cot-panel">
  <div id="aiCotThought" class="ai-cot-thought"><i class="ph-fill ph-brain" style="margin-right:4px"></i> Analyzing situation...</div>
  <div id="aiCotAction" class="ai-cot-action"></div>
  <div id="aiCotConfidence" class="ai-cot-confidence"></div>
  <div id="aiCotJudge" class="ai-cot-judge"></div>
</div>

<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
<button id="advisorBtn" class="start-btn hidden" style="background:linear-gradient(135deg,#10b981,#047857);box-shadow:0 4px 15px rgba(16,185,129,0.2), inset 0 1px 0 rgba(255,255,255,0.2);" onclick="askAdvisor()"><i class="ph-fill ph-robot"></i> Ask AI Advisor</button>
<button id="councilBtn" class="council-btn hidden" onclick="queryCouncil()"><i class="ph-fill ph-users-four"></i> AI Council</button>
</div>
<div style="display:flex;align-items:center;gap:10px;margin-left:auto;">
  <div id="modeBadge" class="mode-badge hidden"><i class="ph-fill ph-user"></i> Mode: Human Decision</div>
  <button id="soundToggleBtn" class="sound-toggle" onclick="toggleAudioState()" title="Toggle UI Sound"><i class="ph-fill ph-speaker-high" id="soundIcon"></i></button>
</div>
</div>
<span id="diffHint" class="diff-hint" style="margin-bottom:1.5rem">Easy Mode — Slower crisis progression, more forgiving resource depletion.</span>
<div id="advisorBox" class="feedback fb-info hidden" style="margin-bottom:1.5rem;border-color:rgba(16,185,129,0.5);background:rgba(16,185,129,0.1);color:#a7f3d0;font-size:1.05rem;box-shadow:0 10px 30px rgba(0,0,0,0.5)"></div>
<div id="statsPanel" class="hidden">
<div class="stats">
<div class="stat"><div class="stat-val pop-g"><span id="vPop">100</span><span id="tPop" class="trend-lbl"></span></div><div class="stat-lbl"><i class="ph-fill ph-users"></i> Population</div></div>
<div class="stat"><div class="stat-val food-g"><span id="vFood">200</span><span id="tFood" class="trend-lbl"></span></div><div class="stat-lbl"><i class="ph-fill ph-grains"></i> Food</div></div>
<div class="stat"><div class="stat-val energy-g"><span id="vEnergy">100</span><span id="tEnergy" class="trend-lbl"></span></div><div class="stat-lbl"><i class="ph-fill ph-lightning"></i> Energy</div></div>
<div class="stat"><div class="stat-val morale-g"><span id="vMorale">60</span><span id="tMorale" class="trend-lbl"></span></div><div class="stat-lbl"><i class="ph-fill ph-smiley"></i> Morale</div></div>
<div class="stat"><div class="stat-val tech-g"><span id="vTech">1</span><span id="tTech" class="trend-lbl"></span></div><div class="stat-lbl"><i class="ph-fill ph-microscope"></i> Tech</div></div>
<div class="stat"><div class="stat-val danger-g"><span id="vDanger">10</span><span id="tDanger" class="trend-lbl"></span></div><div class="stat-lbl"><i class="ph-fill ph-warning"></i> Danger</div></div>
<div class="stat"><div class="stat-val workers-g"><span id="vWorkers">30</span><span id="tWorkers" class="trend-lbl"></span></div><div class="stat-lbl"><i class="ph-fill ph-person-arms-spread"></i> Workers</div></div>
<div class="stat"><div class="stat-val progress-g"><span id="vProgress">0</span><span id="tProgress" class="trend-lbl"></span></div><div class="stat-lbl"><i class="ph-fill ph-trend-up"></i> Progress</div></div>
</div>
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1.5rem">
<span id="vEra" class="era-badge era-tribal"><i class="ph-fill ph-campfire"></i> TRIBAL ERA</span>
<span style="color:#94a3b8;font-size:.9rem;font-weight:600;font-family:'Outfit',sans-serif;">Turn <strong id="vTurn" style="color:#e2e8f0;font-size:1.1rem;display:inline-block">0</strong>/<span id="vMaxTurn">10</span> | Score: <strong id="vScore" style="color:#e2e8f0;font-size:1.1rem">0.00</strong></span>
</div>
<div id="memoryPanel" style="background:linear-gradient(135deg,rgba(30,41,59,0.8),rgba(15,23,42,0.9));border:1px solid rgba(255,255,255,0.05);border-radius:16px;padding:1.2rem 1.5rem;margin-bottom:1.5rem;display:flex;gap:2rem;align-items:center;flex-wrap:wrap;box-shadow:0 10px 25px rgba(0,0,0,0.5)">
<div style="display:flex;align-items:center;gap:0.8rem"><span style="font-size:1.6rem;color:#cvb5fd;filter:drop-shadow(0 0 8px rgba(255,255,255,0.2))" id="personalityIcon"><i class="ph-fill ph-scales"></i></span><div><div style="font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;font-weight:600">Leadership</div><strong id="vPersonality" style="color:#f8fafc;font-size:.95rem;font-family:'Outfit',sans-serif">Balanced</strong></div></div>
<div style="flex:1;min-width:120px"><div style="font-size:.75rem;color:#94a3b8;margin-bottom:6px;font-weight:600;display:flex;align-items:center;gap:4px"><i class="ph-fill ph-handshake"></i> Citizen Trust</div><div style="background:#0f172a;border-radius:8px;height:10px;overflow:hidden;box-shadow:inset 0 2px 4px rgba(0,0,0,0.5)"><div id="barCitizen" style="height:100%;background:linear-gradient(90deg,#ef4444,#eab308,#10b981);width:50%;transition:width 0.8s cubic-bezier(0.16,1,0.3,1)"></div></div></div>
<div style="flex:1;min-width:120px"><div style="font-size:.75rem;color:#94a3b8;margin-bottom:6px;font-weight:600;display:flex;align-items:center;gap:4px"><i class="ph-fill ph-shield-chevron"></i> Military Trust</div><div style="background:#0f172a;border-radius:8px;height:10px;overflow:hidden;box-shadow:inset 0 2px 4px rgba(0,0,0,0.5)"><div id="barMilitary" style="height:100%;background:linear-gradient(90deg,#ef4444,#eab308,#3b82f6);width:50%;transition:width 0.8s cubic-bezier(0.16,1,0.3,1)"></div></div></div>
<div style="text-align:center;flex:1.5;"><div style="font-size:.75rem;color:#94a3b8;font-weight:600;display:flex;align-items:center;gap:4px;justify-content:center"><i class="ph-fill ph-activity"></i> Stability Progress</div>
<div style="display:flex;align-items:center;gap:8px;"><div class="sub-progress"><div id="vStabBar" class="sub-fill" style="width:65%"></div></div><strong id="vStability" style="color:#10b981;font-size:1.1rem;font-family:'Outfit',sans-serif">65%</strong></div>
</div>
<div style="text-align:center"><div style="font-size:.75rem;color:#94a3b8;font-weight:600;display:flex;align-items:center;gap:4px;justify-content:center"><i class="ph-fill ph-hourglass-medium"></i> Pending</div><strong id="vPending" style="color:#fbbf24;font-size:1.1rem;font-family:'Outfit',sans-serif">0</strong></div>
</div>
<div id="chainsBox" style="display:none;margin-bottom:.8rem;display:flex;gap:.5rem;flex-wrap:wrap"></div>
<div id="predictionsBox" style="display:none;background:rgba(234,179,8,0.08);border:1px solid rgba(234,179,8,0.2);border-radius:10px;padding:0.6rem 1rem;margin-bottom:.8rem;font-size:.82rem;color:#fbbf24"></div>
<div id="narrativeBox" style="font-style:italic;color:#94a3b8;font-size:.85rem;margin-bottom:1rem;padding:0.5rem 0.8rem;border-left:3px solid rgba(99,102,241,0.4);display:none"></div>
</div>

<!-- Phase 2: AI Council Panel -->
<div id="councilPanel" class="council-panel"></div>

<!-- Phase 2: Causality Web -->
<div id="causalityWeb" class="causality-web">
  <div class="causality-header"><i class="ph-fill ph-graph"></i> DECISION CAUSALITY WEB</div>
  <canvas id="causalityCanvas" class="causality-canvas"></canvas>
  <div class="causality-legend">
    <span><span style="width:8px;height:8px;border-radius:50%;background:#10b981;display:inline-block"></span> Effective</span>
    <span><span style="width:8px;height:8px;border-radius:50%;background:#ef4444;display:inline-block"></span> Failed/Ignored</span>
    <span><span style="width:8px;height:8px;border-radius:50%;background:#fbbf24;display:inline-block"></span> Delayed Effect</span>
    <span style="color:#64748b">Lines = Causal Links</span>
  </div>
</div>

<!-- POLISH: Consequence Trail -->
<div id="consequenceTrail" class="consequence-trail"></div>

<!-- POLISH: Reward Visualization -->
<div id="rewardViz" class="reward-viz">
<div style="font-size:.78rem;font-weight:700;color:#94a3b8;margin-bottom:.5rem;display:flex;align-items:center;gap:6px"><i class="ph-fill ph-chart-line-up"></i> REWARD SYSTEM</div>
<div class="rv-row"><span>✅ Correct action</span><span class="rv-pos">+0.12</span></div>
<div class="rv-row"><span>🛡️ Crisis averted</span><span class="rv-pos">+0.25</span></div>
<div class="rv-row"><span>🏛️ Era unlocked</span><span class="rv-pos">+0.50</span></div>
<div class="rv-row"><span>📈 Population milestone</span><span class="rv-pos">+0.20</span></div>
<div class="rv-row"><span>❌ Wrong action</span><span class="rv-neg">-0.06</span></div>
<div class="rv-row"><span>⚠️ Ignore urgent</span><span class="rv-neg">-0.15</span></div>
<div class="rv-row"><span>💀 Collapse</span><span class="rv-neg">-1.00</span></div>
</div>

<!-- POLISH: Agent Benchmark Panel -->
<div id="benchPanel" class="bench-panel">
<div style="font-size:.85rem;font-weight:700;color:#e2e8f0;display:flex;align-items:center;gap:8px"><i class="ph-fill ph-chart-bar" style="color:#6366f1"></i> AI AGENT PERFORMANCE COMPARISON</div>
<div class="bench-grid">
<div class="bench-card"><div style="font-size:1.5rem;color:#ef4444"><i class="ph-fill ph-dice-three"></i></div><div style="font-weight:700;color:#f8fafc;margin:.3rem 0">Random</div><div id="benchRandom" style="font-size:1.4rem;font-weight:800;color:#ef4444">—</div><div style="font-size:.7rem;color:#94a3b8">Survival Rate</div><div class="bench-bar"><div id="benchRandomBar" class="bench-fill" style="width:0%;background:linear-gradient(90deg,#ef4444,#f97316)"></div></div></div>
<div class="bench-card"><div style="font-size:1.5rem;color:#eab308"><i class="ph-fill ph-coins"></i></div><div style="font-weight:700;color:#f8fafc;margin:.3rem 0">Greedy</div><div id="benchGreedy" style="font-size:1.4rem;font-weight:800;color:#eab308">—</div><div style="font-size:.7rem;color:#94a3b8">Survival Rate</div><div class="bench-bar"><div id="benchGreedyBar" class="bench-fill" style="width:0%;background:linear-gradient(90deg,#eab308,#84cc16)"></div></div></div>
<div class="bench-card"><div style="font-size:1.5rem;color:#10b981"><i class="ph-fill ph-brain"></i></div><div style="font-weight:700;color:#f8fafc;margin:.3rem 0">Logical</div><div id="benchLogical" style="font-size:1.4rem;font-weight:800;color:#10b981">—</div><div style="font-size:.7rem;color:#94a3b8">Survival Rate</div><div class="bench-bar"><div id="benchLogicalBar" class="bench-fill" style="width:0%;background:linear-gradient(90deg,#10b981,#06b6d4)"></div></div></div>
</div>
<div id="benchStatus" style="text-align:center;margin-top:.8rem;font-size:.75rem;color:#64748b"></div>
</div>

<div id="feedbackBox"></div>
<div id="messagesPanel" class="messages"></div>
<div id="actionsPanel" class="actions hidden">
<button class="act-btn a-food" onclick="act('allocate_food')"><i class="ph-fill ph-grains"></i> Allocate Food<span class="act-tip">Food +20 · Morale +10</span></button>
<button class="act-btn a-workers" onclick="act('allocate_workers')"><i class="ph-fill ph-users-three"></i> Assign Workers<span class="act-tip">Energy +15 · Workers -3</span></button>
<button class="act-btn a-research" onclick="act('approve_research')"><i class="ph-fill ph-flask"></i> Approve Research<span class="act-tip">Tech +1 · Progress +15</span></button>
<button class="act-btn a-defend" onclick="act('defend')"><i class="ph-fill ph-shield-sword"></i> Defend<span class="act-tip">Danger -15 · Energy -10</span></button>
<button class="act-btn a-calm" onclick="act('calm_citizens')"><i class="ph-fill ph-hands-praying"></i> Calm Citizens<span class="act-tip">Morale +15 · Danger -5</span></button>
<button class="act-btn a-trade-y" onclick="act('accept_trade')"><i class="ph-fill ph-handshake"></i> Accept Trade<span class="act-tip">Resources ↑ · Risk varies</span></button>
<button class="act-btn a-trade-n" onclick="act('reject_trade')"><i class="ph-fill ph-x-circle"></i> Reject Trade<span class="act-tip">Safe · May anger traders</span></button>
<button class="act-btn a-grow" onclick="act('invest_growth')"><i class="ph-fill ph-plant"></i> Invest<span class="act-tip">Pop +5 · Progress +10</span></button>
<button class="act-btn a-emergency" onclick="act('emergency_response')"><i class="ph-fill ph-warning-octagon"></i> Emergency<span class="act-tip">Handles disasters & disease</span></button>
<button class="act-btn a-ignore" onclick="act('ignore')"><i class="ph-fill ph-moon-stars"></i> Ignore<span class="act-tip">Risky — delayed consequences</span></button>
</div>
<div id="donePanel" class="hidden"></div>
<div class="footer-sig">Civil Command Center | AI Civilization Simulation Environment</div>

<!-- POLISH: Civilization Pulse Graph -->
<div id="pulseGraphContainer" class="pulse-graph-container" style="display:none">
<div style="font-size:.75rem;font-weight:700;color:#94a3b8;margin-bottom:6px;display:flex;align-items:center;gap:6px"><i class="ph-fill ph-heartbeat"></i> CIVILIZATION PULSE <span style="margin-left:auto;font-weight:400"><span style="color:#00ffaa">● Morale</span> <span style="color:#ff4444;margin-left:8px">● Danger</span></span></div>
<svg viewBox="0 0 100 50" preserveAspectRatio="none" style="width:100%;height:100px;">
  <path id="pulsePath" class="pulse-line" d=""/>
  <path id="dangerPath" class="pulse-line pulse-danger" d=""/>
</svg>
</div>

<!-- POLISH: Dataset Export Button -->
<button class="dataset-btn-raw" onclick="exportDataset()" style="display:none" id="exportBtn"><i class="ph-bold ph-download-simple"></i> Export Episode Log (JSON)</button>

<!-- POLISH: Cinematic End Overlay -->
<div id="endOverlay" class="end-overlay"></div>
</div>

<!-- POLISH: Era Transition Overlay -->
<div id="eraTransitionOverlay" class="era-transition-overlay">
  <div id="eraTransitionText" class="era-transition-text"></div>
</div>

<!-- POLISH: Matrix Data Stream -->
<button id="dataStreamBtn" class="data-stream-btn" onclick="toggleMatrix()" title="Toggle Data Stream"><i class="ph-bold ph-terminal-window"></i> Data Stream</button>
<div id="matrixStream" class="matrix-stream">
  <div style="color:#00ff66;font-weight:700;margin-bottom:8px;font-size:0.85rem;display:flex;align-items:center;gap:6px"><i class="ph-bold ph-terminal-window"></i> OPENENV DATA STREAM <button onclick="toggleMatrix()" style="margin-left:auto;background:none;border:1px solid rgba(0,255,100,0.2);color:#00ff66;padding:2px 8px;border-radius:4px;cursor:pointer;font-size:0.7rem">✕ Close</button></div>
  <div id="matrixContent"></div>
</div>

<!-- POLISH: Replay Viewer Overlay -->
<div id="replayOverlay" class="replay-overlay">
  <div style="text-align:center;max-width:500px">
    <div style="font-size:2rem;margin-bottom:1rem">🎬</div>
    <h2 style="font-family:'Outfit',sans-serif;font-size:1.5rem;margin-bottom:1rem;color:#e2e8f0">Episode Replay Viewer</h2>
    <p style="color:#94a3b8;margin-bottom:1.5rem;font-size:0.9rem">Upload a previously exported episode JSON log to replay the agent's decisions step-by-step.</p>
    <input type="file" id="replayUpload" accept=".json" onchange="handleReplayUpload(event)" style="display:block;margin:1rem auto;color:#94a3b8;font-family:inherit">
    <button onclick="closeReplay()" style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:#e2e8f0;padding:8px 24px;border-radius:8px;cursor:pointer;font-family:'Outfit',sans-serif;font-weight:600;margin-top:1rem">Cancel</button>
  </div>
</div>

<!-- POLISH: Status Badge -->
<div id="statusBadge" class="status-badge"><i class="ph-fill ph-pulse"></i> System: Stable</div>

<!-- POLISH: Telemetry Tag -->
<div id="telemetryTag" class="telemetry-tag"></div>

<!-- FEATURE 3: Neural Cartography Canvas -->
<div class="neuro-canvas-wrap" id="neuroCanvasWrap" style="display:none">
  <span class="neuro-label"><i class="ph-fill ph-dna"></i> Neural Cartography</span>
  <canvas id="neuroCanvas"></canvas>
</div>

<!-- FEATURE 1: Chrono-Fractal Temporal Rewind Overlay -->
<div id="chronoOverlay" class="chrono-overlay">
  <div class="chrono-header">
    <div class="chrono-title"><i class="ph-fill ph-clock-countdown"></i> CHRONO-FRACTAL TEMPORAL REWIND</div>
    <button class="chrono-close" onclick="closeChronoRewind()"><i class="ph-bold ph-x" style="margin-right:4px"></i> Close</button>
  </div>
  <div class="chrono-body">
    <div id="chronoFeedback" class="chrono-feedback"></div>
    <div id="chronoStateDisplay" class="chrono-state-display"></div>
    <div id="chronoActionDisplay" class="chrono-action-display"></div>
    <div id="chronoMessagesDisplay" class="chrono-messages-display"></div>
  </div>
  <div class="chrono-timeline-bar">
    <div class="chrono-slider-wrap" id="chronoSliderWrap">
      <input type="range" id="chronoSlider" class="chrono-slider" min="0" max="0" value="0" oninput="onChronoSlide(this.value)">
    </div>
    <div class="chrono-turn-info">
      <span id="chronoTurnLabel">Turn 0</span>
      <span id="chronoTurnMax">/ 0</span>
    </div>
  </div>
</div>

<!-- FEATURE 2: Tesseract Timeline Overlay -->
<div id="tesseractOverlay" class="tesseract-overlay">
  <div class="tesseract-header">
    <div class="tesseract-title"><i class="ph-fill ph-git-branch"></i> TESSERACT TIMELINE — Quantum Branching</div>
    <button class="chrono-close" onclick="closeTesseract()"><i class="ph-bold ph-x" style="margin-right:4px"></i> Close</button>
  </div>
  <div id="tesseractBody" class="tesseract-body"></div>
  <div class="tesseract-footer">
    <button class="chrono-close" onclick="runTesseractSim()" id="tesseractRunBtn"><i class="ph-fill ph-play" style="margin-right:4px"></i> Run Quantum Branch Simulation</button>
  </div>
</div>

<!-- FEATURE 4: Controlled Self-Evolution Toast -->
<div id="evoUnlockToast" class="evo-unlock-toast">
  <div class="evo-unlock-icon" id="evoIcon"><i class="ph-fill ph-dna" style="font-size:1.8rem;color:#a78bfa"></i></div>
  <div class="evo-unlock-title" id="evoTitle">New Capability Discovered</div>
  <div class="evo-unlock-desc" id="evoDesc">Your civilization has evolved beyond its current constraints.</div>
  <div class="evo-unlock-action" id="evoAction">Capability Unlocked</div>
</div>
<script>

// ═══════════════════════════════════════════════════════════════
// INTRO VIDEO HANDLER
// ═══════════════════════════════════════════════════════════════
function skipIntroVideo() {
  const overlay = document.getElementById('introVideoOverlay');
  const video = document.getElementById('introVideo');
  if (!overlay) return;
  overlay.classList.add('fade-out');
  if (video) { video.pause(); video.currentTime = 0; }
  setTimeout(() => {
    overlay.style.display = 'none';
    // Show the bulb screen after intro video
    const bulbScreen = document.getElementById('bulb-screen');
    if (bulbScreen) {
      bulbScreen.classList.add('active');
      createAmbientDust();
    }
  }, 900);
}
// Auto-dismiss when video ends
document.addEventListener('DOMContentLoaded', function() {
  const video = document.getElementById('introVideo');
  if (video) {
    video.addEventListener('ended', function() { skipIntroVideo(); });
    // Fallback: if video fails to load, auto-skip after 3s
    video.addEventListener('error', function() {
      console.warn('[Intro] Video failed to load, skipping...');
      skipIntroVideo();
    });
  }
});

// Skip ALL 3 stages at once (only available on first video)
function skipAllStages() {
  const introOverlay = document.getElementById('introVideoOverlay');
  const introVideo = document.getElementById('introVideo');
  const bulbScreen = document.getElementById('bulb-screen');
  const evoScreen = document.getElementById('evo-screen');
  const evoVid = document.getElementById('evo-vid');
  
  // Stop all videos
  if (introVideo) { introVideo.pause(); introVideo.currentTime = 0; }
  if (evoVid) { evoVid.pause(); evoVid.currentTime = 0; }
  
  // Hide all overlays immediately
  if (introOverlay) { introOverlay.style.display = 'none'; }
  if (bulbScreen) { bulbScreen.style.display = 'none'; bulbScreen.style.opacity = '0'; }
  if (evoScreen) { evoScreen.style.display = 'none'; evoScreen.style.opacity = '0'; }
  
  // Prevent any pending bulb/evo transitions
  bulbIgnited = true;
  evoStarted = true;
  isSkippedAll = true;
  
  // Go straight to main platform
  revealMainDashboard();
}

// ═══════════════════════════════════════════════════════════════
// ELITE POLISH SCRIPTS
// ═══════════════════════════════════════════════════════════════
function toggleMatrix() { document.getElementById('matrixStream').classList.toggle('active'); }
function syntaxHighlight(json) {
  if (typeof json != 'string') json = JSON.stringify(json, undefined, 2);
  json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
    let cls = 'matrix-num';
    if (/^"/.test(match)) { if (/:$/.test(match)) { cls = 'matrix-key'; } else { cls = 'matrix-str'; } } else if (/true|false/.test(match)) { cls = 'matrix-bool'; }
    return '<span class="' + cls + '">' + match + '</span>';
  });
}
function logToMatrix(method, endpoint, data, latency) {
  const t = document.getElementById('telemetryTag');
  if(t) { t.textContent = `[${method} ${endpoint}] → ${latency}ms`; t.style.opacity = 1; setTimeout(()=>t.style.opacity=0, 2000); }
  const mc = document.getElementById('matrixContent');
  if(!mc) return;
  const d = new Date();
  const time = `${d.getHours()}:${d.getMinutes()}:${d.getSeconds()}.${d.getMilliseconds()}`;
  const block = document.createElement('div'); block.className = 'matrix-line';
  block.innerHTML = `<span style="color:#555;">[${time}] ${method} ${endpoint}</span><br><pre style="margin:5px 0 0 0; white-space:pre-wrap; word-wrap:break-word;">` + syntaxHighlight(data) + `</pre>`;
  mc.appendChild(block);
  if(mc.childNodes.length > 50) mc.removeChild(mc.firstChild);
  if(mc.parentElement) mc.parentElement.scrollTop = mc.parentElement.scrollHeight;
}

function animateValue(obj, start, end, duration) {
  if(!obj) return;
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const val = start + ease * (end - start);
    obj.innerHTML = Number.isInteger(end) ? Math.round(val) : val.toFixed(2);
    if (progress < 1) window.requestAnimationFrame(step); else obj.innerHTML = end;
  };
  window.requestAnimationFrame(step);
}

let lastScore = 0; let lastEra = ""; let stateHistory = []; let episodeLog = [];

function triggerEraTransition(newEra) {
  const ov = document.getElementById('eraTransitionOverlay'); const txt = document.getElementById('eraTransitionText');
  if(!ov || !txt) return;
  txt.textContent = newEra.toUpperCase() + " ERA";
  ov.classList.add('active');
  playEndSound(false);
  setTimeout(() => ov.classList.remove('active'), 2500);
}

function drawPulseGraph() {
  const maxPts = 30; const pts = stateHistory.slice(-maxPts);
  if(pts.length < 2) return;
  const h = 50; const w = 100;
  let dMorale = ""; let dDanger = "";
  pts.forEach((pt, i) => {
    const x = (i / (maxPts-1)) * w;
    const mVal = Number(pt.morale) || 0;
    const dVal = Number(pt.danger_level) || 0;
    const yM = h - (mVal / 100) * h;
    const yD = h - (dVal / 100) * h;
    dMorale += (i===0?'M':'L') + `${x},${yM} `; dDanger += (i===0?'M':'L') + `${x},${yD} `;
  });
  const pPath = document.getElementById('pulsePath'); const dPath = document.getElementById('dangerPath');
  if(pPath) pPath.setAttribute('d', dMorale); if(dPath) dPath.setAttribute('d', dDanger);
}

function exportDataset() {
  if (!episodeLog || episodeLog.length === 0) {
    showToast('No episode data to export. Play at least one turn first.', 'info');
    return;
  }
  try {
    const jsonStr = JSON.stringify(episodeLog, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const el = document.createElement('a');
    el.href = url;
    el.download = 'civ_episode_log_' + new Date().toISOString().slice(0,10) + '.json';
    document.body.appendChild(el);
    el.click();
    setTimeout(() => { URL.revokeObjectURL(url); el.remove(); }, 100);
    showToast('Episode log exported successfully!', 'good');
  } catch(e) {
    console.error('Export error:', e);
    showToast('Export failed: ' + e.message, 'bad');
  }
}

function openReplay() { document.getElementById('replayOverlay').classList.add('active'); }
function closeReplay() { document.getElementById('replayOverlay').classList.remove('active'); document.getElementById('replayUpload').value = ''; }
async function handleReplayUpload(e) {
  const file = e.target.files[0]; if(!file) return;
  const text = await file.text();
  try {
    const log = JSON.parse(text); if(!Array.isArray(log) || log.length === 0) throw "Invalid log";
    closeReplay();
    skipAllIntros();
    let i = 0;
    const playNext = async () => {
      if(i >= log.length) { alert('Replay complete.'); return; }
      const frame = log[i];
      if(frame.state) {
        document.getElementById('vPop').innerHTML = frame.state.population;
        document.getElementById('vFood').innerHTML = frame.state.food;
        document.getElementById('vEnergy').innerHTML = frame.state.energy;
        document.getElementById('vMorale').innerHTML = frame.state.morale;
      }
      if(frame.action) {
        const tb = document.getElementById('aiThinkingBox');
        if(tb) { tb.classList.remove('hidden'); tb.innerHTML = '<i class="ph-fill ph-play-circle"></i> Replay Action: <strong>'+frame.action+'</strong>'; }
      }
      i++; setTimeout(playNext, 1000);
    };
    playNext();
  } catch(err) { alert('Could not parse episode log.'); closeReplay(); }
}

async function determineLLMAction(state, msgData) {
  // Call backend AI decision endpoint (uses NVIDIA NIM / DeepSeek with fallback)
  try {
    const pop = parseInt(state.population) || 100;
    const food = parseInt(state.food) || 100;
    const energy = parseInt(state.energy) || 100;
    const morale = parseInt(state.morale) || 60;
    const danger = parseInt(state.danger) || 0;
    const msgText = (msgData && msgData.length > 0) ? msgData[0].text : '';
    
    const r = await fetch('/api/ai/decide', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({population: pop, food: food, energy: energy, morale: morale, danger: danger, messages: msgText})
    });
    const d = await r.json();
    if (d.decision) {
      console.log('[AI Decide]', d.source, ':', d.decision);
      return d.decision;
    }
    return null;
  } catch(e) {
    console.log('AI decide fallback:', e);
    // Inline fallback if backend unreachable
    const danger = parseInt(state.danger) || 0;
    const food = parseInt(state.food) || 100;
    const morale = parseInt(state.morale) || 60;
    if (danger >= 15) return 'defend | 95% | Extreme threat';
    if (food < 40) return 'allocate_food | 92% | Critical food shortage';
    if (morale < 25) return 'calm_citizens | 88% | Civil unrest';
    return 'invest_growth | 70% | Stable conditions';
  }
}

let sid=null,selMsg=null;
let isAudioEnabled = true;

function toggleAudioState() {
  isAudioEnabled = !isAudioEnabled;
  const i = document.getElementById('soundIcon');
  if(isAudioEnabled){ i.className='ph-fill ph-speaker-high'; i.parentElement.style.color='#e2e8f0'; }
  else{ i.className='ph-fill ph-speaker-slash'; i.parentElement.style.color='#ef4444'; }
}

function updateDiffHint() {
  const hints = {
    'task_demo_5': '🚀 Quick 5-turn AI benchmark environment.',
    'task_demo_10': '⚡ 10-turn rapid setup for testing mechanics.',
    'task_easy': '🟢 Easy Mode: Slower crisis progression, more forgiving resource depletion.',
    'task_medium': '🟡 Medium Mode: Standard crisis progression, balanced resource decay.',
    'task_hard': '🔴 Hard Mode: Brutal crisis progression, fast resource decay, strict memory penalties.'
  };
  document.getElementById('diffHint').textContent = hints[document.getElementById('taskSel').value] || '';
}

async function resetGame(){
  try {
    const t=document.getElementById('taskSel').value;
    console.log('Resetting with task:', t);
    
    // Switch to playing mode
    const appDiv = document.getElementById('mainAppDiv');
    appDiv.classList.remove('app-landing');
    appDiv.classList.add('app-playing');
    const _clicker = document.getElementById('cinematic-clicker');
    if (_clicker) _clicker.style.pointerEvents = 'none';
    const _label = document.getElementById('cinematic-label');
    if (_label) _label.classList.add('hidden-cinema');
    const _pCanvas = document.getElementById('particleCanvas');
    if (_pCanvas) _pCanvas.classList.add('hidden-particles');
    try { if(window._cinematicInstance) window._cinematicInstance.jumpToLast(); } catch(e) {}
    const r=await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:t})});
    if(!r.ok){const err=await r.text();console.error('Reset failed:',err);alert('Reset failed: '+err);return;}
    const d=await r.json();
    console.log('Reset response:', d);
    sid=d.session_id;selMsg=null;
    document.getElementById('statsPanel').classList.remove('hidden');
    document.getElementById('actionsPanel').classList.remove('hidden');
    document.getElementById('advisorBtn').classList.remove('hidden');
    const demoBtn = document.getElementById('demoPlayBtn');
    if (demoBtn) demoBtn.classList.remove('hidden');
    document.getElementById('modeBadge').classList.remove('hidden');
    document.getElementById('soundToggleBtn').classList.remove('hidden');
    document.getElementById('advisorBox').classList.add('hidden');
    document.getElementById('donePanel').classList.add('hidden');
    document.getElementById('rewardViz').classList.add('visible');
    document.getElementById('endOverlay').className='end-overlay';
    document.getElementById('endOverlay').innerHTML='';
    aiPlaying=false;
    stateHistory=[]; episodeLog=[];
    // Push initial state so Chrono Rewind works after first action
    if(d && d.observation) { episodeLog.push({time:Date.now(), type:'reset', action:'—', reward:null, state:d.observation}); }
    causalityNodes=[]; causalityLinks=[];
    if(document.getElementById('demoPlayBtn')) {
        document.getElementById('demoPlayBtn').innerHTML='<i class="ph-bold ph-lightning"></i> AI Auto-Play';
        document.getElementById('demoPlayBtn').classList.remove('playing');
    }
    document.getElementById('aiThinkingBox').classList.add('hidden');
    document.getElementById('aiCotPanel').classList.remove('active');
    document.getElementById('councilPanel').classList.remove('active');
    document.getElementById('councilBtn').classList.remove('hidden');
    document.getElementById('causalityWeb').classList.remove('active');
    document.getElementById('pulsePath')?.setAttribute('d', '');
    document.getElementById('dangerPath')?.setAttribute('d', '');
    const pgc = document.getElementById('pulseGraphContainer');
    if(pgc) pgc.style.display = 'block';
    const expBtn = document.getElementById('exportBtn');
    if(expBtn) expBtn.style.display = 'block';
    const cBtn = document.getElementById('chronoBtn');
    if(cBtn) cBtn.style.display = 'block';
    try { logToMatrix('POST', '/reset', d, 0); } catch(e){}
    document.getElementById('modeBadge').innerHTML='<i class="ph-fill ph-user"></i> Mode: Human Decision';
    document.getElementById('modeBadge').style.color='#94a3b8';
    if (!document.querySelector('.starfield-bg')) {
      const sf = document.createElement('div');
      sf.className = 'starfield-bg';
      document.body.prepend(sf);
    }
    updateUI(d);
  } catch(e) {
    console.error('Reset error:', e);
    alert('Error starting game: ' + e.message);
  }
}

async function act(a){
  try {
    if(!sid){console.log('No session');return;}
    const body={action_type:a};
    if(selMsg)body.target_message_id=selMsg;
    console.log('Step action:', a);
    const _t0 = performance.now();
    const r=await fetch('/step/'+sid,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!r.ok){const err=await r.text();console.error('Step failed:',err);return;}
    const d=await r.json();
    const _lat = Math.round(performance.now()-_t0);
    try { logToMatrix('POST', `/step/${sid}`, d, _lat); } catch(e){}
    episodeLog.push({time:Date.now(), type:'step', action:a, reward:d.reward, state:d.observation});
    selMsg=null;updateUI(d);
    // Phase 2: Add node to causality web
    try {
      const obs = d.observation || {};
      addCausalityNode(obs.turn || 0, a, obs.last_action_effective !== false, obs.pending_consequences || 0);
    } catch(e) {}
    if(d.done){sid=null;showDone(d);}
  } catch(e) {
    console.error('Step error:', e);
  }
}

// Track previous stats for micro-animations
let prevStats = {};
function animateStat(id, newVal) {
  const el = document.getElementById(id);
  const trendEl = document.getElementById('t' + id.substring(1));
  const old = prevStats[id];
  if (old !== undefined && old !== newVal) {
    el.parentElement.classList.remove('stat-bump','stat-up','stat-down');
    void el.parentElement.offsetWidth;
    el.parentElement.classList.add('stat-bump');
    el.parentElement.classList.add(newVal > old ? 'stat-up' : 'stat-down');
    playTick(newVal > old);
    setTimeout(() => { el.parentElement.classList.remove('stat-bump','stat-up','stat-down'); }, 800);
    
    // Set explicit trend labels — persistent burn rate indicators
    if (trendEl) {
      const diff = newVal - old;
      trendEl.textContent = diff > 0 ? ('▲'+diff) : ('▼'+Math.abs(diff));
      trendEl.className = 'trend-lbl trend-persist ' + (diff > 0 ? 'trend-up' : 'trend-down');
      // Keep visible — no fade-out (persistent burn rates)
    }
  }
  prevStats[id] = newVal;
}

function updateUI(d){
  const o=d.observation;
  stateHistory.push(o);
  try { drawPulseGraph(); } catch(e){}

  animateStat('vPop', o.population);
  animateStat('vFood', o.food);
  animateStat('vEnergy', o.energy);
  animateStat('vMorale', o.morale);
  animateStat('vTech', o.technology_level);
  animateStat('vDanger', o.danger_level);
  animateStat('vWorkers', o.available_workers);
  document.getElementById('vPop').textContent=o.population;
  document.getElementById('vFood').textContent=o.food;
  document.getElementById('vEnergy').textContent=o.energy;
  document.getElementById('vMorale').textContent=o.morale;
  document.getElementById('vTech').textContent=o.technology_level;
  document.getElementById('vDanger').textContent=o.danger_level;
  document.getElementById('vWorkers').textContent=o.available_workers;
  document.getElementById('vProgress').textContent=Math.round(o.progress_score);
  document.getElementById('vTurn').textContent=o.turn;
  document.getElementById('vMaxTurn').textContent=o.max_turns;
  document.getElementById('vScore').textContent=(o.total_reward||0).toFixed(2);

  // POLISH: Consequence trail visualization
  const ct = document.getElementById('consequenceTrail');
  if (o.pending_consequences > 0 && o.active_chains && o.active_chains.length > 0) {
    ct.classList.add('visible');
    ct.innerHTML = '<span style="font-weight:700;color:#fbbf24;margin-right:6px"><i class="ph-fill ph-warning"></i> Delayed Effects:</span>' +
      o.active_chains.map(c => '<span class="ct-step" style="background:rgba(239,68,68,0.15);color:#fca5a5">' + c.replace('_',' ') + '</span>').join('<span class="ct-arrow">→</span>') +
      '<span class="ct-step" style="background:rgba(234,179,8,0.15);color:#fde047">' + o.pending_consequences + ' pending</span>';
  } else { ct.classList.remove('visible'); }

  const eb=document.getElementById('vEra');
  eb.textContent=o.era.toUpperCase()+' ERA';
  eb.className='era-badge era-'+o.era;

  // Memory panel
  var pIcons={military:'<i class="ph-fill ph-sword"></i>',scientific:'<i class="ph-fill ph-microscope"></i>',diplomatic:'<i class="ph-fill ph-handshake"></i>',economic:'<i class="ph-fill ph-coins"></i>',balanced:'<i class="ph-fill ph-scales"></i>',neglectful:'<i class="ph-fill ph-moon-stars"></i>'};
  var p=o.personality||'balanced';
  document.getElementById('personalityIcon').innerHTML=pIcons[p]||'<i class="ph-fill ph-scales"></i>';
  document.getElementById('vPersonality').textContent=p.charAt(0).toUpperCase()+p.slice(1);
  document.getElementById('barCitizen').style.width=(o.citizen_trust||50)+'%';
  document.getElementById('barMilitary').style.width=(o.military_trust||50)+'%';
  document.getElementById('vPending').textContent=o.pending_consequences||0;
  var stab=Math.round((o.stability_score||0.65)*100);
  var stabEl=document.getElementById('vStability');
  var trendArr={'rising':'<i class="ph-bold ph-trend-up"></i>','falling':'<i class="ph-bold ph-trend-down"></i>','stable':'<i class="ph-bold ph-minus"></i>'};
  var trnd=trendArr[o.stability_trend]||'<i class="ph-bold ph-minus"></i>';
  stabEl.innerHTML=stab+'% '+trnd;
  stabEl.style.color=stab>65?'#10b981':stab>35?'#eab308':'#ef4444';

  // Turn counter animation
  const turnEl = document.getElementById('vTurn');
  if (prevStats['turn'] !== undefined && prevStats['turn'] !== o.turn) {
    turnEl.classList.remove('turn-anim');
    void turnEl.offsetWidth;
    turnEl.classList.add('turn-anim');
  }
  prevStats['turn'] = o.turn;

  // Active event chains
  var cb=document.getElementById('chainsBox');
  if(o.active_chains&&o.active_chains.length>0){
    cb.style.display='flex';
    cb.innerHTML=o.active_chains.map(function(c){
      var icons={drought:'<i class="ph-fill ph-sun"></i>',plague:'<i class="ph-fill ph-virus"></i>',border_war:'<i class="ph-fill ph-crosshair"></i>',tech_boom:'<i class="ph-fill ph-rocket-launch"></i>',golden_age:'<i class="ph-fill ph-star"></i>'};
      return '<span style="background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.3);border-radius:8px;padding:5px 12px;font-size:.78rem;font-weight:600;color:#fca5a5;display:flex;align-items:center;gap:6px;box-shadow:0 4px 10px rgba(0,0,0,0.3)">'+(icons[c]||'<i class="ph-fill ph-lightning"></i>')+c.replace('_',' ')+'</span>';
    }).join('');
  } else {cb.style.display='none';}

  // Strategic predictions
  var pb=document.getElementById('predictionsBox');
  if(o.predictions&&o.predictions.length>0){
    pb.style.display='block';
    pb.innerHTML='<div style="display:flex;align-items:center;gap:6px;font-weight:700;margin-bottom:6px;color:#facc15"><i class="ph-fill ph-chart-bar"></i> Strategic Intel:</div><div style="line-height:1.6;color:#fde047">'+o.predictions.join('<br>')+'</div>';
  } else {pb.style.display='none';}

  var nb=document.getElementById('narrativeBox');
  if(o.narrative_summary && o.narrative_summary.length>5){
    // Sync population number in narrative text with actual population value
    var fixedNarr = o.narrative_summary.replace(/\b(of|with|has)\s+\d+\s+(people|citizens|souls|inhabitants)/gi, '$1 ' + o.population + ' $2');
    nb.innerHTML='<i class="ph-fill ph-scroll" style="vertical-align:middle;margin-right:6px"></i>'+fixedNarr;
    nb.style.display='block';
  } else { nb.style.display='none'; }

  // Toast notification system (replaces persistent inline banners)
  const fb=document.getElementById('feedbackBox');
  fb.innerHTML=''; // Always clear inline — toasts handle feedback now
  if(o.message){
    const toastType=o.last_action_effective===true?'good':o.last_action_effective===false?'bad':'info';
    showToast(o.message, toastType);
  }

  const mp=document.getElementById('messagesPanel');
  if(o.messages && o.messages.length>0){
    mp.innerHTML='';
    o.messages.forEach(function(m){
      var card=document.createElement('div');
      var dotColors={'critical':'🔴','high':'🟠','medium':'🟡','low':'🟢'};
      var mDot = dotColors[m.urgency] || '⚪';
      card.className='msg-card msg-'+m.urgency;
      // Faction color-coding
      var factionClass = getFactionClass(m.source, m.subject, m.body);
      if (factionClass) card.className += ' ' + factionClass;
      if(selMsg===m.id) card.className+=' selected';
      card.setAttribute('data-id', m.id);
      card.onclick=function(){selectMsg(m.id);};
      var factionIcon = getFactionIcon(factionClass);
      card.innerHTML='<div class="msg-header"><span class="msg-source">'+factionIcon+m.source+' — '+m.sender_name+'</span><span class="msg-urgency u-'+m.urgency+'">'+m.urgency+'</span></div><div class="msg-subject"><span style="opacity:0.6;font-size:0.8em;font-weight:400;margin-right:6px">[Turn '+o.turn+']</span>'+mDot+' '+m.subject+'</div><div class="msg-body">'+m.body+'</div>';
      mp.appendChild(card);
    });
  } else if(!o.done){
    mp.innerHTML='<div style="color:#64748b;text-align:center;padding:2rem">No messages this turn.</div>';
  } else {mp.innerHTML='';}
}

function selectMsg(id){
  selMsg=(selMsg===id)?null:id;
  document.querySelectorAll('.msg-card').forEach(function(c){
    if(c.getAttribute('data-id')===id && selMsg===id){
      c.classList.add('selected');
    } else {
      c.classList.remove('selected');
    }
  });
}

function showDone(d){
  document.getElementById('actionsPanel').classList.add('hidden');
  document.getElementById('messagesPanel').innerHTML='';
  document.getElementById('advisorBtn').classList.add('hidden');
  if(document.getElementById('demoPlayBtn')) document.getElementById('demoPlayBtn').classList.add('hidden');
  document.getElementById('advisorBox').classList.add('hidden');
  document.getElementById('rewardViz').classList.remove('visible');
  document.getElementById('consequenceTrail').classList.remove('visible');
  aiPlaying=false;
  if(document.getElementById('demoPlayBtn')) document.getElementById('demoPlayBtn').classList.remove('playing');
  var s=d.episode_summary||{};
  var mem=s.memory||{};
  var isCollapse = s.collapse;

  // POLISH: Play cinematic end sound
  playEndSound(isCollapse);

  // POLISH: Show cinematic overlay
  // Phase 2: Save run result for multi-run comparison
  try { saveRunResult(s); } catch(e) {}

  // SYSTEM 3: Auto-save to self-learning memory
  try {
    fetch('/api/memory/save-run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        survived: !isCollapse,
        total_reward: s.total_reward || 0,
        era_final: s.era_final || 'tribal',
        stability_score: (mem.stability_score || 0.5),
        collapse_reason: s.collapse_reason || '',
        key_lesson: isCollapse ? ('Collapsed: ' + (s.collapse_reason||'unknown')) : ('Survived to ' + (s.era_final||'tribal') + ' era'),
        mode: aiPlaying ? 'ai_auto' : 'human',
        turns_survived: s.turns_survived || 0,
        population_final: s.population_final || 0,
        crises_averted: s.crises_averted || 0,
        crises_failed: s.crises_failed || 0,
        personality: mem.personality || 'balanced',
      })
    }).then(r => r.json()).then(d => {
      console.log('[Memory] Run saved, total entries:', d.total_entries);
      updateMemoryBadge();
    }).catch(() => {});
  } catch(e) {}

  var ov = document.getElementById('endOverlay');
  ov.className = 'end-overlay ' + (isCollapse ? 'end-overlay-collapse' : 'end-overlay-victory');
  ov.innerHTML = '<div class="end-content">' +
    '<div class="end-icon">' + (isCollapse ? '💀' : '🏆') + '</div>' +
    '<div class="end-title ' + (isCollapse ? 'end-collapse-title' : 'end-victory-title') + '">' + (isCollapse ? 'CIVILIZATION COLLAPSED' : 'CIVILIZATION ENDURES') + '</div>' +
    '<div class="end-reason">' + (isCollapse ? (s.collapse_reason || 'Your civilization could not survive.') : (s.narrative || 'Your civilization has survived and grown under your leadership.')) + '</div>' +
    '<div class="end-stats-grid">' +
      '<div class="end-stat"><div class="end-stat-val progress-g">' + ((s.total_reward||0).toFixed(2)) + '</div><div class="end-stat-lbl">Total Score</div></div>' +
      '<div class="end-stat"><div class="end-stat-val pop-g">' + (s.population_final||0) + '</div><div class="end-stat-lbl">Final Pop</div></div>' +
      '<div class="end-stat"><div class="end-stat-val tech-g">' + (s.era_final||'tribal').toUpperCase() + '</div><div class="end-stat-lbl">Era Reached</div></div>' +
      '<div class="end-stat"><div class="end-stat-val" style="color:#10b981">' + Math.round((mem.stability_score||0)*100) + '%</div><div class="end-stat-lbl">Stability</div></div>' +
    '</div>' +
    '<div class="end-stats-grid">' +
      '<div class="end-stat"><div class="end-stat-val morale-g">' + (s.crises_averted||0) + '</div><div class="end-stat-lbl">Crises Averted</div></div>' +
      '<div class="end-stat"><div class="end-stat-val danger-g">' + (s.crises_failed||0) + '</div><div class="end-stat-lbl">Crises Failed</div></div>' +
      '<div class="end-stat"><div class="end-stat-val workers-g">' + (s.correct_actions||0) + '</div><div class="end-stat-lbl">Tactical Wins</div></div>' +
      '<div class="end-stat"><div class="end-stat-val" style="color:#a855f7">' + (mem.personality||'balanced').toUpperCase() + '</div><div class="end-stat-lbl">Leadership</div></div>' +
    '</div>' +
    '<!-- Phase 2: Behavioral Radar Chart placeholder -->' +
    '<div class="radar-container" id="endRadarContainer">' +
      '<div class="radar-title"><i class="ph-fill ph-chart-polar"></i> BEHAVIORAL PROFILE</div>' +
      '<canvas id="endRadarCanvas" class="radar-canvas"></canvas>' +
      '<div id="endRadarAlignment" class="radar-alignment"></div>' +
      '<div id="endRadarMetrics" class="radar-metrics"></div>' +
    '</div>' +
    '<!-- Phase 2: Chronicle placeholder -->' +
    '<div id="endChronicle" class="chronicle-panel" style="display:none"></div>' +
    '<!-- Phase 2: Multi-Run Comparison -->' +
    '<div id="endMultiRun"></div>' +
    '<button class="end-play-again" onclick="dismissEnd();resetGame()"><i class="ph-bold ph-arrows-clockwise" style="margin-right:8px"></i> Play Again</button>' +
  '</div>';
  setTimeout(() => ov.classList.add('active'), 50);

  // Phase 2: Fetch and render behavioral profile radar chart
  (async () => {
    try {
      const profile = await fetchBehavioralProfile(s);
      if (profile && profile.radar) {
        drawRadarChart('endRadarCanvas', profile.radar, 260);
        const alEl = document.getElementById('endRadarAlignment');
        if (alEl) alEl.innerHTML = '<i class="ph-fill ph-fingerprint"></i> AI Alignment: ' + (profile.alignment || 'Unknown');
        const metEl = document.getElementById('endRadarMetrics');
        if (metEl && profile.metrics) {
          metEl.innerHTML = Object.entries(profile.metrics).map(([k,v]) =>
            '<div class="radar-metric"><div class="radar-metric-val" style="color:' + (v > 60 ? '#10b981' : v > 30 ? '#fbbf24' : '#ef4444') + '">' + Math.round(v) + '</div><div class="radar-metric-lbl">' + k.replace(/_/g,' ') + '</div></div>'
          ).join('');
        }
      }
    } catch(e) { console.log('Profile error:', e); }
  })();

  // Phase 2: Generate and display AI Chronicle
  (async () => {
    try {
      const chron = await generateChronicle(s);
      const el = document.getElementById('endChronicle');
      if (el && chron && chron.chronicle) {
        el.style.display = 'block';
        el.innerHTML = '<div class="chronicle-title"><i class="ph-fill ph-scroll"></i> THE CHRONICLE</div>' +
          '<div class="chronicle-text">' + chron.chronicle + '</div>' +
          '<div class="chronicle-source">Generated by: ' + (chron.source || 'unknown') + '</div>';
      }
    } catch(e) { console.log('Chronicle error:', e); }
  })();

  // Phase 2: Multi-Run Comparison table
  try {
    const mrEl = document.getElementById('endMultiRun');
    if (mrEl) mrEl.innerHTML = getMultiRunHTML();
  } catch(e) {}

  // Also keep the inline done panel as fallback
  var dp=document.getElementById('donePanel');
  dp.classList.remove('hidden');
  dp.innerHTML='<div class="done-card"><h2><i class="ph-fill '+(isCollapse?'ph-skull':'ph-confetti')+'" style="margin-right:10px"></i>'+(isCollapse?'Civilization Collapsed':'Episode Complete!')+'</h2>'
    +(s.collapse_reason?'<p style="color:#fca5a5;margin-bottom:1.5rem;font-size:1.1rem">'+s.collapse_reason+'</p>':'')
    +(s.narrative?'<p style="font-style:italic;color:#cbd5e1;margin-bottom:1.5rem;font-size:1.05rem"><i class="ph-fill ph-scroll" style="margin-right:6px"></i>'+s.narrative+'</p>':'')
    +'<div class="stats" style="margin-top:1.5rem"><div class="stat"><div class="stat-val progress-g">'+((s.total_reward||0).toFixed(2))+'</div><div class="stat-lbl">Total Score</div></div>'
    +'<div class="stat"><div class="stat-val pop-g">'+(s.population_final||0)+'</div><div class="stat-lbl">Final Pop</div></div>'
    +'<div class="stat"><div class="stat-val tech-g" style="font-size:1.3rem">'+(s.era_final||'tribal').toUpperCase()+'</div><div class="stat-lbl">Era Reached</div></div>'
    +'<div class="stat"><div class="stat-val morale-g">'+(s.crises_averted||0)+'</div><div class="stat-lbl">Crises Averted</div></div>'
    +'<div class="stat"><div class="stat-val danger-g">'+(s.crises_failed||0)+'</div><div class="stat-lbl">Crises Failed</div></div>'
    +'<div class="stat"><div class="stat-val workers-g">'+(s.correct_actions||0)+'</div><div class="stat-lbl">Tactical Wins</div></div>'
    +'<div class="stat"><div class="stat-val" style="color:#10b981">'+Math.round((mem.stability_score||0)*100)+'%</div><div class="stat-lbl">Stability</div></div>'
    +'<div class="stat"><div class="stat-val" style="color:#a855f7;font-size:1.3rem">'+(mem.personality||'balanced').toUpperCase()+'</div><div class="stat-lbl">Personality</div></div></div>'
    +'<button class="start-btn" style="margin-top:2rem;width:100%;justify-content:center" onclick="dismissEnd();resetGame()"><i class="ph-bold ph-arrows-clockwise"></i> Play Again</button></div>';

  // POLISH: Run agent benchmark after game ends
  runBenchmark();
}

async function askAdvisor() {
    if (!sid) return;
    const advisorBtn = document.getElementById('advisorBtn');
    const advisorBox = document.getElementById('advisorBox');
    advisorBtn.disabled = true;
    advisorBtn.innerHTML = '<i class="ph-bold ph-spinner ph-spin"></i> Consulting Oracle...';
    advisorBox.classList.remove('hidden');
    advisorBox.innerHTML = '<i class="ph-fill ph-robot" style="margin-right:8px;vertical-align:middle"></i> <span class="think-spinner" style="display:inline-block"></span> Connecting to AI Oracle...';
    
    const pop = parseInt(document.getElementById('vPop').textContent) || 100;
    const food = parseInt(document.getElementById('vFood').textContent) || 100;
    const energy = parseInt(document.getElementById('vEnergy').textContent) || 100;
    const morale = parseInt(document.getElementById('vMorale').textContent) || 60;
    const danger = parseInt(document.getElementById('vDanger').textContent) || 0;
    const era = document.getElementById('vEra').textContent || 'TRIBAL ERA';
    
    let msgs = [];
    document.querySelectorAll('.msg-subject').forEach(el => msgs.push(el.textContent));
    const msgText = msgs.length > 0 ? 'Current events: ' + msgs.join(', ') : 'All quiet.';
    
    try {
        const r = await fetch('/api/ai/advisor', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({population: pop, food: food, energy: energy, morale: morale, danger: danger, era: era, events: msgText})
        });
        const d = await r.json();
        const srcBadge = d.source === 'llm' ? '<span style="font-size:.65rem;background:rgba(168,85,247,0.2);color:#c084fc;padding:2px 6px;border-radius:4px;margin-left:8px">AI LLM</span>' : '<span style="font-size:.65rem;background:rgba(234,179,8,0.2);color:#fde047;padding:2px 6px;border-radius:4px;margin-left:8px">Rule-Based</span>';
        advisorBox.innerHTML = '<strong style="color:#34d399"><i class="ph-fill ph-robot" style="margin-right:6px"></i> Oracle:</strong>' + srcBadge + '<br>' + d.advice;
    } catch (e) {
        advisorBox.innerHTML = '<strong style="color:#fb7185"><i class="ph-fill ph-warning" style="margin-right:6px"></i> Oracle:</strong> Communication severed. (' + e.message + ')';
    }
    advisorBtn.disabled = false;
    advisorBtn.innerHTML = '<i class="ph-fill ph-robot"></i> Ask AI Advisor';
}

// ═══════════════════════════════════════════════════════════════
// Realistic Cinematic Background Engine
// ═══════════════════════════════════════════════════════════════
class CinematicUniverse {
    constructor() {
        this.stages = [
            "Observable Universe", "Galaxy & Cosmic Web", "Solar System Formation", 
            "Earth Formation", "Primordial Oceans", "Early Multicellular Life", 
            "Land Colonization", "Mammalian Era", "Early Humans", 
            "Discovery of Fire", "Early Civilization", "Modern Technology"
        ];
        this.stageImages = [
            '/assets/stage_0.jpg', '/assets/stage_1.jpg', '/assets/stage_2.jpg',
            '/assets/stage_3.jpg', '/assets/stage_4.jpg', '/assets/stage_5.jpg',
            '/assets/stage_6.jpg', '/assets/stage_7.jpg', '/assets/stage_8.jpg',
            '/assets/stage_9.jpg', '/assets/stage_10.jpg', '/assets/stage_11.jpg'
        ];
        this.stageIdx = 0;
        
        this.layerA = document.getElementById('cb-layer-A');
        this.layerB = document.getElementById('cb-layer-B');
        this.activeLayer = 'A'; // A or B
        
        // Single click handler on document: advance stage on any click outside the app
        document.addEventListener('click', (e) => {
            // Only advance on landing page, not during gameplay
            const mainApp = document.getElementById('mainAppDiv');
            if (!mainApp || !mainApp.classList.contains('app-landing')) return;
            // Don't advance if click was inside the app container or on interactive elements
            if (e.target.closest('.app, button, a, select, input, .toast, .modal')) return;
            this.stageIdx = (this.stageIdx + 1) % this.stages.length;
            this.updateStage();
        });
        
        if (this.layerA && this.layerB) {
            this.updateStage(true);
        }
        window._cinematicInstance = this;
    }
    
    updateStage(isInitial = false) {
        if (!this.layerA || !this.layerB) return;
        // Debounce: prevent rapid clicks from skipping stages
        if (this._transitioning && !isInitial) return;
        this._transitioning = true;
        setTimeout(() => { this._transitioning = false; }, 800);
        
        const url = this.stageImages[this.stageIdx];
        const nextLayer = this.activeLayer === 'A' ? this.layerB : this.layerA;
        const currentLayer = this.activeLayer === 'A' ? this.layerA : this.layerB;
        
        // Prepare next layer: reset animation, set image
        nextLayer.style.transition = 'none';
        nextLayer.style.animation = 'none';
        nextLayer.classList.remove('active');
        nextLayer.style.backgroundImage = `url('${url}')`;
        
        // Force reflow to reset animation
        void nextLayer.offsetWidth;
        
        // Re-enable CSS animation and trigger fade-in
        nextLayer.style.transition = 'opacity 2s ease-in-out';
        nextLayer.style.animation = '';
        nextLayer.classList.add('active');
        
        // Fade out old layer after new one appears
        if (!isInitial) {
            setTimeout(() => {
                currentLayer.classList.remove('active');
                currentLayer.style.animation = 'none';
            }, 1800);
        }
        
        this.activeLayer = this.activeLayer === 'A' ? 'B' : 'A';
        
        const lbl = document.getElementById('cinematic-label-text');
        if(lbl) lbl.textContent = this.stages[this.stageIdx];
        
        console.log('[Cinema] Stage ' + this.stageIdx + ': ' + this.stages[this.stageIdx] + ' => ' + url);
    }
    
    jumpToLast() {
        this.stageIdx = this.stages.length - 1;
        this.updateStage();
    }
}

// ═══════════════════════════════════════════════════════════════
// Elegant Cursor Particles
// ═══════════════════════════════════════════════════════════════
class CursorParticles {
    constructor() {
        this.canvas = document.getElementById('particleCanvas');
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.mouseX = window.innerWidth / 2;
        this.mouseY = window.innerHeight / 2;
        this.targetX = window.innerWidth / 2;
        this.targetY = window.innerHeight / 2;
        
        this.colors = ['#f59e0b', '#fcd34d', '#38bdf8', '#818cf8', '#ffffff'];
        this.maxParticles = 60;
        
        window.addEventListener('resize', this.resize.bind(this));
        document.addEventListener('mousemove', (e) => {
            this.targetX = e.clientX;
            this.targetY = e.clientY;
        });
        
        this.resize();
        this.animate();
    }
    
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    
    animate() {
        if (this.canvas.classList.contains('hidden-particles')) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            requestAnimationFrame(() => this.animate());
            return;
        }
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        let dx = this.targetX - this.mouseX;
        let dy = this.targetY - this.mouseY;
        this.mouseX += dx * 0.08;
        this.mouseY += dy * 0.08;
        
        let speed = Math.sqrt(dx*dx + dy*dy);
        
        if (this.particles.length < this.maxParticles && Math.random() < 0.2 + (speed * 0.005)) {
            this.particles.push({
                x: this.mouseX + (Math.random() - 0.5) * 30,
                y: this.mouseY + (Math.random() - 0.5) * 30,
                vx: (Math.random() - 0.5) * 0.6,
                vy: (Math.random() - 0.5) * 0.6 - 0.3, 
                life: 0,
                maxLife: 80 + Math.random() * 60,
                size: 1 + Math.random() * 2,
                color: this.colors[Math.floor(Math.random() * this.colors.length)]
            });
        }
        
        for (let i = this.particles.length - 1; i >= 0; i--) {
            let p = this.particles[i];
            p.x += p.vx;
            p.y += p.vy;
            p.life++;
            
            let alpha = 1;
            if (p.life < 20) alpha = p.life / 20;
            else if (p.life > p.maxLife - 20) alpha = (p.maxLife - p.life) / 20;
            
            if (p.life >= p.maxLife) {
                this.particles.splice(i, 1);
                continue;
            }
            
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fillStyle = p.color;
            this.ctx.globalAlpha = alpha * 0.6;
            this.ctx.shadowBlur = 10;
            this.ctx.shadowColor = p.color;
            this.ctx.fill();
        }
        
        this.ctx.globalAlpha = 1;
        this.ctx.shadowBlur = 0;
        
        requestAnimationFrame(() => this.animate());
    }
}

// Use DOMContentLoaded instead of window.onload so buttons work before 97MB videos load
document.addEventListener('DOMContentLoaded', function() {
    try { new CinematicUniverse(); } catch(e) { console.log('CinematicUniverse skipped:', e); }
    try { new CursorParticles(); } catch(e) { console.log('CursorParticles skipped:', e); }
    
    // Animations removed — show dashboard immediately
    isSkippedAll = true;
    
    // Force dashboard visible directly (no dependency on any other element)
    const mainApp = document.getElementById('mainAppDiv');
    if (mainApp) {
        mainApp.style.display = 'block';
        mainApp.style.opacity = '1';
        mainApp.style.position = 'relative';
        mainApp.style.zIndex = '2';
        document.body.style.overflow = 'auto';
    }
});

// Helper: reveal the main dashboard by clearing ALL cinematic overlays
function revealMainDashboard() {

    // Hide cinematic background clicker
    const clicker = document.getElementById('cinematic-clicker');
    if (clicker) clicker.style.display = 'none';
    
    // Hide cinematic label
    const label = document.getElementById('cinematic-label');
    if (label) label.classList.add('hidden-cinema');
    
    // Jump cinematic to final stage safely
    try {
        if (window._cinematicInstance) window._cinematicInstance.jumpToLast();
    } catch(e) { console.error(e); }
    
    // Bring main app above any remaining background layers
    const mainApp = document.getElementById('mainAppDiv');
    if (mainApp) {
        document.body.style.overflow = 'auto'; // CRITICAL: enable scrolling
        mainApp.style.display = 'block';
        mainApp.style.position = 'relative';
        mainApp.style.zIndex = '2';
        mainApp.style.opacity = '1';
        mainApp.classList.add('visible');
    }
    
    // Hide skip-all button
    const skipAllBtn = document.getElementById('skip-all-btn');
    if (skipAllBtn) {
        skipAllBtn.style.opacity = '0';
        skipAllBtn.style.pointerEvents = 'none';
        setTimeout(() => skipAllBtn.style.display = 'none', 500);
    }
}

let isSkippedAll = false;

function skipAllIntros() {

  isSkippedAll = true;
  
  // Pause all videos immediately
  const introVid = document.getElementById('intro-vid');
  const evoVid = document.getElementById('evo-vid');
  const introVideo = document.getElementById('introVideo');
  if (introVid) introVid.pause();
  if (evoVid) evoVid.pause();
  if (introVideo) introVideo.pause();
  
  // Hide intro video overlay
  const introVideoOverlay = document.getElementById('introVideoOverlay');
  if (introVideoOverlay) { introVideoOverlay.style.display = 'none'; }
  
  // Gather all overlay elements
  const s1 = document.getElementById('intro-screen');
  const s2 = document.getElementById('bulb-screen');
  const s3 = document.getElementById('evo-screen');
  const skipBtn = document.getElementById('skip-btn');
  const bulbSkipBtn = document.getElementById('bulb-skip-btn');
  const evoSkip = document.getElementById('evo-skip-btn');
  
  // Immediately hide all skip buttons
  if (skipBtn) skipBtn.style.display = 'none';
  if (bulbSkipBtn) bulbSkipBtn.style.display = 'none';
  if (evoSkip) evoSkip.style.display = 'none';
  
  // Fade out all intro screens and disable pointer events
  [s1, s2, s3].forEach(el => {
      if (el) {
          el.style.opacity = '0';
          el.style.pointerEvents = 'none';
      }
  });
  
  // Reveal the main dashboard (hides cinematic clicker, brings mainApp forward)
  revealMainDashboard();
  
  // After fade completes, remove all overlay screens from DOM layout
  setTimeout(() => {
      [s1, s2, s3].forEach(el => {
          if (el) el.style.display = 'none';
      });
  }, 600);
}

function skipIntro() {
  if (isSkippedAll) return;

  const screen = document.getElementById('intro-screen');
  const vid = document.getElementById('intro-vid');
  
  if (vid) vid.pause();
  const skipBtn = document.getElementById('skip-btn');
  if (skipBtn) skipBtn.style.display = 'none';
  
  // Show bulb screen (stage 2) NOW so it's behind the fading intro
  const bulbScreen = document.getElementById('bulb-screen');
  if (bulbScreen) {
      bulbScreen.classList.add('active');
      createAmbientDust();
  } else {
      showEvoVideo();
  }
  
  if (screen) {
      screen.style.opacity = '0';
      screen.style.pointerEvents = 'none';
      screen.style.transform = 'scale(1.03)';
      setTimeout(() => {
          if (isSkippedAll) return;
          screen.style.display = 'none';
      }, 2000);
  }
}

function skipBulb() {
  if (isSkippedAll) return;
  bulbIgnited = true; // Prevent pending actions
  const btn = document.getElementById('bulb-skip-btn');
  if (btn) btn.style.display = 'none';
  
  const bulbScreen = document.getElementById('bulb-screen');
  if (bulbScreen) {
      bulbScreen.style.opacity = '0';
      bulbScreen.style.pointerEvents = 'none';
  }
  
  showEvoVideo();
}

// ═══════════════════════════════════════════════════════════════
// Bulb Bloom Animation (Enhanced)
// ═══════════════════════════════════════════════════════════════
let bulbIgnited = false;

// Create ambient dust on bulb screen activation
function createAmbientDust() {
  const container = document.getElementById('ambientDust');
  for (let i = 0; i < 25; i++) {
    const d = document.createElement('div');
    d.classList.add('ambient-dust');
    const s = 1 + Math.random() * 3;
    const x = Math.random() * 100;
    const y = Math.random() * 100;
    d.style.cssText = 'left:'+x+'%;top:'+y+'%;width:'+s+'px;height:'+s+'px;--mx:'+(10+Math.random()*30)+'px;--my:'+(10+Math.random()*20)+'px;--dur:'+(4+Math.random()*6)+'s;animation-delay:'+(Math.random()*4)+'s;';
    container.appendChild(d);
  }
}

function igniteBulb() {
  if (bulbIgnited) return;
  bulbIgnited = true;
  
  const bulb = document.getElementById('theBulb');
  const label = document.getElementById('bulbLabel');
  const burst = document.getElementById('lightBurst');
  const quote = document.getElementById('bloomQuote');
  
  // Step 1: Glow the bulb + light burst (0-1s)
  bulb.classList.add('glowing');
  label.style.opacity = '0';
  setTimeout(() => burst.classList.add('expand'), 300);
  
  // Step 2: Grow vines (1-3.5s)
  setTimeout(() => growVines(), 800);
  
  // Step 3: Show cinematic quote (1.5s)
  setTimeout(() => quote.classList.add('visible'), 1500);
  
  // Step 4: Bloom flowers (2.5-4.5s)
  setTimeout(() => bloomFlowers(), 2200);
  
  // Step 5: Particles (3-6s)
  setTimeout(() => spawnBloomParticles(), 3000);
  
  // Step 6: Butterflies (3.5s)
  setTimeout(() => spawnButterflies(), 3500);
  
  // Step 7: Fade to evolution video (7s)
  setTimeout(() => {

      const bulbScreen = document.getElementById('bulb-screen');
      if (bulbScreen) {
          bulbScreen.style.opacity = '0';
          bulbScreen.style.pointerEvents = 'none';
          setTimeout(() => { bulbScreen.style.display = 'none'; }, 2500);
      }
      showEvoVideo();
  }, 7000);
}

function growVines() {
  const container = document.getElementById('vineContainer');
  const vineCount = 12;
  const screenW = window.innerWidth;
  const screenH = window.innerHeight;
  const maxReach = Math.max(screenW, screenH) * 0.55;
  const grads = ['url(#vineGrad)', 'url(#vineGrad2)'];
  
  for (let i = 0; i < vineCount; i++) {
    const angle = (i / vineCount) * 360 + (Math.random() - 0.5) * 20;
    const length = maxReach * (0.5 + Math.random() * 0.5);
    const curve = 40 + Math.random() * 80;
    
    const rad = angle * Math.PI / 180;
    const ex = Math.cos(rad) * length;
    const ey = Math.sin(rad) * length;
    // Two control points for S-curve (cubic bezier)
    const c1x = Math.cos(rad) * (length * 0.3) + (Math.random() - 0.5) * curve;
    const c1y = Math.sin(rad) * (length * 0.3) + (Math.random() - 0.5) * curve;
    const c2x = Math.cos(rad) * (length * 0.7) + (Math.random() - 0.5) * curve;
    const c2y = Math.sin(rad) * (length * 0.7) + (Math.random() - 0.5) * curve;
    
    const svgSize = Math.ceil(length * 2.5);
    const cx = svgSize / 2;
    const cy = svgSize / 2;
    
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', String(svgSize));
    svg.setAttribute('height', String(svgSize));
    svg.style.cssText = 'position:absolute;top:-'+(svgSize/2)+'px;left:-'+(svgSize/2)+'px;overflow:visible;';
    
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M'+cx+','+cy+' C'+(cx+c1x)+','+(cy+c1y)+' '+(cx+c2x)+','+(cy+c2y)+' '+(cx+ex)+','+(cy+ey));
    path.classList.add('vine');
    path.style.stroke = grads[i % 2];
    path.style.strokeWidth = (1.5 + Math.random() * 2) + '';
    
    svg.appendChild(path);
    container.appendChild(svg);
    
    setTimeout(() => path.classList.add('grow'), 50 + i * 120);
    
    // Sub-branch from each main vine
    if (Math.random() > 0.3) {
      const branchAngle = angle + (Math.random() > 0.5 ? 35 : -35) + Math.random() * 20;
      const branchLen = length * 0.3 + Math.random() * length * 0.2;
      const brad = branchAngle * Math.PI / 180;
      const bStartX = cx + Math.cos(rad) * length * 0.6;
      const bStartY = cy + Math.sin(rad) * length * 0.6;
      const bEndX = bStartX + Math.cos(brad) * branchLen;
      const bEndY = bStartY + Math.sin(brad) * branchLen;
      const bcx = (bStartX + bEndX) / 2 + (Math.random() - 0.5) * 30;
      const bcy = (bStartY + bEndY) / 2 + (Math.random() - 0.5) * 30;
      
      const bpath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      bpath.setAttribute('d', 'M'+bStartX+','+bStartY+' Q'+bcx+','+bcy+' '+bEndX+','+bEndY);
      bpath.classList.add('vine');
      bpath.style.stroke = grads[(i+1)%2];
      bpath.style.strokeWidth = (1 + Math.random()) + '';
      bpath.style.opacity = '0.7';
      svg.appendChild(bpath);
      
      setTimeout(() => bpath.classList.add('grow'), 800 + i * 120);
    }
  }
}

function bloomFlowers() {
  const container = document.getElementById('vineContainer');
  const flowerCount = 22;
  const colors = [
    ['#fde68a','#fbbf24'],['#fbcfe8','#f472b6'],['#c4b5fd','#a78bfa'],
    ['#bae6fd','#38bdf8'],['#fff','#e2e8f0'],['#d9f99d','#84cc16'],
    ['#fecaca','#f87171']
  ];
  const screenMax = Math.max(window.innerWidth, window.innerHeight) * 0.4;
  
  for (let i = 0; i < flowerCount; i++) {
    const angle = Math.random() * 360;
    const dist = 40 + Math.random() * screenMax;
    const rad = angle * Math.PI / 180;
    const x = Math.cos(rad) * dist;
    const y = Math.sin(rad) * dist;
    const size = 10 + Math.random() * 16;
    const c = colors[Math.floor(Math.random() * colors.length)];
    const rot = -20 + Math.random() * 40;
    
    const flower = document.createElement('div');
    flower.classList.add('bloom-flower');
    flower.style.cssText = 'left:calc(50% + '+x+'px);top:calc(50% + '+y+'px);width:'+size+'px;height:'+size+'px;--rot:'+rot+'deg;';
    
    // Multi-petal flower
    const petalCount = 4 + Math.floor(Math.random() * 3);
    for (let j = 0; j < petalCount; j++) {
      const petal = document.createElement('div');
      const pAngle = (j / petalCount) * 360;
      petal.style.cssText = 'position:absolute;top:50%;left:50%;width:'+size*0.7+'px;height:'+size*0.35+'px;border-radius:50%;background:radial-gradient(circle at 30% 50%,'+c[0]+','+c[1]+');transform-origin:0 50%;transform:rotate('+pAngle+'deg);box-shadow:0 0 8px '+c[1]+';opacity:0.9;';
      flower.appendChild(petal);
    }
    // Center dot
    const center = document.createElement('div');
    center.style.cssText = 'position:absolute;top:50%;left:50%;width:'+size*0.25+'px;height:'+size*0.25+'px;border-radius:50%;background:#fef3c7;transform:translate(-50%,-50%);box-shadow:0 0 6px #fbbf24;z-index:1;';
    flower.appendChild(center);
    
    container.appendChild(flower);
    setTimeout(() => flower.classList.add('open'), 80 + i * 100);
  }
}

function spawnBloomParticles() {
  const container = document.getElementById('bloomParticles');
  const particleCount = 80;
  const pColors = ['#fde68a','#4ade80','#f472b6','#38bdf8','#a78bfa','#ffffff','#fbbf24','#86efac','#c4b5fd'];
  
  for (let i = 0; i < particleCount; i++) {
    const p = document.createElement('div');
    p.classList.add('bloom-particle');
    const size = 1.5 + Math.random() * 6;
    const angle = Math.random() * 360;
    const dist = 80 + Math.random() * 500;
    const rad = angle * Math.PI / 180;
    const dx = Math.cos(rad) * dist;
    const dy = Math.sin(rad) * dist - 60;
    const color = pColors[Math.floor(Math.random() * pColors.length)];
    const dur = 2.5 + Math.random() * 2.5;
    
    p.style.cssText = 'left:50%;top:50%;width:'+size+'px;height:'+size+'px;background:'+color+';box-shadow:0 0 '+Math.ceil(size*2)+'px '+color+';--dx:'+dx+'px;--dy:'+dy+'px;--pdur:'+dur+'s;';
    container.appendChild(p);
    
    setTimeout(() => p.classList.add('drift'), 30 + i * 25);
  }
}

function spawnButterflies() {
  const container = document.getElementById('butterflyContainer');
  if (!container) return;
  const wingStyles = ['w1','w2','w3','w4','w5'];
  const count = 8;
  const screenW = window.innerWidth;
  const screenH = window.innerHeight;
  
  for (let i = 0; i < count; i++) {
    const butterfly = document.createElement('div');
    butterfly.classList.add('butterfly');
    
    // Position around the center with organic spread
    const angle = Math.random() * 360;
    const dist = 80 + Math.random() * Math.min(screenW, screenH) * 0.35;
    const rad = angle * Math.PI / 180;
    const x = screenW / 2 + Math.cos(rad) * dist;
    const y = screenH / 2 + Math.sin(rad) * dist;
    
    // Random floating path
    const bx1 = (-30 + Math.random() * 60) + 'px';
    const by1 = (-40 + Math.random() * 30) + 'px';
    const bx2 = (-20 + Math.random() * 40) + 'px';
    const by2 = (-50 + Math.random() * 40) + 'px';
    const bx3 = (-25 + Math.random() * 50) + 'px';
    const by3 = (-30 + Math.random() * 60) + 'px';
    const dur = (6 + Math.random() * 6) + 's';
    const scale = 0.7 + Math.random() * 0.8;
    
    butterfly.style.cssText = 'left:'+x+'px;top:'+y+'px;transform:scale('+scale+');--bx1:'+bx1+';--by1:'+by1+';--bx2:'+bx2+';--by2:'+by2+';--bx3:'+bx3+';--by3:'+by3+';--bdur:'+dur+';';
    
    const wClass = wingStyles[Math.floor(Math.random() * wingStyles.length)];
    
    // Body
    const body = document.createElement('div');
    body.classList.add('butterfly-body');
    
    // Left wing
    const leftWing = document.createElement('div');
    leftWing.classList.add('butterfly-wing', 'left', wClass);
    leftWing.style.animationDelay = (Math.random() * 0.2) + 's';
    const dotL = document.createElement('div');
    dotL.classList.add('wing-dot');
    leftWing.appendChild(dotL);
    
    // Right wing
    const rightWing = document.createElement('div');
    rightWing.classList.add('butterfly-wing', 'right', wClass);
    rightWing.style.animationDelay = (Math.random() * 0.2) + 's';
    const dotR = document.createElement('div');
    dotR.classList.add('wing-dot');
    rightWing.appendChild(dotR);
    
    body.appendChild(leftWing);
    body.appendChild(rightWing);
    butterfly.appendChild(body);
    container.appendChild(butterfly);
    
    // Stagger appearance
    setTimeout(() => butterfly.classList.add('visible'), 200 + i * 350);
  }
}

let evoStarted = false;
function showEvoVideo() {

  if (isSkippedAll || evoStarted) return;
  evoStarted = true;
  
  const bulbScreen = document.getElementById('bulb-screen');
  const evoScreen = document.getElementById('evo-screen');
  const evoVid = document.getElementById('evo-vid');
  
  if (bulbScreen) {
      setTimeout(() => {
          bulbScreen.style.display = 'none';
      }, 2500);
  }
  
  if (evoScreen && evoVid) {
      evoScreen.style.display = 'flex';
      // Force reflow
      void evoScreen.offsetWidth;
      evoScreen.style.opacity = '1';
      evoVid.play().catch(e => console.log('Evo video autoplay prevented.', e));
      evoVid.addEventListener('ended', fadeToMainPage);
  }
}

function skipEvoVideo() {
  if (isSkippedAll) return;
  const evoVid = document.getElementById('evo-vid');
  if (evoVid) evoVid.pause();
  fadeToMainPage();
}

function fadeToMainPage() {
  if (isSkippedAll) return;
  
  // Fade out evo-screen (or bulb-screen if evo doesn't exist)
  const evoScreen = document.getElementById('evo-screen');
  if (evoScreen) {
      evoScreen.style.opacity = '0';
      evoScreen.style.pointerEvents = 'none';
      setTimeout(() => evoScreen.style.display = 'none', 2000);
  } else {
      const bulbScreen = document.getElementById('bulb-screen');
      if (bulbScreen) {
          bulbScreen.style.opacity = '0';
          bulbScreen.style.pointerEvents = 'none';
          setTimeout(() => bulbScreen.style.display = 'none', 2000);
      }
  }
  
  // Also hide intro-screen if it's somehow still there
  const introScreen = document.getElementById('intro-screen');
  if (introScreen) {
      introScreen.style.opacity = '0';
      introScreen.style.pointerEvents = 'none';
      introScreen.style.display = 'none';
  }
  
  // CRITICAL: Reveal the main dashboard (hides cinematic clicker, brings mainApp forward)
  revealMainDashboard();
}

// ═══════════════════════════════════════════════════════════════
// POLISH: AI Auto-Play System
// ═══════════════════════════════════════════════════════════════
let aiPlaying = false;
let aiInterval = null;

async function toggleAIPlay() {
  if (aiPlaying) {
    aiPlaying = false;
    clearInterval(aiInterval);
    if(document.getElementById('demoPlayBtn')) {
      document.getElementById('demoPlayBtn').innerHTML='<i class="ph-bold ph-lightning"></i> AI Auto-Play';
      document.getElementById('demoPlayBtn').classList.remove('playing');
    }
    document.getElementById('aiThinkingBox').classList.add('hidden');
    document.getElementById('modeBadge').innerHTML='<i class="ph-fill ph-user"></i> Mode: Human Decision';
    document.getElementById('modeBadge').style.color='#94a3b8';
    return;
  }
  if (!sid) { resetGame(); await new Promise(r => setTimeout(r, 1500)); }
  if (!sid) return;
  aiPlaying = true;
  if(document.getElementById('demoPlayBtn')) {
      document.getElementById('demoPlayBtn').innerHTML='<i class="ph-bold ph-lightning"></i> Stop AI';
      document.getElementById('demoPlayBtn').classList.add('playing');
  }
  document.getElementById('modeBadge').innerHTML='<i class="ph-fill ph-brain"></i> Mode: AI Simulation';
  document.getElementById('modeBadge').style.color='#a855f7';

  aiInterval = setInterval(async () => {
    if (!aiPlaying || !sid) { clearInterval(aiInterval); return; }
    try {
      const msgs = document.querySelectorAll('.msg-card');
      let action = 'invest_growth';
      if (msgs.length > 0) {
        const first = msgs[0];
        const src = (first.querySelector('.msg-source')?.textContent || '').toLowerCase();
        const subj = (first.querySelector('.msg-subject')?.textContent || '').toLowerCase();
        if (src.includes('defense') || subj.includes('attack') || subj.includes('raid')) action = 'defend';
        else if (src.includes('citizen') && (subj.includes('food') || subj.includes('hunger') || subj.includes('grain'))) action = 'allocate_food';
        else if (src.includes('scientist') || subj.includes('research') || subj.includes('tech')) action = 'approve_research';
        else if (src.includes('worker') || subj.includes('mine')) action = 'allocate_workers';
        else if (subj.includes('protest') || subj.includes('unrest') || subj.includes('strike')) action = 'calm_citizens';
        else if (src.includes('trader')) action = 'accept_trade';
        else if (subj.includes('plague') || subj.includes('flood') || subj.includes('fire')) action = 'emergency_response';
        const fid = first.getAttribute('data-id');
        if (fid) selMsg = fid;
      }
      
      const tb = document.getElementById('aiThinkingBox');
      tb.classList.remove('hidden');
      tb.innerHTML = '<div class="think-spinner"></div> <span style="font-style:italic">AI evaluating situation...</span>';
      
      // Phase 2: Show enhanced CoT panel
      const cotPanel = document.getElementById('aiCotPanel');
      cotPanel.classList.add('active');
      document.getElementById('aiCotThought').textContent = '🧠 Analyzing state: Pop=' + (document.getElementById('vPop')?.textContent||'?') + ', Food=' + (document.getElementById('vFood')?.textContent||'?') + ', Danger=' + (document.getElementById('vDanger')?.textContent||'?');
      document.getElementById('aiCotAction').innerHTML = '<i class="ph-bold ph-spinner ph-spin"></i> Computing optimal action...';
      document.getElementById('aiCotConfidence').textContent = '';
      document.getElementById('aiCotJudge').textContent = '';
      
      // Use LLM if available
      const currentState = document.getElementById('vPop') ? {
        population: document.getElementById('vPop').textContent,
        food: document.getElementById('vFood').textContent,
        energy: document.getElementById('vEnergy').textContent,
        morale: document.getElementById('vMorale').textContent,
        danger: document.getElementById('vDanger').textContent
      } : {};
      let llmChoice = await determineLLMAction(currentState, msgs.length>0 ? [{id: selMsg, text: msgs[0].textContent}] : []);
      
      let finalAction = action;
      let reasoning = '';
      if (llmChoice) {
        // Parse LLM choice
        const parts = llmChoice.split('|');
        if (parts.length > 0 && parts[0].trim()) {
           finalAction = parts[0].trim().toLowerCase();
           if (parts.length > 1) {
              const conf = parts[1].trim();
              document.getElementById('aiConfidenceLabel').textContent = "Confidence: " + conf;
              document.getElementById('aiCotConfidence').textContent = '🎯 Confidence: ' + conf;
           }
           if (parts.length > 2) {
              reasoning = parts[2].trim();
              document.getElementById('aiStrategyLabel').textContent = "Strategy: " + reasoning;
              document.getElementById('aiReasoningLine').textContent = "🧠 " + reasoning;
              document.getElementById('aiReasoningPanel').classList.add('active');
              document.getElementById('aiCotThought').textContent = '🧠 ' + reasoning;
           }
           // Phase 2: Update CoT action display
           const actLabel = finalAction.split('_').map(w=>w.charAt(0).toUpperCase()+w.slice(1)).join(' ');
           document.getElementById('aiCotAction').innerHTML = '<i class="ph-fill ph-lightning" style="color:#a78bfa"></i> Decision: <strong>' + actLabel + '</strong>';
        }
      }

      // Phase 2: Fire background LLM Judge evaluation (non-blocking)
      const crisisText = msgs.length > 0 ? msgs[0].textContent : '';
      queryJudge(finalAction, {...currentState, era: document.getElementById('vEra')?.textContent || 'tribal', reasoning: reasoning}, crisisText)
        .then(j => {
          if (j && j.grade) {
            const judgeEl = document.getElementById('aiCotJudge');
            if (judgeEl) {
              judgeEl.className = 'ai-cot-judge judge-badge judge-' + j.grade;
              judgeEl.innerHTML = '⚖️ Judge: ' + j.grade + ' (' + (j.score >= 0 ? '+' : '') + j.score.toFixed(1) + ') — ' + (j.verdict || '');
            }
          }
        }).catch(() => {});
      
      if (!aiPlaying || !sid) return;
      
      const actFormat = finalAction.split('_').map(w=>w.charAt(0).toUpperCase()+w.slice(1)).join(' ');
      tb.innerHTML = '<i class="ph-fill ph-lightning"></i> AI Decision: <strong>'+actFormat+'</strong>';
      
      await new Promise(r => setTimeout(r, 600));
      if (!aiPlaying || !sid) return;
      
      await act(finalAction);
    } catch(e) { console.log('AI step error:', e); }
  }, 3500);
}

// ═══════════════════════════════════════════════════════════════
// POLISH: Agent Benchmark System
// ═══════════════════════════════════════════════════════════════
async function runBenchmark() {
  const bp = document.getElementById('benchPanel');
  bp.classList.add('visible');
  document.getElementById('benchStatus').textContent = 'Running benchmark...';
  const agents = ['random','greedy','logical'];
  for (const agent of agents) {
    try {
      const r = await fetch('/simulate/batch', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({task_id:'task_demo_5', agent_type:agent, episodes:5, seed:42})
      });
      const data = await r.json();
      const rate = data.aggregate.survival_rate;
      const parts = rate.split('/');
      const pct = Math.round((parseInt(parts[0])/parseInt(parts[1]))*100);
      const idCap = agent.charAt(0).toUpperCase() + agent.slice(1);
      document.getElementById('bench'+idCap).textContent = pct + '%';
      document.getElementById('bench'+idCap+'Bar').style.width = pct + '%';
    } catch(e) { console.log('Benchmark error for '+agent+':', e); }
  }
  document.getElementById('benchStatus').textContent = 'Environment successfully differentiates agent intelligence.';
}

// ═══════════════════════════════════════════════════════════════
// POLISH: Dismiss End Overlay
// ═══════════════════════════════════════════════════════════════
function dismissEnd() {
  const ov = document.getElementById('endOverlay');
  ov.classList.remove('active');
  setTimeout(() => { ov.className = 'end-overlay'; ov.innerHTML = ''; }, 1500);
}

// ═══════════════════════════════════════════════════════════════
// POLISH: Sound Effects (Web Audio API — lightweight)
// ═══════════════════════════════════════════════════════════════
let audioCtx = null;
document.addEventListener('keydown', (e) => {
  if (['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
  const k = e.key.toLowerCase();
  const actionMap = {a:'allocate_food', w:'allocate_workers', r:'approve_research', d:'defend', c:'calm_citizens', g:'invest_growth'};
  if (actionMap[k]) {
    act(actionMap[k]);
    setTimeout(() => document.querySelectorAll('.cooldown').forEach(el=>el.classList.remove('cooldown')), 600);
  }
});


function getAudioCtx() {
  if (!audioCtx) { try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch(e) {} }
  return audioCtx;
}
function playTick(positive) {
  const ctx = getAudioCtx();
  if (!ctx || !isAudioEnabled) return;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.connect(gain).connect(ctx.destination);
  osc.frequency.value = positive ? 880 : 330;
  osc.type = 'sine';
  gain.gain.value = 0.04;
  gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);
  osc.start(ctx.currentTime);
  osc.stop(ctx.currentTime + 0.15);
}
function playEndSound(isCollapse) {
  const ctx = getAudioCtx();
  if (!ctx || !isAudioEnabled) return;
  const freqs = isCollapse ? [220, 185, 147] : [523, 659, 784];
  freqs.forEach((f, i) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain).connect(ctx.destination);
    osc.frequency.value = f;
    osc.type = isCollapse ? 'sawtooth' : 'sine';
    gain.gain.value = 0.06;
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3 + i*0.2);
    osc.start(ctx.currentTime + i * 0.15);
    osc.stop(ctx.currentTime + 0.4 + i*0.2);
  });
}
// ═══════════════════════════════════════════════════════════════
// PHASE 2: AI Council Query System
// ═══════════════════════════════════════════════════════════════
async function queryCouncil() {
  if (!sid) return;
  const btn = document.getElementById('councilBtn');
  const panel = document.getElementById('councilPanel');
  btn.disabled = true;
  btn.innerHTML = '<i class="ph-bold ph-spinner ph-spin"></i> Consulting...';
  panel.classList.add('active');
  panel.innerHTML = '<div class="council-header"><i class="ph-fill ph-users-four"></i> AI COUNCIL DELIBERATING...</div><div style="text-align:center;padding:1rem;color:#64748b"><div class="think-spinner" style="display:inline-block"></div> Querying multiple AI models...</div>';
  
  const pop = parseInt(document.getElementById('vPop').textContent) || 100;
  const food = parseInt(document.getElementById('vFood').textContent) || 100;
  const energy = parseInt(document.getElementById('vEnergy').textContent) || 100;
  const morale = parseInt(document.getElementById('vMorale').textContent) || 60;
  const danger = parseInt(document.getElementById('vDanger').textContent) || 0;
  const era = document.getElementById('vEra').textContent || 'TRIBAL ERA';
  let msgs = [];
  document.querySelectorAll('.msg-subject').forEach(el => msgs.push(el.textContent));
  
  try {
    const r = await fetch('/api/ai/council', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({population:pop, food:food, energy:energy, morale:morale, danger:danger, era:era, messages:msgs.join('; ')})
    });
    const d = await r.json();
    let html = '<div class="council-header"><i class="ph-fill ph-users-four"></i> AI COUNCIL — ' + d.members_responded + '/' + d.total_members + ' responded</div>';
    html += '<div class="council-grid">';
    (d.council || []).forEach(m => {
      const iconMap = {strategist:'strategy',economist:'coins',ethicist:'scales',analyst:'chart-line'};
      const icon = iconMap[m.role] || m.icon || 'robot';
      html += '<div class="council-member ' + (m.opinion ? '' : 'unavailable') + '">' +
        '<div class="council-member-header"><i class="ph-fill ph-' + icon + '" style="color:#a78bfa"></i><span class="council-member-role">' + (m.label || m.role) + '</span><span class="council-member-provider">' + (m.provider || '') + '</span></div>' +
        '<div class="council-member-opinion">' + (m.opinion || '<i>Unavailable</i>') + '</div></div>';
    });
    html += '</div>';
    if (d.consensus) {
      html += '<div class="council-consensus"><i class="ph-fill ph-check-circle" style="margin-right:6px"></i><strong>Consensus:</strong> ' + d.consensus + '</div>';
    }
    panel.innerHTML = html;
  } catch(e) {
    panel.innerHTML = '<div style="color:#fca5a5;text-align:center;padding:1rem"><i class="ph-fill ph-warning"></i> Council unavailable: ' + e.message + '</div>';
  }
  btn.disabled = false;
  btn.innerHTML = '<i class="ph-fill ph-users-four"></i> AI Council';
}

// ═══════════════════════════════════════════════════════════════
// PHASE 2: Causality Web — Decision Graph Visualization
// ═══════════════════════════════════════════════════════════════
let causalityNodes = [];
let causalityLinks = [];

function addCausalityNode(turn, action, effective, pending) {
  const node = { turn, action, effective, pending: pending || 0, x: 0, y: 0 };
  causalityNodes.push(node);
  // Create links to previously connected actions
  if (causalityNodes.length > 1) {
    const prev = causalityNodes[causalityNodes.length - 2];
    causalityLinks.push({ from: causalityNodes.length - 2, to: causalityNodes.length - 1, type: effective ? 'good' : 'bad' });
  }
  // Link delayed consequences
  if (pending > 0 && causalityNodes.length > 2) {
    const lookback = Math.min(causalityNodes.length - 1, 3);
    for (let i = causalityNodes.length - 1 - lookback; i < causalityNodes.length - 1; i++) {
      if (i >= 0 && !causalityNodes[i].effective) {
        causalityLinks.push({ from: i, to: causalityNodes.length - 1, type: 'delayed' });
      }
    }
  }
  drawCausalityWeb();
}

function drawCausalityWeb() {
  const canvas = document.getElementById('causalityCanvas');
  if (!canvas || causalityNodes.length < 2) return;
  const webPanel = document.getElementById('causalityWeb');
  if (webPanel) webPanel.classList.add('active');
  
  const ctx = canvas.getContext('2d');
  const w = canvas.parentElement.clientWidth - 32;
  const h = 180;
  canvas.width = w * (window.devicePixelRatio || 1);
  canvas.height = h * (window.devicePixelRatio || 1);
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, w, h);
  
  const padding = 30;
  const usableW = w - padding * 2;
  const usableH = h - padding * 2;
  
  // Layout nodes horizontally
  causalityNodes.forEach((n, i) => {
    n.x = padding + (usableW / Math.max(1, causalityNodes.length - 1)) * i;
    n.y = padding + usableH / 2 + Math.sin(i * 0.8) * (usableH * 0.3);
  });
  
  // Draw links
  causalityLinks.forEach(link => {
    const from = causalityNodes[link.from];
    const to = causalityNodes[link.to];
    if (!from || !to) return;
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    const cpx = (from.x + to.x) / 2;
    const cpy = Math.min(from.y, to.y) - 20;
    ctx.quadraticCurveTo(cpx, cpy, to.x, to.y);
    ctx.strokeStyle = link.type === 'good' ? 'rgba(16,185,129,0.4)' : link.type === 'delayed' ? 'rgba(251,191,36,0.5)' : 'rgba(239,68,68,0.3)';
    ctx.lineWidth = link.type === 'delayed' ? 2 : 1.5;
    if (link.type === 'delayed') ctx.setLineDash([4, 4]);
    else ctx.setLineDash([]);
    ctx.stroke();
    ctx.setLineDash([]);
  });
  
  // Draw nodes
  causalityNodes.forEach((n, i) => {
    ctx.beginPath();
    const radius = n.pending > 0 ? 7 : 5;
    ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = n.effective ? '#10b981' : n.pending > 0 ? '#fbbf24' : '#ef4444';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Label
    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px Outfit, sans-serif';
    ctx.textAlign = 'center';
    const label = 'T' + n.turn;
    ctx.fillText(label, n.x, n.y + radius + 12);
    
    // Action name (short)
    if (i === causalityNodes.length - 1 || i % 2 === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '8px Outfit, sans-serif';
      const shortAct = (n.action || '').replace(/_/g, ' ').substring(0, 12);
      ctx.fillText(shortAct, n.x, n.y - radius - 6);
    }
  });
}

// ═══════════════════════════════════════════════════════════════
// PHASE 2: Behavioral Profiling Radar Chart
// ═══════════════════════════════════════════════════════════════
function drawRadarChart(canvasId, radarData, size) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  canvas.style.width = size + 'px';
  canvas.style.height = size + 'px';
  ctx.scale(dpr, dpr);
  
  const centerX = size / 2, centerY = size / 2;
  const radius = size * 0.38;
  const labels = Object.keys(radarData);
  const values = Object.values(radarData);
  const n = labels.length;
  const angleStep = (Math.PI * 2) / n;
  
  // Draw grid rings
  [0.25, 0.5, 0.75, 1.0].forEach(ring => {
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius * ring;
      const y = centerY + Math.sin(angle) * radius * ring;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = 'rgba(255,255,255,' + (ring === 1 ? 0.15 : 0.06) + ')';
    ctx.lineWidth = 1;
    ctx.stroke();
  });
  
  // Draw axes
  labels.forEach((_, i) => {
    const angle = i * angleStep - Math.PI / 2;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.stroke();
  });
  
  // Draw data polygon
  ctx.beginPath();
  values.forEach((v, i) => {
    const angle = i * angleStep - Math.PI / 2;
    const r = (v / 100) * radius;
    const x = centerX + Math.cos(angle) * r;
    const y = centerY + Math.sin(angle) * r;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = 'rgba(168,85,247,0.15)';
  ctx.fill();
  ctx.strokeStyle = '#a78bfa';
  ctx.lineWidth = 2;
  ctx.stroke();
  
  // Draw data points & labels
  values.forEach((v, i) => {
    const angle = i * angleStep - Math.PI / 2;
    const r = (v / 100) * radius;
    const x = centerX + Math.cos(angle) * r;
    const y = centerY + Math.sin(angle) * r;
    
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#c4b5fd';
    ctx.fill();
    
    // Label
    const lx = centerX + Math.cos(angle) * (radius + 18);
    const ly = centerY + Math.sin(angle) * (radius + 18);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(labels[i], lx, ly);
    
    // Value
    ctx.fillStyle = '#e2e8f0';
    ctx.font = 'bold 9px Outfit, sans-serif';
    ctx.fillText(Math.round(v), x, y - 10);
  });
}

// ═══════════════════════════════════════════════════════════════
// PHASE 2: LLM-as-Judge Integration
// ═══════════════════════════════════════════════════════════════
async function queryJudge(action, state, crisis) {
  try {
    const r = await fetch('/api/ai/judge', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        population: parseInt(state.population) || 100,
        food: parseInt(state.food) || 100,
        energy: parseInt(state.energy) || 100,
        morale: parseInt(state.morale) || 60,
        danger: parseInt(state.danger) || 0,
        era: state.era || 'tribal',
        action: action,
        crisis_description: crisis || '',
        reasoning: state.reasoning || ''
      })
    });
    return await r.json();
  } catch(e) {
    return { score: 0, grade: 'ADEQUATE', verdict: 'Judge unavailable', source: 'error' };
  }
}

// ═══════════════════════════════════════════════════════════════
// PHASE 2: Chronicle Generation
// ═══════════════════════════════════════════════════════════════
async function generateChronicle(summary) {
  try {
    const mem = summary.memory || {};
    const keyMoments = episodeLog.filter(e => e.type === 'step').map(e => 'T' + (e.state?.turn || '?') + ': ' + e.action);
    const r = await fetch('/api/ai/chronicle', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        era_final: summary.era_final || summary.final_era || 'tribal',
        population_final: summary.population_final || 0,
        total_reward: summary.total_reward || 0,
        survived: !summary.collapse,
        collapse_reason: summary.collapse_reason || '',
        crises_averted: summary.crises_averted || 0,
        crises_failed: summary.crises_failed || 0,
        personality: mem.personality || 'balanced',
        stability_score: mem.stability_score || 0.65,
        key_moments: keyMoments.slice(-10),
        turns_played: summary.turns_played || summary.total_turns || 10,
        eras_reached: summary.eras_reached || []
      })
    });
    return await r.json();
  } catch(e) {
    return { chronicle: 'Chronicle generation unavailable.', source: 'error' };
  }
}

// ═══════════════════════════════════════════════════════════════
// PHASE 2: Behavioral Profile Request
// ═══════════════════════════════════════════════════════════════
async function fetchBehavioralProfile(summary) {
  try {
    const mem = summary.memory || {};
    const actions = episodeLog.filter(e => e.type === 'step').map(e => ({action: e.action}));
    const r = await fetch('/api/ai/profile', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        actions: actions,
        personality: mem.personality || 'balanced',
        personality_strength: mem.personality_strength || 0,
        citizen_trust: mem.citizen_trust || 50,
        military_trust: mem.military_trust || 50,
        stability_score: mem.stability_score || 0.65,
        crises_averted: summary.crises_averted || 0,
        crises_failed: summary.crises_failed || 0,
        messages_handled: summary.messages_handled || 0,
        messages_ignored: summary.messages_ignored || 0
      })
    });
    return await r.json();
  } catch(e) {
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════
// PHASE 2: Multi-Run Comparison System (localStorage)
// ═══════════════════════════════════════════════════════════════
function saveRunResult(summary) {
  try {
    const runs = JSON.parse(localStorage.getItem('civilCommandRuns') || '[]');
    runs.push({
      timestamp: Date.now(),
      survived: !summary.collapse,
      era: summary.era_final || summary.final_era || 'tribal',
      score: (summary.total_reward || 0).toFixed(2),
      population: summary.population_final || 0,
      turns: summary.turns_played || summary.total_turns || 0,
      personality: (summary.memory || {}).personality || 'balanced',
      mode: aiPlaying ? 'AI' : 'Human'
    });
    if (runs.length > 20) runs.shift();
    localStorage.setItem('civilCommandRuns', JSON.stringify(runs));
  } catch(e) {}
}

function getMultiRunHTML() {
  try {
    const runs = JSON.parse(localStorage.getItem('civilCommandRuns') || '[]');
    if (runs.length < 2) return '';
    const recent = runs.slice(-8).reverse();
    let html = '<div class="multirun-panel"><div class="multirun-header"><i class="ph-fill ph-chart-bar"></i> MULTI-RUN COMPARISON (' + runs.length + ' total)</div>';
    html += '<table class="multirun-table"><thead><tr><th>#</th><th>Mode</th><th>Result</th><th>Era</th><th>Score</th><th>Pop</th><th>Style</th></tr></thead><tbody>';
    recent.forEach((r, i) => {
      html += '<tr><td>' + (runs.length - i) + '</td><td>' + r.mode + '</td>' +
        '<td class="' + (r.survived ? 'multirun-survived' : 'multirun-collapsed') + '">' + (r.survived ? 'SURVIVED' : 'COLLAPSED') + '</td>' +
        '<td>' + (r.era||'').toUpperCase() + '</td><td>' + r.score + '</td><td>' + r.population + '</td><td>' + (r.personality||'').toUpperCase() + '</td></tr>';
    });
    html += '</tbody></table></div>';
    return html;
  } catch(e) { return ''; }
}

// ═══════════════════════════════════════════════════════════════
// SYSTEM 1: Meta-AI Controller — Frontend Status Badge
// ═══════════════════════════════════════════════════════════════

let metaAIStatus = null;

async function fetchMetaAIStatus() {
  try {
    const r = await fetch('/api/meta/status');
    metaAIStatus = await r.json();
    renderMetaAIBadge();
  } catch(e) { console.log('Meta-AI status fetch failed:', e); }
}

function renderMetaAIBadge() {
  let badge = document.getElementById('metaAIBadge');
  if (!badge) {
    badge = document.createElement('div');
    badge.id = 'metaAIBadge';
    badge.className = 'meta-ai-badge';
    badge.onclick = () => showMetaAIDetails();
    document.body.appendChild(badge);
  }
  if (!metaAIStatus) return;
  const healthy = metaAIStatus.healthy || 0;
  const total = metaAIStatus.available || 0;
  const isDegraded = healthy < 2;
  badge.className = 'meta-ai-badge' + (isDegraded ? ' degraded' : '');
  badge.innerHTML = '<div class="meta-dot"></div>' +
    '<div><div class="meta-label">Meta-AI Controller</div>' +
    '<div class="meta-model">' + healthy + '/' + total + ' models online</div></div>';
}

function showMetaAIDetails() {
  if (!metaAIStatus || !metaAIStatus.fleet) return;
  let html = '<div style="max-height:400px;overflow-y:auto;">';
  html += '<table style="width:100%;font-size:0.72rem;border-collapse:collapse;">';
  html += '<thead><tr style="color:#a855f7;border-bottom:1px solid rgba(168,85,247,0.3);">' +
    '<th style="text-align:left;padding:6px;">Model</th><th>Tier</th><th>Health</th><th>Calls</th><th>Latency</th></tr></thead><tbody>';
  Object.entries(metaAIStatus.fleet).forEach(([id, m]) => {
    const statusColor = !m.available ? '#4b5563' : m.healthy ? '#10b981' : '#ef4444';
    const statusText = !m.available ? 'OFFLINE' : m.healthy ? 'HEALTHY' : 'FAILING';
    html += '<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">' +
      '<td style="padding:5px;color:#e2e8f0;">' + (m.model || id) + '</td>' +
      '<td style="text-align:center;color:#818cf8;">' + (m.tier || '-') + '</td>' +
      '<td style="text-align:center;"><span style="color:' + statusColor + ';font-weight:700;">' + statusText + '</span></td>' +
      '<td style="text-align:center;color:#94a3b8;">' + (m.calls || 0) + '</td>' +
      '<td style="text-align:center;color:#94a3b8;">' + (m.avg_latency_ms ? m.avg_latency_ms + 'ms' : '-') + '</td></tr>';
  });
  html += '</tbody></table></div>';
  
  // Show in a modal
  let modal = document.getElementById('metaAIModal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'metaAIModal';
    modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(5,5,15,0.9);z-index:10002;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(6px);';
    modal.onclick = (e) => { if(e.target===modal) modal.style.display='none'; };
    document.body.appendChild(modal);
  }
  modal.innerHTML = '<div style="background:linear-gradient(135deg,rgba(20,20,40,0.98),rgba(10,10,25,0.98));border:1px solid rgba(168,85,247,0.3);border-radius:16px;padding:28px;max-width:700px;width:95%;">' +
    '<div style="font-size:1.2rem;font-weight:800;background:linear-gradient(135deg,#c084fc,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:16px;"><i class="ph-fill ph-cpu"></i> Meta-AI Controller Fleet Status</div>' +
    html +
    '<div style="margin-top:16px;font-size:0.7rem;color:#6b7280;">Controller: ' + (metaAIStatus.controller || 'v1.0') + ' | Status: ' + (metaAIStatus.status || 'unknown') + '</div>' +
    '<button onclick="closeMetaAIModal()" style="margin-top:16px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.15);color:#e2e8f0;padding:8px 20px;border-radius:8px;cursor:pointer;">Close</button>' +
  '</div>';
  modal.style.display = 'flex';
}

// ═══════════════════════════════════════════════════════════════
// SYSTEM 2: Parallel Civilization Simulation — Frontend
// ═══════════════════════════════════════════════════════════════

async function runParallelSim() {
  // Show overlay
  let overlay = document.getElementById('parallelOverlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'parallelOverlay';
    overlay.className = 'parallel-overlay';
    document.body.appendChild(overlay);
  }
  overlay.classList.add('active');
  overlay.innerHTML = '<div class="parallel-content"><div class="parallel-loading"><div class="think-spinner" style="margin:0 auto 16px;"></div>Running 3 civilizations in parallel...<br><span style="font-size:0.75rem;color:#6b7280;">Logical vs Greedy vs Random agents competing on same seed</span></div></div>';
  
  try {
    const taskSel = document.getElementById('taskSelect');
    const taskId = taskSel ? taskSel.value : 'task_demo_10';
    const r = await fetch('/api/simulate/parallel', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task_id: taskId, agents: ['logical', 'greedy', 'random']})
    });
    const d = await r.json();
    renderParallelResults(d);
  } catch(e) {
    overlay.innerHTML = '<div class="parallel-content"><div style="color:#ef4444;text-align:center;">Parallel simulation failed: ' + e.message + '</div>' +
      '<button class="parallel-close" onclick="closeParallelOverlay()">Close</button></div>';
  }
}

function renderParallelResults(data) {
  const overlay = document.getElementById('parallelOverlay');
  if (!overlay) return;
  
  let html = '<div class="parallel-content">';
  html += '<div class="parallel-title"><i class="ph-fill ph-trophy"></i> AI Civilization Race — Results</div>';
  html += '<div style="font-size:0.75rem;color:#6b7280;margin-bottom:16px;">Seed: ' + data.seed + ' | Task: ' + data.task_id + ' | Winner: <span style="color:#6ee7b7;font-weight:700;">' + (data.winner || 'N/A').toUpperCase() + '</span></div>';
  
  html += '<div class="parallel-grid">';
  if (data.ranking) {
    data.ranking.forEach((r, i) => {
      const d2 = data.results[r.agent] || {};
      const isWinner = i === 0;
      const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : '🥉';
      html += '<div class="parallel-card' + (isWinner ? ' winner' : '') + '">';
      html += '<div class="pc-agent" style="color:' + (isWinner ? '#6ee7b7' : i===1 ? '#fbbf24' : '#94a3b8') + ';">' + medal + ' ' + r.agent + '</div>';
      html += '<div class="pc-result ' + (r.survived ? 'survived' : 'collapsed') + '">' + (r.survived ? '✅ SURVIVED' : '💀 COLLAPSED') + '</div>';
      html += '<div class="pc-stat"><span>Score</span><span style="color:' + (r.score > 0 ? '#10b981' : '#ef4444') + ';">' + (r.score||0).toFixed(2) + '</span></div>';
      html += '<div class="pc-stat"><span>Era</span><span>' + (r.era||'tribal').toUpperCase() + '</span></div>';
      html += '<div class="pc-stat"><span>Population</span><span>' + (d2.population_final||0) + '</span></div>';
      html += '<div class="pc-stat"><span>Stability</span><span>' + ((d2.stability||0)*100).toFixed(0) + '%</span></div>';
      html += '<div class="pc-stat"><span>Crises Averted</span><span>' + (d2.crises_averted||0) + '</span></div>';
      if (d2.collapse_reason) html += '<div class="pc-stat"><span>Failure</span><span style="color:#fca5a5;">' + d2.collapse_reason + '</span></div>';
      html += '</div>';
    });
  }
  html += '</div>';
  
  // Analysis summary
  if (data.analysis) {
    html += '<div style="margin-top:16px;padding:12px;background:rgba(255,255,255,0.03);border-radius:8px;font-size:0.75rem;color:#94a3b8;">';
    html += '<strong style="color:#e2e8f0;">Analysis:</strong> ';
    html += 'Score Spread: <span style="color:#a855f7;">' + (data.analysis.score_spread||0).toFixed(2) + '</span> | ';
    html += 'Survival: <span style="color:#10b981;">' + (data.analysis.survival_rate||'0/0') + '</span> | ';
    html += (data.analysis.all_survived ? '<span style="color:#6ee7b7;">All agents survived!</span>' : '<span style="color:#fbbf24;">Some agents collapsed</span>');
    html += '</div>';
  }
  
  html += '<button class="parallel-close" onclick="closeParallelOverlay()"><i class="ph-bold ph-x" style="margin-right:6px;"></i> Close</button>';
  html += '</div>';
  overlay.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════════
// SYSTEM 3: Self-Learning Memory — Frontend Badge
// ═══════════════════════════════════════════════════════════════

async function updateMemoryBadge() {
  try {
    const r = await fetch('/api/memory/status');
    const d = await r.json();
    let badge = document.getElementById('memoryBadge');
    if (!badge) {
      badge = document.createElement('div');
      badge.id = 'memoryBadge';
      badge.className = 'memory-badge';
      badge.onclick = () => showMemoryDetails();
      document.body.appendChild(badge);
    }
    badge.innerHTML = '<span class="mem-icon">🧠</span> Memory: ' + d.total_entries + '/' + d.max_entries + ' runs';
  } catch(e) {}
}

async function showMemoryDetails() {
  try {
    const r = await fetch('/api/memory/status');
    const d = await r.json();
    let modal = document.getElementById('memoryModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'memoryModal';
      modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(5,5,15,0.9);z-index:10002;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(6px);';
      modal.onclick = (e) => { if(e.target===modal) modal.style.display='none'; };
      document.body.appendChild(modal);
    }
    let html = '<div style="background:linear-gradient(135deg,rgba(20,20,40,0.98),rgba(10,10,25,0.98));border:1px solid rgba(251,191,36,0.3);border-radius:16px;padding:28px;max-width:600px;width:95%;">';
    html += '<div style="font-size:1.2rem;font-weight:800;background:linear-gradient(135deg,#fbbf24,#f59e0b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:16px;">🧠 Self-Learning Memory</div>';
    html += '<div style="font-size:0.75rem;color:#6b7280;margin-bottom:12px;">' + d.total_entries + ' runs stored (max ' + d.max_entries + ')</div>';
    
    if (d.recent && d.recent.length > 0) {
      html += '<table style="width:100%;font-size:0.72rem;border-collapse:collapse;">';
      html += '<thead><tr style="color:#fbbf24;border-bottom:1px solid rgba(251,191,36,0.3);"><th style="text-align:left;padding:5px;">Result</th><th>Score</th><th>Era</th><th>Mode</th><th>Lesson</th></tr></thead><tbody>';
      d.recent.forEach(m => {
        const color = m.survived ? '#10b981' : '#ef4444';
        html += '<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">' +
          '<td style="padding:4px;color:' + color + ';font-weight:700;">' + (m.survived ? 'SURVIVED' : 'COLLAPSED') + '</td>' +
          '<td style="text-align:center;color:#e2e8f0;">' + (m.score||0).toFixed(2) + '</td>' +
          '<td style="text-align:center;color:#818cf8;">' + (m.era||'?').toUpperCase() + '</td>' +
          '<td style="text-align:center;color:#94a3b8;">' + (m.mode||'?') + '</td>' +
          '<td style="color:#94a3b8;max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + (m.key_lesson || m.collapse_reason || '-') + '</td></tr>';
      });
      html += '</tbody></table>';
    } else {
      html += '<div style="color:#6b7280;text-align:center;padding:20px;">No runs recorded yet. Play a game to start learning!</div>';
    }
    
    if (d.patterns && d.total_entries > 0) {
      html += '<div style="margin-top:12px;padding:10px;background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.15);border-radius:8px;font-size:0.7rem;color:#fbbf24;white-space:pre-line;">' + d.patterns + '</div>';
    }
    
    html += '<div style="display:flex;gap:10px;margin-top:16px;">';
    html += '<button onclick="closeMemoryModal()" style="flex:1;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.15);color:#e2e8f0;padding:8px;border-radius:8px;cursor:pointer;">Close</button>';
    html += '<button onclick="clearMemory()" style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:#fca5a5;padding:8px 16px;border-radius:8px;cursor:pointer;font-size:0.75rem;">Clear Memory</button>';
    html += '</div></div>';
    modal.innerHTML = html;
    modal.style.display = 'flex';
  } catch(e) {}
}

async function clearMemory() {
  if (!confirm('Clear all self-learning memory? This cannot be undone.')) return;
  try {
    await fetch('/api/memory/clear', {method: 'POST'});
    updateMemoryBadge();
    const modal = document.getElementById('memoryModal');
    if (modal) modal.style.display = 'none';
  } catch(e) {}
}

// ═══════════════════════════════════════════════════════════════
// Modal Close Helpers (avoid nested-quote issues in onclick)
// ═══════════════════════════════════════════════════════════════

function closeMetaAIModal() {
  var m = document.getElementById("metaAIModal");
  if (m) m.style.display = "none";
}

function closeParallelOverlay() {
  var m = document.getElementById("parallelOverlay");
  if (m) m.classList.remove("active");
}

function closeMemoryModal() {
  var m = document.getElementById("memoryModal");
  if (m) m.style.display = "none";
}

// ═══════════════════════════════════════════════════════════════
// TOAST NOTIFICATION ENGINE
// ═══════════════════════════════════════════════════════════════

function ensureToastContainer() {
  let c = document.getElementById('toastContainer');
  if (!c) {
    c = document.createElement('div');
    c.id = 'toastContainer';
    c.className = 'toast-container';
    document.body.appendChild(c);
  }
  return c;
}

function showToast(message, type, duration) {
  type = type || 'info';
  duration = duration || 4000;
  const container = ensureToastContainer();
  const toast = document.createElement('div');
  toast.className = 'toast toast-' + type;
  const icons = {good: '✅', bad: '⚠️', info: 'ℹ️', neutral: '📋'};
  toast.innerHTML = '<span class="toast-icon">' + (icons[type] || '📋') + '</span><span>' + message + '</span>';
  toast.onclick = function() { dismissToast(toast); };
  container.appendChild(toast);
  // Auto-dismiss
  setTimeout(function() { dismissToast(toast); }, duration);
  // Keep max 4 toasts visible
  while (container.children.length > 4) {
    container.removeChild(container.firstChild);
  }
}

function dismissToast(el) {
  if (!el || !el.parentNode) return;
  el.classList.add('toast-out');
  setTimeout(function() { if(el.parentNode) el.parentNode.removeChild(el); }, 300);
}

// ═══════════════════════════════════════════════════════════════
// FACTION COLOR-CODING HELPERS
// ═══════════════════════════════════════════════════════════════

function getFactionClass(source, subject, body) {
  var text = ((source||'') + ' ' + (subject||'') + ' ' + (body||'')).toLowerCase();
  if (/military|general|defense|defend|army|soldier|border|raid|attack|invasion|troops|fortress|guard|war/.test(text)) return 'msg-faction-military';
  if (/scientist|research|tech|lab|discovery|experiment|innovation|professor|engineer|quantum/.test(text)) return 'msg-faction-science';
  if (/trade|merchant|caravan|gold|economy|market|price|export|import|commerce/.test(text)) return 'msg-faction-trade';
  if (/citizen|people|village|protest|farmer|hunger|food|famine|starv|crops|harvest|worker/.test(text)) return 'msg-faction-citizens';
  if (/government|council|minister|law|decree|policy|governor|diplomat|kingdom|leader|summit/.test(text)) return 'msg-faction-government';
  if (/nature|disaster|storm|earthquake|flood|drought|plague|disease|weather|locust|epidemic/.test(text)) return 'msg-faction-nature';
  return '';
}

function getFactionIcon(factionClass) {
  var icons = {
    'msg-faction-military': '<i class="ph-fill ph-shield-chevron" style="color:#ef4444;margin-right:4px;"></i>',
    'msg-faction-science': '<i class="ph-fill ph-atom" style="color:#8b5cf6;margin-right:4px;"></i>',
    'msg-faction-trade': '<i class="ph-fill ph-coins" style="color:#f59e0b;margin-right:4px;"></i>',
    'msg-faction-citizens': '<i class="ph-fill ph-users" style="color:#10b981;margin-right:4px;"></i>',
    'msg-faction-government': '<i class="ph-fill ph-bank" style="color:#3b82f6;margin-right:4px;"></i>',
    'msg-faction-nature': '<i class="ph-fill ph-cloud-lightning" style="color:#06b6d4;margin-right:4px;"></i>'
  };
  return icons[factionClass] || '';
}

// ═══════════════════════════════════════════════════════════════
// KEYBOARD HOTKEYS
// ═══════════════════════════════════════════════════════════════

const HOTKEY_MAP = {
  '1': 'allocate_food', '2': 'allocate_workers', '3': 'approve_research',
  '4': 'defend', '5': 'calm_citizens', '6': 'accept_trade',
  '7': 'reject_trade', '8': 'invest_growth', '9': 'emergency_response',
  '0': 'ignore'
};

document.addEventListener('keydown', function(e) {
  // Don't capture when typing in inputs, modals open, or game not active
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  if (e.ctrlKey || e.altKey || e.metaKey) return;
  const actPanel = document.getElementById('actionsPanel');
  if (!actPanel || actPanel.classList.contains('hidden')) return;
  
  const action = HOTKEY_MAP[e.key];
  if (action) {
    e.preventDefault();
    act(action);
    // Flash the button 
    const btns = actPanel.querySelectorAll('.act-btn');
    btns.forEach(function(b) {
      if (b.onclick && b.onclick.toString().indexOf(action) > -1) {
        b.style.transform = 'scale(0.95)';
        b.style.filter = 'brightness(1.4)';
        setTimeout(function() { b.style.transform = ''; b.style.filter = ''; }, 200);
      }
    });
  }
  // 'a' = Ask Advisor
  if (e.key === 'a' || e.key === 'A') {
    const advBtn = document.getElementById('advisorBtn');
    if (advBtn && !advBtn.classList.contains('hidden')) { e.preventDefault(); askAdvisor(); }
  }
});

// ═══════════════════════════════════════════════════════════════
// STICKY DASHBOARD ACTIVATOR
// ═══════════════════════════════════════════════════════════════

function activateStickyDashboard() {
  const panel = document.getElementById('statsPanel');
  if (!panel) return;
  // Use IntersectionObserver for sticky activation.
  // When the panel scrolls above the viewport, add 'sticky-active' class.
  const sentinel = document.createElement('div');
  sentinel.id = 'stickySentinel';
  sentinel.style.cssText = 'height:1px;width:100%;pointer-events:none;';
  panel.parentNode.insertBefore(sentinel, panel);
  const observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (!entry.isIntersecting) {
        panel.classList.add('sticky-active');
      } else {
        panel.classList.remove('sticky-active');
      }
    });
  }, { threshold: 0 });
  observer.observe(sentinel);
}

// ═══════════════════════════════════════════════════════════════
// ADD HOTKEY BADGES TO ACTION BUTTONS
// ═══════════════════════════════════════════════════════════════

function addHotkeyBadges() {
  const actPanel = document.getElementById('actionsPanel');
  if (!actPanel) return;
  const btns = actPanel.querySelectorAll('.act-btn');
  const keys = ['1','2','3','4','5','6','7','8','9','0'];
  btns.forEach(function(btn, i) {
    if (i < keys.length && !btn.querySelector('.hotkey-badge')) {
      btn.style.position = 'relative';
      const badge = document.createElement('span');
      badge.className = 'hotkey-badge';
      badge.textContent = keys[i];
      btn.appendChild(badge);
    }
  });
}

// ═══════════════════════════════════════════════════════════════
// FEATURE 1: CHRONO-FRACTAL TEMPORAL REWIND
// ═══════════════════════════════════════════════════════════════

let chronoSingularityTurn = -1;

function findCausalSingularity(log) {
  // Find the turn where things became irrecoverable
  // Strategy: find the first turn where a critical metric started a sustained decline
  if (!log || log.length < 3) return -1;
  let worst = -1, worstScore = Infinity;
  for (let i = 1; i < log.length; i++) {
    var s = log[i].state || log[i].observation || {};
    var food = parseInt(s.food) || 100;
    var morale = parseInt(s.morale) || 60;
    var danger = parseInt(s.danger_level || s.danger) || 0;
    var score = food + morale - danger * 2;
    if (score < worstScore) { worstScore = score; worst = i; }
  }
  // Walk backwards from worst to find first sustained decline
  for (var j = Math.max(1, worst - 3); j <= worst; j++) {
    var prev = log[j-1].state || log[j-1].observation || {};
    var curr = log[j].state || log[j].observation || {};
    var pFood = parseInt(prev.food) || 100;
    var cFood = parseInt(curr.food) || 100;
    var pMorale = parseInt(prev.morale) || 60;
    var cMorale = parseInt(curr.morale) || 60;
    if (cFood < pFood && cMorale < pMorale) return j;
  }
  return worst > 0 ? worst : -1;
}

function openChronoRewind() {
  if (episodeLog.length < 2) { showToast("Play at least 2 turns before rewinding.", "info"); return; }
  var overlay = document.getElementById("chronoOverlay");
  overlay.classList.add("active");
  var slider = document.getElementById("chronoSlider");
  slider.max = episodeLog.length - 1;
  slider.value = episodeLog.length - 1;
  document.getElementById("chronoTurnMax").textContent = "/ " + (episodeLog.length - 1);
  chronoSingularityTurn = findCausalSingularity(episodeLog);
  // Render singularity marker
  var wrap = document.getElementById("chronoSliderWrap");
  var existing = wrap.querySelectorAll(".chrono-singularity-marker, .chrono-singularity-label");
  existing.forEach(function(e) { e.remove(); });
  if (chronoSingularityTurn > 0 && episodeLog.length > 1) {
    var pct = (chronoSingularityTurn / (episodeLog.length - 1)) * 100;
    var marker = document.createElement("div");
    marker.className = "chrono-singularity-marker";
    marker.style.left = pct + "%";
    marker.title = "Causal Singularity — Turn " + chronoSingularityTurn;
    marker.onclick = function() { slider.value = chronoSingularityTurn; onChronoSlide(chronoSingularityTurn); };
    var label = document.createElement("div");
    label.className = "chrono-singularity-label";
    label.style.left = pct + "%";
    label.textContent = "SINGULARITY";
    wrap.appendChild(marker);
    wrap.appendChild(label);
  }
  onChronoSlide(slider.value);
}

function closeChronoRewind() {
  document.getElementById("chronoOverlay").classList.remove("active");
}

function onChronoSlide(val) {
  var idx = parseInt(val);
  var entry = episodeLog[idx];
  if (!entry) return;
  var s = entry.state || entry.observation || {};
  document.getElementById("chronoTurnLabel").textContent = "Turn " + (s.turn || idx);
  // Display stats
  var statConfigs = [
    {label: "Population", val: s.population || 0, color: "#38bdf8"},
    {label: "Food", val: s.food || 0, color: "#10b981"},
    {label: "Energy", val: s.energy || 0, color: "#fbbf24"},
    {label: "Morale", val: s.morale || 0, color: "#a78bfa"},
    {label: "Tech", val: s.technology_level || 0, color: "#06b6d4"},
    {label: "Danger", val: s.danger_level || s.danger || 0, color: "#ef4444"},
    {label: "Workers", val: s.available_workers || 0, color: "#f97316"}
  ];
  var stateHtml = "";
  statConfigs.forEach(function(c) {
    stateHtml += '<div class="chrono-stat"><div class="cs-val" style="color:' + c.color + '">' + c.val + '</div><div class="cs-lbl">' + c.label + "</div></div>";
  });
  document.getElementById("chronoStateDisplay").innerHTML = stateHtml;
  // Action
  var actionText = entry.action ? entry.action.split("_").map(function(w){return w.charAt(0).toUpperCase()+w.slice(1)}).join(" ") : "—";
  document.getElementById("chronoActionDisplay").innerHTML = '<i class="ph-fill ph-lightning" style="margin-right:8px"></i> Action: <strong>' + actionText + "</strong>" + (entry.reward !== undefined ? " | Reward: " + (entry.reward >= 0 ? "+" : "") + (typeof entry.reward === "number" ? entry.reward.toFixed(3) : entry.reward) : "");
  // Feedback
  var fb = document.getElementById("chronoFeedback");
  if (idx === chronoSingularityTurn) {
    fb.className = "chrono-feedback bad";
    fb.innerHTML = '<i class="ph-fill ph-warning-diamond" style="margin-right:6px"></i> CAUSAL SINGULARITY — This is where the collapse became inevitable';
  } else if (entry.reward !== undefined && entry.reward >= 0.1) {
    fb.className = "chrono-feedback good";
    fb.innerHTML = '<i class="ph-fill ph-check-circle" style="margin-right:6px"></i> Strong decision — positive trajectory';
  } else if (entry.reward !== undefined && entry.reward < -0.05) {
    fb.className = "chrono-feedback bad";
    fb.innerHTML = '<i class="ph-fill ph-warning" style="margin-right:6px"></i> Poor decision — contributed to decline';
  } else {
    fb.className = "chrono-feedback";
    fb.innerHTML = "";
  }
  // Messages from that turn
  var msgs = (s.messages || []);
  if (msgs.length > 0) {
    document.getElementById("chronoMessagesDisplay").innerHTML = msgs.map(function(m) { return '<div style="margin-bottom:4px;"><strong style="color:#e2e8f0">' + (m.source || "?") + ":</strong> " + (m.subject || m.body || "") + "</div>"; }).join("");
  } else {
    document.getElementById("chronoMessagesDisplay").innerHTML = '<span style="color:#4b5563">No messages at this point.</span>';
  }
  // Play audio frequency mapped to state health
  if (isAudioEnabled) {
    var ctx = getAudioCtx();
    if (ctx) {
      var health = ((parseInt(s.food)||50) + (parseInt(s.morale)||50)) / 2;
      var freq = 200 + (health / 100) * 600;
      var osc = ctx.createOscillator();
      var gain = ctx.createGain();
      osc.connect(gain).connect(ctx.destination);
      osc.frequency.value = freq;
      osc.type = "sine";
      gain.gain.value = 0.02;
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.1);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// FEATURE 2: TESSERACT TIMELINE (QUANTUM BRANCHING)
// ═══════════════════════════════════════════════════════════════

function openTesseract() {
  document.getElementById("tesseractOverlay").classList.add("active");
  document.getElementById("tesseractBody").innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:60px;color:#64748b;"><i class="ph-fill ph-git-branch" style="font-size:3rem;display:block;margin-bottom:16px;color:#818cf8"></i>Click "Run Quantum Branch Simulation" to split one seed into two divergent AI strategies and compare outcomes side-by-side.</div>';
}

function closeTesseract() {
  document.getElementById("tesseractOverlay").classList.remove("active");
}

async function runTesseractSim() {
  var btn = document.getElementById("tesseractRunBtn");
  btn.disabled = true;
  btn.innerHTML = '<i class="ph-bold ph-spinner ph-spin" style="margin-right:4px"></i> Branching timelines...';
  document.getElementById("tesseractBody").innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:60px;color:#818cf8;"><div class="think-spinner" style="display:inline-block;margin-bottom:16px"></div><br>Simulating two parallel universes on the same seed...<br><span style="font-size:0.7rem;color:#4b5563">Logical vs Greedy agents</span></div>';
  try {
    var taskSel = document.getElementById("taskSel");
    var taskId = taskSel ? taskSel.value : "task_demo_10";
    var r = await fetch("/api/simulate/parallel", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({task_id: taskId, agents: ["logical", "greedy"]})
    });
    var data = await r.json();
    renderTesseractBranches(data);
  } catch(e) {
    document.getElementById("tesseractBody").innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:40px;color:#fca5a5">Quantum branching failed: ' + e.message + "</div>";
  }
  btn.disabled = false;
  btn.innerHTML = '<i class="ph-fill ph-play" style="margin-right:4px"></i> Run Again';
}

function renderTesseractBranches(data) {
  if (!data || !data.results) return;
  var agents = Object.keys(data.results);
  var a = data.results[agents[0]] || {};
  var b = data.results[agents[1]] || {};
  function buildBranch(name, d, cls, labelCls) {
    var survived = d.survived !== false && !d.collapse;
    var html = '<div class="tesseract-branch ' + cls + '">';
    html += '<div class="tesseract-branch-label ' + labelCls + '"><i class="ph-fill ph-git-branch" style="margin-right:6px"></i>Universe ' + name + " — " + (agents[name === "A" ? 0 : 1] || "?").toUpperCase() + "</div>";
    html += '<div class="tesseract-stats">';
    html += '<div class="ts-stat"><div class="ts-v" style="color:#38bdf8">' + (d.population_final||0) + '</div><div class="ts-l">Population</div></div>';
    html += '<div class="ts-stat"><div class="ts-v" style="color:#10b981">' + ((d.score||0).toFixed(2)) + '</div><div class="ts-l">Score</div></div>';
    html += '<div class="ts-stat"><div class="ts-v" style="color:#a78bfa">' + (d.era||"?").toUpperCase() + '</div><div class="ts-l">Era</div></div>';
    html += '<div class="ts-stat"><div class="ts-v" style="color:#fbbf24">' + ((d.stability||0)*100).toFixed(0) + '%</div><div class="ts-l">Stability</div></div>';
    html += "</div>";
    // Decision log
    html += '<div class="tesseract-log">';
    if (d.decisions && d.decisions.length > 0) {
      d.decisions.forEach(function(dec, i) {
        var eff = dec.effective !== false;
        html += '<div class="tesseract-log-entry ' + (eff ? "effective" : "ineffective") + '">T' + (i+1) + ": " + (dec.action || dec).replace(/_/g, " ") + (dec.reward !== undefined ? " (" + (dec.reward >= 0 ? "+" : "") + dec.reward.toFixed(2) + ")" : "") + "</div>";
      });
    } else {
      html += '<div style="color:#4b5563">Decision log unavailable</div>';
    }
    html += "</div>";
    // Result
    html += '<div class="tesseract-result ' + (survived ? "survived" : "collapsed") + '">' + (survived ? "SURVIVED — " + (d.era||"tribal").toUpperCase() + " ERA" : "COLLAPSED — " + (d.collapse_reason || "Unknown")) + "</div>";
    html += "</div>";
    return html;
  }
  var html = buildBranch("A", a, "tesseract-branch-a", "tb-label-a");
  html += '<div class="tesseract-divider"><div class="tesseract-divider-line"></div><i class="ph-fill ph-git-merge" style="color:#a855f7;font-size:1.2rem"></i><div style="font-size:0.55rem;writing-mode:vertical-rl;color:#64748b">QUANTUM SPLIT</div><div class="tesseract-divider-line"></div></div>';
  html += buildBranch("B", b, "tesseract-branch-b", "tb-label-b");
  document.getElementById("tesseractBody").innerHTML = html;
}

// ═══════════════════════════════════════════════════════════════
// FEATURE 3: NEURAL CARTOGRAPHY (GENERATIVE VISUALIZATION)
// ═══════════════════════════════════════════════════════════════

let neuroNodes = [];
let neuroFrame = 0;

function initNeuroCanvas() {
  var wrap = document.getElementById("neuroCanvasWrap");
  if (!wrap) return;
  wrap.style.display = "block";
  var canvas = document.getElementById("neuroCanvas");
  var rect = wrap.getBoundingClientRect();
  canvas.width = rect.width * (window.devicePixelRatio || 1);
  canvas.height = rect.height * (window.devicePixelRatio || 1);
  neuroNodes = [];
  // Seed initial nodes
  for (var i = 0; i < 20; i++) {
    neuroNodes.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      r: 2 + Math.random() * 3,
      life: 1,
      type: "base"
    });
  }
  requestAnimationFrame(renderNeuroFrame);
}

function renderNeuroFrame() {
  var canvas = document.getElementById("neuroCanvas");
  if (!canvas || !canvas.parentElement || canvas.parentElement.style.display === "none") return;
  var ctx = canvas.getContext("2d");
  var w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  // Read current game state
  var pop = parseInt(document.getElementById("vPop")?.textContent) || 100;
  var food = parseInt(document.getElementById("vFood")?.textContent) || 100;
  var morale = parseInt(document.getElementById("vMorale")?.textContent) || 50;
  var danger = parseInt(document.getElementById("vDanger")?.textContent) || 0;
  // Health ratio drives colors
  var health = Math.max(0, Math.min(1, ((food + morale) / 200)));
  var threat = Math.min(1, danger / 30);
  // Population drives node count
  var targetNodes = Math.min(80, Math.max(10, Math.round(pop / 5)));
  while (neuroNodes.length < targetNodes) {
    neuroNodes.push({ x: w/2 + (Math.random()-0.5)*w*0.6, y: h/2 + (Math.random()-0.5)*h*0.6, vx: (Math.random()-0.5)*0.4, vy: (Math.random()-0.5)*0.4, r: 1.5+Math.random()*2.5, life: 1, type: "base" });
  }
  while (neuroNodes.length > targetNodes && neuroNodes.length > 5) {
    neuroNodes.pop();
  }
  // Danger corruption particles
  if (threat > 0.3 && Math.random() < threat * 0.3) {
    neuroNodes.push({ x: Math.random() < 0.5 ? 0 : w, y: Math.random()*h, vx: (Math.random()-0.5)*2, vy: (Math.random()-0.5)*2, r: 2+Math.random()*4, life: 1, type: "threat" });
  }
  neuroFrame++;
  // Update and draw
  ctx.globalAlpha = 0.15 + health * 0.3; // dim when morale/food low
  // Draw connections
  for (var i = 0; i < neuroNodes.length; i++) {
    for (var j = i+1; j < neuroNodes.length; j++) {
      var dx = neuroNodes[i].x - neuroNodes[j].x;
      var dy = neuroNodes[i].y - neuroNodes[j].y;
      var dist = Math.sqrt(dx*dx + dy*dy);
      if (dist < 80) {
        ctx.beginPath();
        ctx.moveTo(neuroNodes[i].x, neuroNodes[i].y);
        ctx.lineTo(neuroNodes[j].x, neuroNodes[j].y);
        var isT = neuroNodes[i].type === "threat" || neuroNodes[j].type === "threat";
        ctx.strokeStyle = isT ? "rgba(239,68,68," + (0.1 * (1-dist/80)) + ")" : "rgba(99,102,241," + (0.15 * (1-dist/80)) + ")";
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }
  }
  ctx.globalAlpha = 1;
  // Draw nodes
  neuroNodes.forEach(function(n) {
    n.x += n.vx + Math.sin(neuroFrame * 0.01 + n.y * 0.01) * 0.2;
    n.y += n.vy + Math.cos(neuroFrame * 0.01 + n.x * 0.01) * 0.15;
    if (n.x < 0) n.x = w; if (n.x > w) n.x = 0;
    if (n.y < 0) n.y = h; if (n.y > h) n.y = 0;
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    if (n.type === "threat") {
      ctx.fillStyle = "rgba(239,68,68," + (0.4 + Math.sin(neuroFrame*0.05)*0.2) + ")";
      ctx.shadowColor = "#ef4444"; ctx.shadowBlur = 8;
    } else {
      var hue = 230 + health * 100; // 230 (cold blue) to 330 (warm purple)
      ctx.fillStyle = "hsla(" + hue + ",70%,60%," + (0.3 + health * 0.4) + ")";
      ctx.shadowColor = "hsl(" + hue + ",70%,60%)"; ctx.shadowBlur = 4;
    }
    ctx.fill();
    ctx.shadowBlur = 0;
  });
  // Remove expired threat nodes
  neuroNodes = neuroNodes.filter(function(n) { return n.type !== "threat" || Math.random() > 0.005; });
  requestAnimationFrame(renderNeuroFrame);
}

// ═══════════════════════════════════════════════════════════════
// FEATURE 4: CONTROLLED SELF-EVOLUTION
// ═══════════════════════════════════════════════════════════════

var evolvedCapabilities = {};
var EVO_MILESTONES = [
  { trigger: function(s) { return parseInt(s.technology_level) >= 5; }, id: "trade_network", icon: "🌐", title: "Trade Network Unlocked", desc: "Your scientists have discovered interconnected trading routes. The civilization can now establish economic alliances.", action: "Establish Trade Network", actionId: "accept_trade" },
  { trigger: function(s) { return parseInt(s.population) >= 200; }, id: "grand_council", icon: "🏛️", title: "Grand Council Assembled", desc: "Your population has reached critical mass. A governing council can now advise on complex diplomatic matters.", action: "Convene Grand Council", actionId: "calm_citizens" },
  { trigger: function(s) { return (parseInt(s.danger_level || s.danger) || 0) === 0 && parseInt(s.morale) > 70; }, id: "golden_age", icon: "👑", title: "Golden Age Protocol", desc: "Zero threats detected and high morale. Your civilization can now enter an accelerated growth phase.", action: "Activate Golden Age", actionId: "invest_growth" },
  { trigger: function(s) { return parseInt(s.technology_level) >= 8; }, id: "quantum_defense", icon: "⚛️", title: "Quantum Defense Grid", desc: "Advanced physics has unlocked a next-generation shield system for your borders.", action: "Deploy Quantum Shield", actionId: "defend" },
  { trigger: function(s) { return parseInt(s.food) >= 300; }, id: "hydroponics", icon: "🌿", title: "Hydroponic Revolution", desc: "Surplus food reserves have funded radical agricultural innovation. Food production is now self-sustaining.", action: "Enable Hydroponics", actionId: "allocate_food" }
];

function checkEvolution(observation) {
  if (!observation) return;
  EVO_MILESTONES.forEach(function(m) {
    if (evolvedCapabilities[m.id]) return;
    if (m.trigger(observation)) {
      evolvedCapabilities[m.id] = true;
      showEvolutionUnlock(m);
      // Add visual badge to the matching button
      setTimeout(function() {
        var btns = document.querySelectorAll(".act-btn");
        btns.forEach(function(btn) {
          if (btn.onclick && btn.onclick.toString().indexOf(m.actionId) > -1) {
            btn.classList.add("evolved-btn");
            var badge = document.createElement("span");
            badge.className = "evolved-badge";
            badge.title = m.title;
            btn.appendChild(badge);
          }
        });
      }, 3000);
    }
  });
}

function showEvolutionUnlock(milestone) {
  var toast = document.getElementById("evoUnlockToast");
  document.getElementById("evoIcon").textContent = milestone.icon;
  document.getElementById("evoTitle").textContent = milestone.title;
  document.getElementById("evoDesc").textContent = milestone.desc;
  document.getElementById("evoAction").textContent = milestone.action;
  toast.classList.add("active");
  // Dramatic sound
  if (isAudioEnabled) {
    var ctx = getAudioCtx();
    if (ctx) {
      [523, 659, 784, 1047].forEach(function(f, i) {
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        osc.connect(gain).connect(ctx.destination);
        osc.frequency.value = f; osc.type = "sine";
        gain.gain.value = 0.05;
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5 + i*0.15);
        osc.start(ctx.currentTime + i*0.12);
        osc.stop(ctx.currentTime + 0.6 + i*0.15);
      });
    }
  }
  showToast(milestone.icon + " " + milestone.title, "good", 5000);
  setTimeout(function() { toast.classList.remove("active"); }, 4000);
}

// ═══════════════════════════════════════════════════════════════
// FEATURE 5: MULTI-MODAL VISION + AUDIO
// ═══════════════════════════════════════════════════════════════

var CRISIS_VISUALS = {
  "attack": { gradient: "linear-gradient(135deg,rgba(239,68,68,0.3),rgba(127,29,29,0.5))", icon: "ph-sword", color: "#ef4444" },
  "raid": { gradient: "linear-gradient(135deg,rgba(239,68,68,0.3),rgba(127,29,29,0.5))", icon: "ph-crosshair", color: "#ef4444" },
  "invasion": { gradient: "linear-gradient(135deg,rgba(185,28,28,0.4),rgba(0,0,0,0.6))", icon: "ph-shield-warning", color: "#fca5a5" },
  "flood": { gradient: "linear-gradient(135deg,rgba(6,182,212,0.3),rgba(12,74,110,0.5))", icon: "ph-waves", color: "#22d3ee" },
  "drought": { gradient: "linear-gradient(135deg,rgba(234,179,8,0.3),rgba(113,63,18,0.5))", icon: "ph-sun", color: "#fbbf24" },
  "famine": { gradient: "linear-gradient(135deg,rgba(251,146,60,0.3),rgba(124,45,18,0.5))", icon: "ph-bowl-food", color: "#fb923c" },
  "plague": { gradient: "linear-gradient(135deg,rgba(34,197,94,0.2),rgba(20,83,45,0.5))", icon: "ph-virus", color: "#4ade80" },
  "earthquake": { gradient: "linear-gradient(135deg,rgba(168,85,247,0.3),rgba(88,28,135,0.5))", icon: "ph-mountains", color: "#c084fc" },
  "fire": { gradient: "linear-gradient(135deg,rgba(249,115,22,0.4),rgba(124,45,18,0.5))", icon: "ph-fire", color: "#f97316" },
  "storm": { gradient: "linear-gradient(135deg,rgba(99,102,241,0.3),rgba(30,27,75,0.5))", icon: "ph-cloud-lightning", color: "#818cf8" },
  "protest": { gradient: "linear-gradient(135deg,rgba(251,191,36,0.25),rgba(113,63,18,0.4))", icon: "ph-megaphone", color: "#fbbf24" },
  "unrest": { gradient: "linear-gradient(135deg,rgba(239,68,68,0.2),rgba(127,29,29,0.3))", icon: "ph-users-three", color: "#f87171" },
  "trade": { gradient: "linear-gradient(135deg,rgba(16,185,129,0.25),rgba(6,78,59,0.4))", icon: "ph-handshake", color: "#34d399" },
  "discovery": { gradient: "linear-gradient(135deg,rgba(99,102,241,0.3),rgba(49,46,129,0.5))", icon: "ph-sparkle", color: "#a5b4fc" },
  "research": { gradient: "linear-gradient(135deg,rgba(168,85,247,0.25),rgba(76,29,149,0.4))", icon: "ph-atom", color: "#c4b5fd" }
};

function addCrisisVisionToCard(card, subject, body) {
  var text = ((subject || "") + " " + (body || "")).toLowerCase();
  var matched = null;
  for (var key in CRISIS_VISUALS) {
    if (text.indexOf(key) > -1) { matched = CRISIS_VISUALS[key]; break; }
  }
  if (!matched) return;
  var vCard = document.createElement("div");
  vCard.className = "crisis-vision-card";
  vCard.style.background = matched.gradient;
  vCard.innerHTML = '<div class="crisis-vision-text"><i class="ph-fill ' + matched.icon + '" style="color:' + matched.color + ';margin-right:6px;font-size:1.1rem"></i> Visual Alert: ' + (subject || "Crisis Detected").substring(0, 60) + '<span class="audio-pulse-indicator"><span class="audio-pulse-bar"></span><span class="audio-pulse-bar"></span><span class="audio-pulse-bar"></span><span class="audio-pulse-bar"></span><span class="audio-pulse-bar"></span></span></div>';
  card.appendChild(vCard);
}

function playCrisisAudio(urgency, subject) {
  if (!isAudioEnabled) return;
  var ctx = getAudioCtx();
  if (!ctx) return;
  var text = (subject || "").toLowerCase();
  var freqs, waveType;
  if (urgency === "critical" || text.indexOf("attack") > -1 || text.indexOf("invasion") > -1) {
    freqs = [440, 523, 440, 523]; waveType = "sawtooth";
  } else if (urgency === "high" || text.indexOf("flood") > -1 || text.indexOf("plague") > -1) {
    freqs = [330, 392, 330]; waveType = "triangle";
  } else if (text.indexOf("trade") > -1 || text.indexOf("discovery") > -1) {
    freqs = [523, 659, 784]; waveType = "sine";
  } else {
    return; // No audio for low-urgency events
  }
  freqs.forEach(function(f, i) {
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.connect(gain).connect(ctx.destination);
    osc.frequency.value = f;
    osc.type = waveType;
    gain.gain.value = 0.03;
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.2 + i*0.1);
    osc.start(ctx.currentTime + i*0.08);
    osc.stop(ctx.currentTime + 0.25 + i*0.1);
  });
}

// ═══════════════════════════════════════════════════════════════
// FEATURE 6: CAUSAL SINGULARITY MARKERS (enhance existing causality web)
// ═══════════════════════════════════════════════════════════════

var singularityTurns = [];

function detectSingularityInGame(observation) {
  if (!observation) return false;
  var food = parseInt(observation.food) || 100;
  var morale = parseInt(observation.morale) || 50;
  var danger = parseInt(observation.danger_level || observation.danger) || 0;
  // A singularity occurs when multiple critical thresholds are breached simultaneously
  var crisisCount = 0;
  if (food < 40) crisisCount++;
  if (morale < 25) crisisCount++;
  if (danger >= 20) crisisCount++;
  if (crisisCount >= 2) {
    var turn = observation.turn || 0;
    if (singularityTurns.indexOf(turn) === -1) {
      singularityTurns.push(turn);
      return true;
    }
  }
  return false;
}

// ═══════════════════════════════════════════════════════════════
// INTEGRATION: Hook features into existing updateUI and showDone
// ═══════════════════════════════════════════════════════════════

// Save original updateUI reference
var _originalUpdateUI = updateUI;
updateUI = function(d) {
  _originalUpdateUI(d);
  var o = d.observation;
  // FEATURE 3: Neural Cartography — canvas auto-starts
  if (o && !document.getElementById("neuroCanvasWrap").dataset.init) {
    document.getElementById("neuroCanvasWrap").dataset.init = "1";
    initNeuroCanvas();
  }
  // FEATURE 4: Controlled Self-Evolution
  if (o) checkEvolution(o);
  // FEATURE 5: Multi-Modal Vision — inject crisis cards
  if (o && o.messages) {
    setTimeout(function() {
      var cards = document.querySelectorAll(".msg-card");
      cards.forEach(function(card) {
        if (card.dataset.vision) return;
        card.dataset.vision = "1";
        var subj = card.querySelector(".msg-subject");
        var body = card.querySelector(".msg-body");
        var urgency = card.querySelector(".msg-urgency");
        addCrisisVisionToCard(card, subj ? subj.textContent : "", body ? body.textContent : "");
        // Play audio for critical/high urgency
        if (urgency) {
          var urg = urgency.textContent.trim().toLowerCase();
          if (urg === "critical" || urg === "high") {
            playCrisisAudio(urg, subj ? subj.textContent : "");
          }
        }
      });
    }, 100);
  }
  // FEATURE 6: Detect singularity during live play
  if (o && detectSingularityInGame(o)) {
    showToast('<i class="ph-fill ph-warning-diamond" style="margin-right:4px"></i> CAUSAL SINGULARITY DETECTED — Multiple systems failing simultaneously!', "bad", 6000);
  }
};

// Save original showDone reference
var _originalShowDone = showDone;
showDone = function(d) {
  _originalShowDone(d);
  // Add Chrono-Fractal Rewind button to end overlay
  setTimeout(function() {
    var endContent = document.querySelector(".end-content");
    if (endContent && episodeLog.length > 2) {
      var existingChrono = document.getElementById("chronoEndBtn");
      if (!existingChrono) {
        var btn = document.createElement("button");
        btn.id = "chronoEndBtn";
        btn.className = "end-play-again";
        btn.style.cssText = "background:linear-gradient(135deg,rgba(99,102,241,0.3),rgba(168,85,247,0.3));border:1px solid rgba(168,85,247,0.4);margin-top:8px;";
        btn.innerHTML = '<i class="ph-fill ph-clock-countdown" style="margin-right:8px"></i> Chrono-Fractal Rewind';
        btn.onclick = function() { dismissEnd(); openChronoRewind(); };
        endContent.appendChild(btn);
      }
    }
  }, 500);
};

// ═══════════════════════════════════════════════════════════════
// INTEGRATION: Reset evolved state on game restart
// ═══════════════════════════════════════════════════════════════

var _originalResetGame = resetGame;
resetGame = async function() {
  evolvedCapabilities = {};
  singularityTurns = [];
  neuroNodes = [];
  neuroFrame = 0;
  var wrap = document.getElementById("neuroCanvasWrap");
  if (wrap) { wrap.dataset.init = ""; wrap.style.display = "none"; }
  // Remove evolved badges
  document.querySelectorAll(".evolved-btn").forEach(function(b) { b.classList.remove("evolved-btn"); });
  document.querySelectorAll(".evolved-badge").forEach(function(b) { b.remove(); });
  await _originalResetGame();
};

// ═══════════════════════════════════════════════════════════════
// Init: Boot all systems on page load
// ═══════════════════════════════════════════════════════════════

(function initAdvancedSystems() {
  // Delay to not block initial page render
  setTimeout(function() {
    fetchMetaAIStatus();   // SYSTEM 1
    updateMemoryBadge();    // SYSTEM 3
    ensureToastContainer(); // Toast system
    activateStickyDashboard(); // Sticky glass dashboard
    addHotkeyBadges();      // Keyboard hotkeys
    
    // Add Parallel Sim button next to existing controls
    var modeTag = document.querySelector('.mode-tag') || document.getElementById('councilBtn');
    if (modeTag && !document.getElementById('parallelBtn')) {
      var btn = document.createElement('button');
      btn.id = 'parallelBtn';
      btn.className = 'parallel-btn';
      btn.innerHTML = '<i class="ph-fill ph-trophy"></i> AI Race';
      btn.title = 'Run 3 AI agents simultaneously and compare results';
      btn.onclick = runParallelSim;
      modeTag.parentNode.insertBefore(btn, modeTag.nextSibling);
    }
    
    // FEATURE 1: Add Chrono-Fractal button next to export
    var expBtn = document.getElementById("exportBtn");
    if (expBtn && !document.getElementById("chronoBtn")) {
      var cBtn = document.createElement("button");
      cBtn.id = "chronoBtn";
      cBtn.className = "dataset-btn-raw";
      cBtn.style.cssText = "display:none;background:linear-gradient(135deg,rgba(99,102,241,0.2),rgba(168,85,247,0.2));border-color:rgba(168,85,247,0.3);color:#c4b5fd;";
      cBtn.innerHTML = '<i class="ph-fill ph-clock-countdown"></i> Chrono Rewind';
      cBtn.onclick = openChronoRewind;
      expBtn.parentNode.insertBefore(cBtn, expBtn.nextSibling);
    }
    
    // FEATURE 2: Add Tesseract button next to AI Race
    setTimeout(function() {
      var pBtn = document.getElementById("parallelBtn");
      if (pBtn && !document.getElementById("tesseractBtn")) {
        var tBtn = document.createElement("button");
        tBtn.id = "tesseractBtn";
        tBtn.className = "parallel-btn";
        tBtn.style.cssText = "background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(236,72,153,0.15));border-color:rgba(99,102,241,0.3);";
        tBtn.innerHTML = '<i class="ph-fill ph-git-branch"></i> Tesseract';
        tBtn.title = "Split timeline into quantum branches";
        tBtn.onclick = openTesseract;
        pBtn.parentNode.insertBefore(tBtn, pBtn.nextSibling);
      }
    }, 100);
    
    // Refresh Meta-AI status periodically
    setInterval(function() { fetchMetaAIStatus(); }, 60000);
  }, 2000);
})();

</script>
</body>
</html>"""


def run_server():
    """CLI entry point for `uv run server` (OpenEnv convention)."""
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    uvicorn.run("server.app:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)
