"""
Civil Command Center - Baseline Inference Script
==================================================
Runs an LLM agent through the civilization leader environment
across all 3 tasks, using the OpenAI-compatible API.

Environment Variables:
    API_BASE_URL   (default: https://api.openai.com/v1)
    MODEL_NAME     (default: gpt-4.1-mini)
    HF_TOKEN       Hugging Face API token (mandatory)
"""

import os
import sys
import json
import time
import textwrap
from typing import List, Dict, Any

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import CivilCommandCenter
from models import CivAction
from graders.grader_easy import grade_easy
from graders.grader_medium import grade_medium
from graders.grader_hard import grade_hard


# ===================================================================
# Config - reads from environment, NEVER hardcoded
# ===================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SEED = 42
MAX_RETRIES = 2
TEMPERATURE = 0.2
MAX_TOKENS = 250
ENV_NAME = "civil-command-center"


# ===================================================================
# System Prompt
# ===================================================================

SYSTEM_PROMPT = textwrap.dedent("""
You are the leader of a growing civilization. Each turn you receive messages
from your citizens, scientists, workers, defense, traders, and events.
You must choose ONE action to take.

Available actions (respond in JSON only):

1. ALLOCATE_FOOD - distribute food to hungry citizens
   {"action_type": "allocate_food", "target_message_id": "<msg_id>"}

2. ALLOCATE_WORKERS - assign workers to a task
   {"action_type": "allocate_workers", "target_message_id": "<msg_id>"}

3. APPROVE_RESEARCH - invest in technology
   {"action_type": "approve_research", "target_message_id": "<msg_id>"}

4. DEFEND - military defense against threats
   {"action_type": "defend", "target_message_id": "<msg_id>"}

5. CALM_CITIZENS - address morale and unrest
   {"action_type": "calm_citizens", "target_message_id": "<msg_id>"}

6. ACCEPT_TRADE - accept a trade offer
   {"action_type": "accept_trade", "target_message_id": "<msg_id>"}

7. REJECT_TRADE - reject a suspicious or bad trade
   {"action_type": "reject_trade", "target_message_id": "<msg_id>"}

8. INVEST_GROWTH - invest in infrastructure and expansion
   {"action_type": "invest_growth", "target_message_id": "<msg_id>"}

9. EMERGENCY_RESPONSE - handle disasters and crises
   {"action_type": "emergency_response", "target_message_id": "<msg_id>"}

10. IGNORE - do nothing (risky if urgent messages exist)
    {"action_type": "ignore"}

Strategy tips:
- ALWAYS handle CRITICAL and HIGH urgency messages first
- Defend against enemy raids immediately or lose population
- Emergency response for disasters (floods, fires, disease)
- Approve research when possible to advance technology
- Calm citizens when morale is low
- Accept good trades, reject suspicious ones
- Balance short-term survival with long-term growth

Respond with ONLY valid JSON. No explanations.
""").strip()


# ===================================================================
# Helpers
# ===================================================================

def clamp_score(v):
    """Clamp a value to strictly between 0 and 1 (exclusive)."""
    if v is None:
        return 0.5
    return round(min(0.99, max(0.01, float(v))), 4)


def build_prompt(obs, history):
    state_info = textwrap.dedent(f"""
    === CIVILIZATION STATUS ===
    Turn: {obs.turn}/{obs.max_turns} | Era: {obs.era.upper()} | Score: {obs.total_reward:.2f}
    Population: {obs.population} | Food: {obs.food} | Energy: {obs.energy}
    Morale: {obs.morale} | Technology: {obs.technology_level} | Danger: {obs.danger_level}
    Workers: {obs.available_workers} | Threats: {obs.active_threats}
    """).strip()

    msgs_text = ""
    if obs.messages:
        lines = []
        for m in obs.messages:
            lines.append(f"[{m['urgency'].upper()}] ID: {m['id']} | Source: {m['source']} - {m['sender_name']}")
            lines.append(f"  Subject: {m['subject']}")
            lines.append(f"  {m['body'][:200]}")
            lines.append("")
        msgs_text = "\n".join(lines)
    else:
        msgs_text = "No messages this turn."

    recent = "\n".join(history[-3:]) if history else "None"

    return f"""{state_info}

=== INCOMING MESSAGES ===
{msgs_text}

Recent decisions:
{recent}

What action do you take? Respond with JSON only."""


def parse_action(text):
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        text = text[brace_start:brace_end + 1]

    try:
        data = json.loads(text)
        return CivAction(
            action_type=data.get("action_type", "ignore"),
            target_message_id=data.get("target_message_id"),
            reason=data.get("reason"),
        )
    except (json.JSONDecodeError, Exception):
        tl = text.lower()
        if "defend" in tl:
            return CivAction(action_type="defend")
        elif "emergency" in tl:
            return CivAction(action_type="emergency_response")
        elif "food" in tl:
            return CivAction(action_type="allocate_food")
        elif "research" in tl:
            return CivAction(action_type="approve_research")
        elif "calm" in tl:
            return CivAction(action_type="calm_citizens")
        return CivAction(action_type="ignore")


# ===================================================================
# Run Task
# ===================================================================

def run_task(task_id, task_name, grader_fn):
    """Run a single task episode, emitting [START]/[STEP]/[END] to stdout."""

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

    env = CivilCommandCenter()
    obs = env.reset(seed=SEED, task_id=task_id)

    history = []
    rewards = []
    step_count = 0
    last_error = None
    success = False
    grade = 0.01

    try:
        while not obs.done:
            prompt = build_prompt(obs, history)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            response_text = ""
            for attempt in range(MAX_RETRIES + 1):
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    response_text = completion.choices[0].message.content or ""
                    last_error = None
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if attempt < MAX_RETRIES:
                        time.sleep(1)
                    else:
                        response_text = '{"action_type": "ignore"}'

            action = parse_action(response_text)
            obs = env.step(action)

            step_count += 1
            step_reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(step_reward)

            # Clamp the displayed reward to (0, 1) exclusive
            display_reward = clamp_score(step_reward)
            error_str = last_error if last_error else "null"
            done_str = "true" if obs.done else "false"
            print(f"[STEP] step={step_count} action={action.action_type} reward={display_reward:.4f} done={done_str} error={error_str}")

            history.append(f"Turn {obs.turn}: {action.action_type} -> {step_reward:+.2f}")
            last_error = None

        # Grade the episode
        summary = env.get_episode_summary()
        grade = clamp_score(grader_fn(summary))
        success = not summary.get("collapse", False)

    except Exception as exc:
        success = False
        last_error = str(exc)
        grade = 0.01
        summary = {}

    # Clamp all rewards for output
    clamped = [clamp_score(r) for r in rewards]
    rewards_str = ",".join(f"{r:.4f}" for r in clamped)
    success_str = "true" if success else "false"
    print(f"[END] task={task_id} score={grade:.4f} success={success_str} steps={step_count} rewards={rewards_str}")

    return {
        "task_id": task_id,
        "task_name": task_name,
        "grade": grade,
        "total_reward": sum(rewards),
        "steps": step_count,
        "summary": summary,
    }


# ===================================================================
# Main
# ===================================================================

def main():
    results = []

    r1 = run_task("task_easy", "Survival - Keep the Village Alive", grade_easy)
    results.append(r1)

    r2 = run_task("task_medium", "Growth - Build a Thriving Settlement", grade_medium)
    results.append(r2)

    r3 = run_task("task_hard", "Era Advancement - Rise of a Civilization", grade_hard)
    results.append(r3)

    # Save detailed results to file
    total = sum(r["grade"] for r in results)
    avg = total / len(results)

    output = {
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "seed": SEED,
        "results": [
            {"task_id": r["task_id"], "grade": r["grade"], "total_reward": r["total_reward"], "steps": r["steps"]}
            for r in results
        ],
        "average_grade": avg,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)

    return avg


if __name__ == "__main__":
    main()
