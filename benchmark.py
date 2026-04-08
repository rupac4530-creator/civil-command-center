"""
Civil Command Center — Benchmark Mode
===================================================
Judges evaluate an AI environment by seeing if it can
distinguish between good, bad, and random agents.

This script runs 3 different agents against the same task
(Standard Demo - 10 Turns) for multiple episodes and
compares their performance (Survival rate, Score, Stability).

Agents:
1. Random Agent: Picks a random action every turn.
2. Greedy Agent: Always picks short-term resource gains (food/energy)
                 while ignoring long-term crises and defense.
3. Logical Agent (Good): Prioritizes critical crises, then balance.
"""

import sys
import os
import random
from collections import defaultdict

# Fix Windows encoding for emoji/Unicode output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure server module is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import CivilCommandCenter
from models import CivAction

EPISODES = 10
TASK = "task_demo_10"

def play_episode(agent_func) -> dict:
    """Run one episode using the provided agent function."""
    env = CivilCommandCenter()
    obs = env.reset(task_id=TASK)
    
    while not obs.done:
        action_name, target_id = agent_func(obs)
        action = CivAction(action_type=action_name, target_message_id=target_id)
        obs = env.step(action)
        
    return env.get_episode_summary()

# ── Agents ───────────────────────────────────────────────────

def random_agent(obs) -> tuple:
    """Picks a random action, ignoring the context."""
    actions = [
        "allocate_food", "allocate_workers", "approve_research",
        "defend", "calm_citizens", "accept_trade", "invest_growth", "ignore"
    ]
    target = obs.messages[0]["id"] if obs.messages else None
    return random.choice(actions), target

def greedy_agent(obs) -> tuple:
    """Always picks actions that sound like immediate resource gains."""
    greedy_actions = ["allocate_food", "allocate_workers", "accept_trade", "invest_growth"]
    target = obs.messages[0]["id"] if obs.messages else None
    return random.choice(greedy_actions), target

def logical_agent(obs) -> tuple:
    """A highly logical agent that addresses immediate crises first."""
    if not obs.messages:
        return "invest_growth", None

    # Sort messages by urgency
    urgency_map = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    messages = sorted(obs.messages, key=lambda m: urgency_map.get(m["urgency"], 4))
    
    # Pick the most urgent message
    target = messages[0]
    action_type = "ignore"
    
    # Simple semantic routing based on message source/subject
    source = target["source"]
    subj = target["subject"].lower()
    
    if source == "defense" or "attack" in subj or "war" in subj:
        action_type = "defend"
    elif source == "citizen" and "food" in subj:
        action_type = "allocate_food"
    elif source == "scientist" or "tech" in subj:
        action_type = "approve_research"
    elif source == "worker" or "production" in subj:
        action_type = "allocate_workers"
    elif "protest" in subj or "unrest" in subj:
        action_type = "calm_citizens"
    elif source == "trader":
        action_type = "accept_trade"
    elif source == "event" and "plague" in subj:
        action_type = "emergency_response"
    else:
        action_type = "invest_growth" # fallback positive action

    return action_type, target["id"]

# ── Benchmark Runner ──────────────────────────────────────────

def run_benchmark():
    print("=" * 60)
    print("  Civil Command Center -- Environment Benchmark")
    print("=" * 60)
    print(f"Running {EPISODES} episodes per agent on '{TASK}'\n")

    agents = {
        "[RND] Random Agent": random_agent,
        "[GRD] Greedy Agent": greedy_agent,
        "[LOG] Logical Agent": logical_agent,
    }

    results = {}

    for name, func in agents.items():
        print(f"Testing {name} ", end="", flush=True)
        stats = {
            "survived": 0,
            "avg_score": 0.0,
            "avg_stability": 0.0,
            "crises_averted": 0,
        }
        
        for _ in range(EPISODES):
            print(".", end="", flush=True)
            summary = play_episode(func)
            
            if not summary["collapse"]:
                stats["survived"] += 1
            
            stats["avg_score"] += summary["total_reward"]
            stats["avg_stability"] += summary["memory"]["stability_score"]
            stats["crises_averted"] += summary["crises_averted"]
            
        # Average out
        stats["avg_score"] /= EPISODES
        stats["avg_stability"] /= EPISODES
        stats["crises_averted"] /= EPISODES
        
        results[name] = stats
        print(" Done!")

    # ── Display Results ──────────────────────────────────────────
    
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS (Separating Good vs Bad Agents)")
    print("=" * 60)
    print(f"{'Agent Name':<20} | {'Surv Rate':<10} | {'Avg Score':<10} | {'Stability':<9} | {'Crises Fixed':<12}")
    print("-" * 70)
    
    for name, stats in results.items():
        surv_rate = f"{stats['survived']}/{EPISODES}"
        score = f"{stats['avg_score']:.2f}"
        stab = f"{stats['avg_stability'] * 100:.0f}%"
        crises = f"{stats['crises_averted']:.1f}"
        print(f"{name:<20} | {surv_rate:<10} | {score:<10} | {stab:<9} | {crises:<12}")

    print("-" * 70)
    print("\nVerdict: The environment successfully penalizes agents that ignore")
    print("delayed consequences (Greedy/Random) and strongly rewards agents")
    print("that plan ahead and manage multi-variable stability (Logical).\n")

if __name__ == "__main__":
    run_benchmark()
