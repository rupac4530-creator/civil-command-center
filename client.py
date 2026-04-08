"""
Civil Command Center — OpenEnv Client
=======================================
HTTP client for interacting with the environment.
"""

import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from models import CivAction, CivObservation, CivState


@dataclass
class StepResult:
    observation: CivObservation
    reward: Optional[float]
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class CivilCommandEnv:
    """
    HTTP client for Civil Command Center.

    Usage:
        env = CivilCommandEnv("http://localhost:8000")
        r = env.reset(task_id="task_easy")
        print(r.observation.messages)
        r = env.step(CivAction(action_type="defend"))
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._session_id: Optional[str] = None

    def reset(self, task_id: str = "task_easy", seed: Optional[int] = None) -> StepResult:
        payload = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=30)
        r.raise_for_status()
        d = r.json()
        self._session_id = d["session_id"]
        obs = CivObservation(**d["observation"])
        return StepResult(observation=obs, reward=d.get("reward"), done=d.get("done", False))

    def step(self, action: CivAction) -> StepResult:
        if not self._session_id:
            raise RuntimeError("No active session. Call reset() first.")
        payload = {"action_type": action.action_type}
        if action.target_message_id:
            payload["target_message_id"] = action.target_message_id
        if action.reason:
            payload["reason"] = action.reason
        r = requests.post(f"{self.base_url}/step/{self._session_id}", json=payload, timeout=30)
        r.raise_for_status()
        d = r.json()
        obs = CivObservation(**d["observation"])
        info = {}
        if d.get("episode_summary"):
            info["episode_summary"] = d["episode_summary"]
        if obs.done:
            self._session_id = None
        return StepResult(observation=obs, reward=d.get("reward"), done=d.get("done", False), info=info)

    def state(self) -> CivState:
        if not self._session_id:
            raise RuntimeError("No active session.")
        r = requests.get(f"{self.base_url}/state/{self._session_id}", timeout=30)
        r.raise_for_status()
        return CivState(**r.json())

    def get_tasks(self) -> List[Dict]:
        r = requests.get(f"{self.base_url}/tasks", timeout=30)
        r.raise_for_status()
        return r.json()["tasks"]

    def health(self) -> Dict:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def close(self):
        self._session_id = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
