#!/usr/bin/env python3
"""
Production HITL WebSocket server for PARTNR data collection
Supports multiple concurrent users collecting data
"""

import asyncio
import base64
import gzip
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import hydra
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from habitat_llm.agent.env import EnvironmentInterface, register_actions, register_measures, register_sensors
from habitat_llm.agent.env.dataset import CollaborationDatasetV0

app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://partnr-hitl.vercel.app", "http://localhost:3000", "*"],  # Vercel domain + local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active sessions
active_sessions: Dict[str, "SimSession"] = {}

class SimSession:
    """Manages one PARTNR simulation session"""

    def __init__(self, session_id: str, config: DictConfig):
        self.session_id = session_id
        self.config = config

        # Initialize environment
        register_sensors(config)
        register_actions(config)
        register_measures(config)

        self.dataset = CollaborationDatasetV0(config.habitat.dataset)
        self.env = EnvironmentInterface(config, dataset=self.dataset)

        # Initialize AI planner for agent 1
        self.ai_planner = hydra.utils.instantiate(
            config.evaluation.agents.agent_1.planner
        )

        # Recording
        self.recording_dir = Path(f"data/hitl_sessions/{session_id}")
        self.recording_dir.mkdir(parents=True, exist_ok=True)

        # Episode tracking
        self.current_episode_idx = 0
        self.episode_data = {
            "frames": [],
            "session": {
                "session_id": session_id,
                "start_time": datetime.now().isoformat(),
                "config": OmegaConf.to_container(config, resolve=True)
            }
        }

        # Load first episode
        self.reset_episode()

    def reset_episode(self):
        """Reset to a new episode"""
        obs = self.env.reset(self.current_episode_idx)

        # Save previous episode if any data
        if len(self.episode_data["frames"]) > 0:
            self._save_episode()

        # Reset for new episode
        self.episode_data = {
            "episode": {
                "episode_id": self.env.current_episode.episode_id,
                "scene_id": self.env.current_episode.scene_id,
                "instruction": self.env.current_episode.instruction,
                "start_position": list(self.env.current_episode.start_position),
            },
            "frames": [],
            "session": self.episode_data.get("session", {})
        }

        return obs

    def step(self, human_action: Dict):
        """
        Step simulation with human action (agent 0) and AI action (agent 1)

        Args:
            human_action: {"action": "Navigate", "target": "table_1"}

        Returns:
            {
                "rgb": base64_image,
                "state": {...},
                "done": bool,
                "success": bool
            }
        """
        # Get AI action for agent 1
        obs_1 = self.env.get_observations(1)
        ai_action = self.ai_planner.plan(
            obs_1,
            self.env.current_episode,
            agent_uid=1
        )

        # Step environment with both actions
        actions = {
            0: (human_action["action"], human_action.get("target")),
            1: ai_action
        }

        obs, reward, done, info = self.env.step(actions)

        # Record frame
        frame_data = {
            "t": datetime.now().timestamp(),
            "users": [
                {
                    "uid": 0,
                    "events": [{"action": human_action["action"], "target": human_action.get("target")}]
                },
                {
                    "uid": 1,
                    "events": [{"action": ai_action[0], "target": ai_action[1]}]
                }
            ],
            "info": {
                "task_percent_complete": info.get("task_percent_complete", 0.0),
                "success": info.get("success", False)
            }
        }
        self.episode_data["frames"].append(frame_data)

        # Encode RGB for client
        rgb_img = obs[0].get("rgb", np.zeros((480, 640, 3), dtype=np.uint8))
        rgb_base64 = self._encode_image(rgb_img)

        return {
            "rgb": rgb_base64,
            "state": {
                "task_progress": info.get("task_percent_complete", 0.0),
                "instruction": self.env.current_episode.instruction,
                "agent_0_position": info.get("agent_0_position", [0, 0, 0]),
                "agent_1_position": info.get("agent_1_position", [0, 0, 0]),
                "ai_action": f"{ai_action[0]}[{ai_action[1]}]" if ai_action[1] else ai_action[0]
            },
            "done": done,
            "success": info.get("success", False)
        }

    def _encode_image(self, img_array: np.ndarray) -> str:
        """Convert numpy array to base64 JPEG"""
        img = Image.fromarray(img_array.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        return base64.b64encode(buffer.getvalue()).decode()

    def _save_episode(self):
        """Save episode data to compressed JSON"""
        episode_id = self.episode_data["episode"]["episode_id"]
        filepath = self.recording_dir / f"{episode_id}.json.gz"

        with gzip.open(filepath, "wt") as f:
            json.dump(self.episode_data, f, indent=2)

        print(f"Saved episode {episode_id} to {filepath}")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time human-AI collaboration"""
    await websocket.accept()

    # Load config using Hydra composition
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with the config directory
    config_dir = os.path.abspath("habitat_llm/conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        # Compose the config
        config = compose(config_name="baselines/decentralized_zero_shot_react_summary")
    
    # Override with custom model path if provided
    model_path = os.getenv("PARTNR_MODEL_PATH", "models/iteration_0")
    OmegaConf.set_struct(config, False)  # Allow setting new keys
    config.evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine = model_path
    config.evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine = model_path

    # Create session
    if session_id not in active_sessions:
        try:
            active_sessions[session_id] = SimSession(session_id, config)
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
            return

    session = active_sessions[session_id]

    try:
        # Send initial frame
        initial_obs = session.env.get_observations(0)
        rgb_base64 = session._encode_image(initial_obs.get("rgb"))

        await websocket.send_json({
            "type": "init",
            "rgb": rgb_base64,
            "instruction": session.env.current_episode.instruction,
            "session_id": session_id
        })

        # Main loop
        while True:
            data = await websocket.receive_json()

            if data["type"] == "action":
                # Step simulation
                result = session.step(data["action"])

                # Send result
                await websocket.send_json({
                    "type": "step",
                    **result
                })

                # If episode done, move to next
                if result["done"]:
                    session.current_episode_idx += 1

                    # Check if more episodes available
                    if session.current_episode_idx < len(session.dataset.episodes):
                        session.reset_episode()
                        await websocket.send_json({
                            "type": "episode_complete",
                            "success": result["success"],
                            "next_episode": True
                        })
                    else:
                        await websocket.send_json({
                            "type": "episode_complete",
                            "success": result["success"],
                            "next_episode": False,
                            "message": "All episodes completed!"
                        })

            elif data["type"] == "skip":
                # Skip to next episode
                session.current_episode_idx += 1
                if session.current_episode_idx < len(session.dataset.episodes):
                    session.reset_episode()
                    await websocket.send_json({"type": "skipped"})

    except WebSocketDisconnect:
        print(f"Client {session_id} disconnected")
        session._save_episode()  # Save final episode
        del active_sessions[session_id]
    except Exception as e:
        print(f"Error in session {session_id}: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "model": os.getenv("PARTNR_MODEL_PATH", "models/iteration_0")
    }


@app.get("/")
def read_root():
    return FileResponse("habitat_llm/server/static/index.html")


# Serve static files
# Note: For Vercel deployment, static files are served by Vercel, but we keep this for local testing
if os.path.exists("habitat_llm/server/static"):
    app.mount("/static", StaticFiles(directory="habitat_llm/server/static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
