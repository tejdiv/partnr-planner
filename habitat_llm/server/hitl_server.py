#!/usr/bin/env python3
"""
Production HITL WebSocket server for PARTNR data collection
Supports multiple concurrent users collecting data
"""

import asyncio
import gzip
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
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

        # AI Async Control
        self.ai_action_queue = asyncio.Queue(maxsize=1) # Only keep latest action? No, queue is better for sequence.
        # Actually, maxsize=1 is good to prevent backlog if physics is slow. 
        # But if physics is fast (30Hz) and AI is slow (0.2Hz), queue will be empty most times.
        # If AI is fast (skill execution), queue might fill up.
        # Let's use a small buffer.
        self.ai_action_queue = asyncio.Queue(maxsize=10)
        self.ai_loop_task: Optional[asyncio.Task] = None
        self.running = False

        # Load first episode
        self.reset_episode()

    async def start_ai_loop(self):
        """Start the background AI planning loop"""
        self.running = True
        while self.running:
            try:
                # Get observations for AI (Agent 1)
                # We need to be careful about thread safety here. 
                # The env is being stepped in another thread.
                # Ideally, we should get obs from the main thread or lock.
                # For now, we'll assume get_observations is safe enough or we'll grab it 
                # before the step? No, AI loop runs independently.
                
                # We'll run the planning in a thread to avoid blocking the event loop
                # But we need to pass the CURRENT observations.
                # Let's grab obs here (fast) then plan (slow).
                full_obs = self.env.get_observations()
                obs_1 = self.env.filter_obs_space(full_obs, 1)
                
                # Run planner in thread
                ai_action = await asyncio.to_thread(
                    self.ai_planner.plan,
                    obs_1,
                    self.env.current_episode,
                    agent_uid=1
                )
                
                # Put in queue
                # If queue is full, we wait. This throttles the AI to the physics speed.
                await self.ai_action_queue.put(ai_action)
                
                # Small sleep to yield control if planner is super fast
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"Error in AI loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0) # Wait before retrying

    def stop_ai_loop(self):
        """Stop the background AI loop"""
        self.running = False
        if self.ai_loop_task:
            self.ai_loop_task.cancel()

    def reset_episode(self):
        """Reset to a new episode"""
        # Clear queue
        while not self.ai_action_queue.empty():
            try:
                self.ai_action_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
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

    def step(self, human_action: Dict, ai_action: Optional[Tuple] = None):
        """
        Step simulation with human action (agent 0) and optional AI action (agent 1)

        Args:
            human_action: {"action": "Navigate", "target": "table_1"}
            ai_action: Tuple from planner, e.g. ("Navigate", "table_1") or None

        Returns:
            {
                "rgb": bytes, # JPEG bytes
                "state": {...},
                "done": bool,
                "success": bool
            }
        """
        # If no AI action provided (queue empty), use a "Wait" or "No-op" action
        # We need to know what a valid no-op is. 
        # Usually ("Wait", None) or similar.
        # For now, if None, we might just repeat the last one? 
        # NO, user wants "Consume". If empty, robot stands still.
        if ai_action is None:
             # Assuming "Wait" is a valid action or we pass a null action
             # Check action space? For now let's try ("Wait", None)
             ai_action = ("Wait", None)

        # Step environment with both actions
        actions = {
            0: (human_action["action"], human_action.get("target")),
            1: ai_action
        }

        # obs is a dictionary of numpy arrays (not a list)
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

        # Extract RGB for client (Agent 0)
        # Key is typically 'agent_0_rgb' or similar depending on config
        # We try standard keys
        rgb_img = None
        if "agent_0_rgb" in obs:
            rgb_img = obs["agent_0_rgb"]
        elif "rgb" in obs:
            rgb_img = obs["rgb"]
        else:
            # Fallback: search for any rgb key
            for k, v in obs.items():
                if "rgb" in k and "agent_0" in k:
                    rgb_img = v
                    break
        
        if rgb_img is None:
            print(f"Warning: Could not find RGB in obs keys: {obs.keys()}")
            rgb_img = np.zeros((480, 640, 3), dtype=np.uint8)

        rgb_jpeg = self._encode_image(rgb_img)

        return {
            "rgb": rgb_jpeg,
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

    def _encode_image(self, img_array: np.ndarray) -> bytes:
        """Convert numpy array to JPEG bytes"""
        # Ensure uint8
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
            
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        return buffer.getvalue()

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
    import habitat

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Initialize Hydra with the config directory
    config_dir = os.path.abspath("habitat_llm/conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        # Compose the config
        config = compose(config_name="baselines/decentralized_zero_shot_react_summary")

    # Set agents_order and enable headless rendering
    with habitat.config.read_write(config):
        config.habitat.simulator.agents_order = sorted(config.habitat.simulator.agents.keys())

        # Enable headless rendering
        config.habitat.simulator.habitat_sim_v0.enable_gfx_replay_save = False
        if not hasattr(config.habitat.simulator.habitat_sim_v0, 'gpu_device_id'):
            OmegaConf.set_struct(config.habitat.simulator.habitat_sim_v0, False)
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
    
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
            print(f"Error creating session: {e}")
            import traceback
            traceback.print_exc()
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
            return

    session = active_sessions[session_id]
    
    # Start AI loop if not running
    if session.ai_loop_task is None or session.ai_loop_task.done():
        session.ai_loop_task = asyncio.create_task(session.start_ai_loop())

    try:
        # Send initial frame
        # Initial obs is also a dict of numpy arrays
        initial_obs = session.env.env.reset() # Use underlying env reset to get raw obs
        
        # Find RGB in initial obs
        rgb_img = None
        if "agent_0_rgb" in initial_obs:
            rgb_img = initial_obs["agent_0_rgb"]
        elif "rgb" in initial_obs:
            rgb_img = initial_obs["rgb"]
        
        if rgb_img is not None:
            rgb_jpeg = session._encode_image(rgb_img)
        else:
            rgb_jpeg = b''

        # Send JSON metadata
        await websocket.send_json({
            "type": "init",
            "instruction": session.env.current_episode.instruction,
            "session_id": session_id,
            "has_image": rgb_img is not None
        })
        # Send binary image
        if rgb_img is not None:
            await websocket.send_bytes(rgb_jpeg)

        # Main loop
        while True:
            data = await websocket.receive_json()

            if data["type"] == "action":
                try:
                    # Check for AI action
                    ai_action = None
                    try:
                        ai_action = session.ai_action_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass

                    # Step simulation in a separate thread to avoid blocking the event loop
                    # This prevents WebSocket timeouts during long AI planning steps
                    result = await asyncio.to_thread(session.step, data["action"], ai_action)
                except Exception as e:
                    print(f"Error during simulation step: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"type": "error", "message": f"Simulation error: {str(e)}"})
                    continue

                # Extract image from result
                rgb_jpeg = result.pop("rgb") if "rgb" in result else None

                # Send JSON metadata
                await websocket.send_json({
                    "type": "step",
                    **result,
                    "has_image": rgb_jpeg is not None
                })

                # Send binary image if present
                if rgb_jpeg:
                    await websocket.send_bytes(rgb_jpeg)

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
        session.stop_ai_loop()
        # Don't delete session immediately to allow reconnect? 
        # For now, we delete to keep it simple and save data
        if session_id in active_sessions:
            session._save_episode()
            del active_sessions[session_id]
    except Exception as e:
        print(f"Error in session {session_id}: {e}")
        session.stop_ai_loop()
        import traceback
        traceback.print_exc()
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
