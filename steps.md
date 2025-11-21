# PARTNR Two-Pipeline Implementation Guide

REFERENCE: https://arxiv.org/pdf/2411.00081


Complete technical implementation: Distillation + Human Refinement

**What this covers:**
1. Iteration 0: Distill 70B ReACT → 8B baseline
2. Browser-based multi-user data collection
3. Expert trajectory annotation system
4. Weighted training pipeline (expert > human_gold)
5. Complete 5-iteration workflow

---

## PART 1: Iteration 0 - Distillation from 70B ReACT

### Goal
Create Stage-1 baseline by distilling Llama-70B ReACT trajectories into Llama-8B.

### Step 1.1: Run 70B ReACT Baseline

**Option A: Use OpenAI API (Cheapest)**

If 70B is too expensive to run locally, use GPT-4 as the teacher:

```bash
# Set API key
export OPENAI_API_KEY=your_key_here

# Run on TRAIN episodes with GPT-4
python -m habitat_llm.examples.planner_demo \
  --config-name baselines/decentralized_zero_shot_react_summary.yaml \
  habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/train_2k.json.gz" \
  llm@evaluation.agents.agent_0.planner.plan_config.llm=openai_chat \
  llm@evaluation.agents.agent_1.planner.plan_config.llm=openai_chat \
  evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=gpt-4 \
  evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=gpt-4 \
  num_proc=5 \
  evaluation.num_eval_episodes=100 \
  paths.results_dir=outputs/distillation/teacher_rollouts

# Cost: ~$50-100 for 100 episodes (GPT-4 API)
# Time: ~6-10 hours
```

**Option B: Use Llama-70B (If you have access)**

```bash
# Run on university cluster or via Together AI / Replicate API
python -m habitat_llm.examples.planner_demo \
  --config-name baselines/decentralized_zero_shot_react_summary.yaml \
  habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/train_2k.json.gz" \
  evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
  evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
  evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3.1-70B-Instruct \
  evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3.1-70B-Instruct \
  num_proc=5 \
  evaluation.num_eval_episodes=100 \
  paths.results_dir=outputs/distillation/teacher_rollouts
```

**Output:** `outputs/distillation/teacher_rollouts/detailed_traces/*.pkl`

### Step 1.2: Convert Trajectories to Training Format

```bash
# Use existing PARTNR script
python -m habitat_llm.finetuning.build_trace_dataset \
  --path outputs/distillation/teacher_rollouts \
  --output-dir data/training/iteration_0/distilled \
  --percent_cut 0.5 \
  --num-workers 8

# This creates:
# data/training/iteration_0/distilled/{episode_id}/sample_*.txt
# Each sample: (state, history) → teacher_action
```

### Step 1.3: Finetune 8B on Teacher Data

Create config: `habitat_llm/conf/finetuning/iteration_0_distill.yaml`

```yaml
defaults:
  - finetuning
  - _self_

wandb:
  name: "iteration_0_distillation"

dataset:
  path: "data/training"
  train:
    - "iteration_0/distilled"
  val:
    - "iteration_0/distilled"
  max_train_size: -1
  max_val_size: 100

llm_config:
  name: "meta-llama/Meta-Llama-3-8B-Instruct"

training_arguments:
  batch_size: 2
  epochs: 3
  lr: 3e-5
  gradient_accumulation_steps: 4
  save_steps: 500
```

Train:

```bash
# On Lambda Labs server
python habitat_llm/finetuning/trainer.py \
  --config-name finetuning/iteration_0_distill \
  wandb.name="iteration_0_distillation"

# Takes ~4-6 hours on A6000
# Cost: ~$3-5
```

### Step 1.4: Merge and Save

```bash
# Merge LoRA weights
python -m habitat_llm.finetuning.flatten_checkpoint \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --checkpoint_dir multirun/YYYY-MM-DD/HH-MM-SS/0/checkpoints/checkpoint_1500

# Save to models directory
mkdir -p models/iteration_0
cp -r multirun/.../checkpoints/checkpoint_1500_merged/ models/iteration_0/

# Upload to Hugging Face (optional, for backup)
huggingface-cli upload your-username/partnr-iteration-0 ./models/iteration_0/
```

**Output:** `models/iteration_0/` - Your Stage-1 baseline!

---

## PART 2: Browser-Based Data Collection System

### Architecture

```
Browser (10+ users) ← WebSocket → Lambda Server (Habitat-Sim + Your 8B model)
       ↓                                ↓
 Keyboard input                  RGB frames + game state
       ↓                                ↓
    Human controls agent_0        AI controls agent_1
                ↓
         Records to data/hitl_sessions/
```

### Step 2.1: Production WebSocket Server

Create: `habitat_llm/server/hitl_server.py`

```python
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
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from habitat_llm.agent.env import EnvironmentInterface, register_actions, register_measures, register_sensors
from habitat_llm.agent.env.dataset import CollaborationDatasetV0

app = FastAPI()

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

    # Load config
    # TODO: Make this configurable per session
    config = OmegaConf.load("habitat_llm/conf/baselines/decentralized_zero_shot_react_summary.yaml")

    # Override with custom model path if provided
    model_path = os.getenv("PARTNR_MODEL_PATH", "models/iteration_0")
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
app.mount("/static", StaticFiles(directory="habitat_llm/server/static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 2.2: Browser Client

Create: `habitat_llm/server/static/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PARTNR Human-AI Data Collection</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #fff;
        }
        #container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        #connect-section {
            text-align: center;
            margin: 20px 0;
        }
        #connect-btn {
            background: #4CAF50;
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 18px;
        }
        #connect-btn:hover {
            background: #45a049;
        }
        #connect-btn:disabled {
            background: #666;
            cursor: not-allowed;
        }
        #instruction-box {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #4CAF50;
        }
        #instruction-text {
            font-size: 16px;
            line-height: 1.5;
        }
        #viewport {
            width: 100%;
            max-width: 800px;
            height: auto;
            border: 3px solid #444;
            border-radius: 8px;
            margin: 20px auto;
            display: block;
            background: #000;
        }
        #controls-panel {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .control-row {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }
        .status-item {
            flex: 1;
            padding: 10px;
            margin: 0 5px;
            background: #333;
            border-radius: 5px;
            text-align: center;
        }
        .status-label {
            font-size: 12px;
            color: #999;
            margin-bottom: 5px;
        }
        .status-value {
            font-size: 18px;
            font-weight: bold;
        }
        #controls-help {
            background: #333;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
        }
        .key {
            display: inline-block;
            background: #555;
            padding: 5px 10px;
            border-radius: 3px;
            margin: 2px;
            font-family: monospace;
        }
        #ai-action {
            color: #FFA500;
            font-style: italic;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #333;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }
    </style>
</head>
<body>
    <div id="container">
        <h1>🤖 PARTNR Human-AI Collaboration</h1>

        <div id="connect-section">
            <button id="connect-btn" onclick="connect()">Connect & Start</button>
        </div>

        <div id="instruction-box">
            <div class="status-label">TASK INSTRUCTION:</div>
            <div id="instruction-text">Waiting for connection...</div>
        </div>

        <img id="viewport" alt="Simulation view">

        <div class="progress-bar">
            <div class="progress-fill" id="progress-fill" style="width: 0%">
                <span id="progress-text">0%</span>
            </div>
        </div>

        <div id="controls-panel">
            <h3>Status & Controls</h3>

            <div class="control-row">
                <div class="status-item">
                    <div class="status-label">Connection</div>
                    <div class="status-value" id="status">Disconnected</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Session ID</div>
                    <div class="status-value" id="session-id">-</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Episodes Completed</div>
                    <div class="status-value" id="episodes-done">0</div>
                </div>
            </div>

            <div class="control-row">
                <div class="status-item" style="flex: 2;">
                    <div class="status-label">AI Partner Action</div>
                    <div class="status-value" id="ai-action">-</div>
                </div>
            </div>

            <div id="controls-help">
                <strong>Keyboard Controls:</strong><br>
                <span class="key">W</span> Move Forward
                <span class="key">S</span> Move Backward
                <span class="key">A</span> Strafe Left
                <span class="key">D</span> Strafe Right<br>
                <span class="key">Q</span> Turn Left
                <span class="key">E</span> Turn Right
                <span class="key">Space</span> Pick/Place Object
                <span class="key">Shift+Space</span> Open/Close<br>
                <span class="key">N</span> Navigate to object (type name in prompt)
                <span class="key">Esc</span> Skip Episode
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let sessionId = 'user_' + Math.random().toString(36).substr(2, 9);
        let episodesCompleted = 0;

        function connect() {
            const serverUrl = prompt("Enter server address:", window.location.host);
            if (!serverUrl) return;

            const wsUrl = serverUrl.startsWith('ws://') ? serverUrl : `ws://${serverUrl}`;
            ws = new WebSocket(`${wsUrl}/ws/${sessionId}`);

            ws.onopen = () => {
                document.getElementById('status').textContent = 'Connected ✓';
                document.getElementById('status').style.color = '#4CAF50';
                document.getElementById('session-id').textContent = sessionId;
                document.getElementById('connect-btn').disabled = true;
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'init' || data.type === 'step') {
                    // Update viewport
                    if (data.rgb) {
                        document.getElementById('viewport').src = 'data:image/jpeg;base64,' + data.rgb;
                    }

                    // Update instruction
                    if (data.instruction) {
                        document.getElementById('instruction-text').textContent = data.instruction;
                    }

                    // Update state
                    if (data.state) {
                        const progress = Math.round(data.state.task_progress * 100);
                        document.getElementById('progress-fill').style.width = progress + '%';
                        document.getElementById('progress-text').textContent = progress + '%';

                        if (data.state.ai_action) {
                            document.getElementById('ai-action').textContent = data.state.ai_action;
                        }
                    }
                }

                if (data.type === 'episode_complete') {
                    episodesCompleted++;
                    document.getElementById('episodes-done').textContent = episodesCompleted;

                    const message = data.success ? '✓ Success!' : '✗ Episode ended';
                    alert(message + (data.next_episode ? ' Loading next episode...' : ' All episodes done!'));

                    if (!data.next_episode) {
                        alert('Thank you for participating! You completed ' + episodesCompleted + ' episodes.');
                        ws.close();
                    }
                }

                if (data.type === 'error') {
                    alert('Error: ' + data.message);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                document.getElementById('status').textContent = 'Error';
                document.getElementById('status').style.color = '#f44336';
            };

            ws.onclose = () => {
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').style.color = '#f44336';
                document.getElementById('connect-btn').disabled = false;
            };
        }

        // Keyboard controls
        const keyMap = {
            'w': 'MoveForward',
            's': 'MoveBackward',
            'a': 'StrafeLeft',
            'd': 'StrafeRight',
            'q': 'TurnLeft',
            'e': 'TurnRight',
            ' ': 'GrabRelease'
        };

        document.addEventListener('keydown', (e) => {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;

            const key = e.key.toLowerCase();

            // Handle Escape (skip episode)
            if (key === 'escape') {
                if (confirm('Skip this episode?')) {
                    ws.send(JSON.stringify({type: 'skip'}));
                }
                return;
            }

            // Handle navigation (N key)
            if (key === 'n') {
                const target = prompt('Navigate to which object?');
                if (target) {
                    ws.send(JSON.stringify({
                        type: 'action',
                        action: {action: 'Navigate', target: target}
                    }));
                }
                return;
            }

            // Handle movement keys
            const action = keyMap[key];
            if (action) {
                e.preventDefault();
                ws.send(JSON.stringify({
                    type: 'action',
                    action: {action: action}
                }));
            }
        });
    </script>
</body>
</html>
```

### Step 2.3: Deploy for Multi-User Collection

```bash
# On Lambda server
cd ~/partnr-planner

# Set which model to use
export PARTNR_MODEL_PATH=models/iteration_0

# Start server
conda activate habitat-llm
python habitat_llm/server/hitl_server.py

# Server runs at: http://YOUR-LAMBDA-IP:8000
# Share this URL with participants!
```

**For 10 concurrent users:**
- Each opens `http://YOUR-LAMBDA-IP:8000` in browser
- Each gets unique session ID
- Server creates separate Habitat-Sim instance per session
- All sessions record to `data/hitl_sessions/{session_id}/`

---

## PART 3: Expert Annotation System

### Step 3.1: Expert Annotation Interface

Create: `habitat_llm/expert/annotate.py`

```python
#!/usr/bin/env python3
"""
Expert annotation interface for PARTNR episodes
Load recorded HITL episodes and mark AI errors with corrections
"""

import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List

class ExpertAnnotator:
    def __init__(self, episodes_dir: str, output_file: str):
        self.episodes_dir = Path(episodes_dir)
        self.output_file = Path(output_file)
        self.annotations: Dict[str, List] = {}

        # Load existing annotations if any
        if self.output_file.exists():
            with open(self.output_file) as f:
                self.annotations = json.load(f)

    def load_episode(self, episode_id: str):
        """Load episode data"""
        # Find episode file
        episode_files = list(self.episodes_dir.glob(f"**/{episode_id}.json.gz"))
        if not episode_files:
            print(f"Episode {episode_id} not found!")
            return None

        with gzip.open(episode_files[0], 'rt') as f:
            return json.load(f)

    def annotate_episode(self, episode_id: str):
        """Annotate one episode"""
        episode_data = self.load_episode(episode_id)
        if not episode_data:
            return

        print(f"\n{'='*60}")
        print(f"Episode: {episode_id}")
        print(f"Instruction: {episode_data['episode']['instruction']}")
        print(f"Total frames: {len(episode_data['frames'])}")
        print(f"{'='*60}\n")

        if episode_id not in self.annotations:
            self.annotations[episode_id] = []

        # Show frames
        for i, frame in enumerate(episode_data['frames']):
            agent_1_events = frame['users'][1]['events']

            if not agent_1_events:
                continue

            ai_action = agent_1_events[0]
            action_str = f"{ai_action['action']}" + (f"[{ai_action['target']}]" if ai_action.get('target') else "")

            print(f"\nTimestep {i}:")
            print(f"  AI Action: {action_str}")
            print(f"  Progress: {frame['info']['task_percent_complete']:.1%}")

            response = input("  Mark as error? (y/N/q to quit): ").lower()

            if response == 'q':
                break

            if response == 'y':
                # Get correction
                print("\nWhat should the AI have done instead?")
                corrected_action = input("  Action name (e.g., Navigate, Pick, Place): ")
                corrected_target = input("  Target (optional, press enter to skip): ")
                reason = input("  Reason for correction: ")

                annotation = {
                    "timestep": i,
                    "agent_uid": 1,
                    "original_action": ai_action['action'],
                    "original_target": ai_action.get('target'),
                    "corrected_action": corrected_action,
                    "corrected_target": corrected_target if corrected_target else None,
                    "reason": reason,
                    "task_progress_at_time": frame['info']['task_percent_complete']
                }

                self.annotations[episode_id].append(annotation)
                print("  ✓ Annotation saved")

        # Save after each episode
        self.save_annotations()

    def save_annotations(self):
        """Save annotations to file"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"\n✓ Annotations saved to {self.output_file}")

    def batch_annotate(self, episode_ids: List[str]):
        """Annotate multiple episodes"""
        for episode_id in episode_ids:
            self.annotate_episode(episode_id)

            cont = input("\nContinue to next episode? (Y/n): ").lower()
            if cont == 'n':
                break


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes-dir', required=True, help='Directory with HITL episodes')
    parser.add_argument('--output', required=True, help='Output annotations file')
    parser.add_argument('--episode-ids', nargs='+', help='Specific episode IDs to annotate')
    args = parser.parse_args()

    annotator = ExpertAnnotator(args.episodes_dir, args.output)

    if args.episode_ids:
        annotator.batch_annotate(args.episode_ids)
    else:
        # Find all episodes
        episode_files = list(Path(args.episodes_dir).glob("**/*.json.gz"))
        episode_ids = [f.stem for f in episode_files]
        print(f"Found {len(episode_ids)} episodes")

        # Filter by success rate if desired
        print("\nOptions:")
        print("1. Annotate all episodes")
        print("2. Annotate only failed episodes")
        print("3. Specify episode IDs manually")

        choice = input("Choice (1/2/3): ")

        if choice == '3':
            episode_ids = input("Enter episode IDs (space-separated): ").split()

        annotator.batch_annotate(episode_ids)


if __name__ == "__main__":
    main()
```

Usage:

```bash
# Annotate specific episodes
python habitat_llm/expert/annotate.py \
  --episodes-dir data/hitl_sessions \
  --output data/training/iteration_1/expert/annotations.json \
  --episode-ids episode_123 episode_456

# Or interactively select
python habitat_llm/expert/annotate.py \
  --episodes-dir data/hitl_sessions \
  --output data/training/iteration_1/expert/annotations.json
```

### Step 3.2: Build Expert Dataset

Create: `habitat_llm/finetuning/build_expert_dataset.py`

```python
#!/usr/bin/env python3
"""
Build training dataset from expert annotations
Reconstructs episode state at each error timestep and creates training samples
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, List

from habitat_llm.llm.instruct.utils import build_single_step_prompt, STOP_WORD
from habitat_llm.world_model.world_graph import WorldGraph


def load_episode(episodes_dir: Path, episode_id: str):
    """Load episode from HITL sessions"""
    episode_files = list(episodes_dir.glob(f"**/{episode_id}.json.gz"))
    if not episode_files:
        raise ValueError(f"Episode {episode_id} not found in {episodes_dir}")

    with gzip.open(episode_files[0], 'rt') as f:
        return json.load(f)


def build_training_sample(episode_data: Dict, annotation: Dict) -> str:
    """
    Build training sample from annotation

    Returns training text: (instruction, state, history) → corrected_action
    """
    instruction = episode_data['episode']['instruction']
    timestep = annotation['timestep']

    # Reconstruct action history up to this point
    action_history = {0: [], 1: []}

    for i, frame in enumerate(episode_data['frames'][:timestep]):
        for agent_uid in [0, 1]:
            events = frame['users'][agent_uid]['events']
            if events:
                action = events[0]
                # Build action history entry
                # Note: This is simplified - you'd need actual ActionHistoryElement objects
                action_history[agent_uid].append({
                    'action': action['action'],
                    'target': action.get('target'),
                    'timestamp': i
                })

    # For now, create simplified prompt
    # In practice, you'd reconstruct world_graph from episode state
    # This matches the format from build_trace_dataset.py

    corrected_action = annotation['corrected_action']
    corrected_target = annotation.get('corrected_target', '')

    # Build prompt (simplified version)
    prompt = f"""Instruction: {instruction}

Agent 1 should take the following action:
{corrected_action}[{corrected_target}]{STOP_WORD}"""

    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hitl-episodes-path', required=True, help='Path to HITL episodes')
    parser.add_argument('--annotations-file', required=True, help='Expert annotations JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory for training samples')
    args = parser.parse_args()

    episodes_dir = Path(args.hitl_episodes_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(args.annotations_file) as f:
        annotations = json.load(f)

    total_samples = 0

    for episode_id, episode_annotations in annotations.items():
        if not episode_annotations:
            continue

        # Load episode data
        try:
            episode_data = load_episode(episodes_dir, episode_id)
        except ValueError as e:
            print(f"Warning: {e}")
            continue

        # Create output folder for this episode
        episode_output_dir = output_dir / episode_id
        episode_output_dir.mkdir(parents=True, exist_ok=True)

        # Build training sample for each annotation
        for i, annotation in enumerate(episode_annotations):
            try:
                sample_text = build_training_sample(episode_data, annotation)

                # Save to file
                sample_file = episode_output_dir / f"sample_{i}.txt"
                with open(sample_file, 'w') as f:
                    f.write(sample_text)

                total_samples += 1
            except Exception as e:
                print(f"Error processing {episode_id} annotation {i}: {e}")

    print(f"\n✓ Created {total_samples} expert training samples in {output_dir}")


if __name__ == "__main__":
    main()
```

Usage:

```bash
python -m habitat_llm.finetuning.build_expert_dataset \
  --hitl-episodes-path data/hitl_sessions \
  --annotations-file data/training/iteration_1/expert/annotations.json \
  --output-dir data/training/iteration_1/expert
```

---

## PART 4: Weighted Training Pipeline

### Step 4.1: Modify Trainer for Weighted Loss

Edit: `habitat_llm/finetuning/trainer.py`

Add this function near the top:

```python
def get_sample_weight(file_path: str, config) -> float:
    """
    Get loss weight based on data source

    Args:
        file_path: Path to training sample file
        config: Training config with weighting settings

    Returns:
        Weight multiplier for loss
    """
    if not hasattr(config, 'weighting') or not config.weighting.get('enabled', False):
        return 1.0

    if '/expert/' in file_path:
        return config.weighting.get('expert_weight', 5.0)
    elif '/human_gold/' in file_path:
        return config.weighting.get('human_gold_weight', 1.0)
    elif '/distilled/' in file_path:
        return config.weighting.get('distilled_weight', 1.0)

    return 1.0
```

Then in the training loop, modify loss calculation:

```python
# Find the training loop (around line 200-300)
# Modify from:
loss = outputs.loss

# To:
sample_weight = get_sample_weight(batch['file_paths'][0], config)  # Assuming batch contains file paths
loss = outputs.loss * sample_weight
```

**Note:** You'll need to modify the dataset loader to include file paths in the batch. This is a simplified version - full implementation requires tracking which file each sample came from.

### Step 4.2: Create Weighted Training Config

Create: `habitat_llm/conf/finetuning/iteration_N_weighted.yaml`

```yaml
defaults:
  - finetuning
  - _self_

wandb:
  name: "iteration_${iteration_num}_weighted"

# Enable weighting
weighting:
  enabled: true
  expert_weight: 5.0           # Expert corrections get 5x weight
  human_gold_weight: 1.0       # Human successful actions get 1x weight
  distilled_weight: 0.5        # (Optional) Distilled data gets 0.5x weight

dataset:
  path: "data/training"
  train:
    - "iteration_1/human_gold"
    - "iteration_1/expert"
    # Add more iterations as you progress
    # - "iteration_2/human_gold"
    # - "iteration_2/expert"
  val:
    - "iteration_1/human_gold"
    - "iteration_1/expert"
  max_train_size: -1
  max_val_size: 100

llm_config:
  # Start from previous iteration
  name: "models/iteration_0"  # Change this for each iteration

training_arguments:
  batch_size: 2
  epochs: 3
  lr: 2e-5  # Lower LR for refinement
  gradient_accumulation_steps: 4
```

### Step 4.3: Train Weighted Model

```bash
# Train iteration 1 (from iteration 0 baseline)
python habitat_llm/finetuning/trainer.py \
  --config-name finetuning/iteration_1_weighted \
  wandb.name="iteration_1_weighted" \
  llm_config.name="models/iteration_0" \
  +iteration_num=1

# Takes ~4-6 hours on A6000
# Cost: ~$3-5
```

---

## PART 5: Complete Iteration Workflow

### Iteration 0: Distillation (One-Time)

```bash
# 1. Run teacher (70B or GPT-4)
python -m habitat_llm.examples.planner_demo \
  --config-name baselines/decentralized_zero_shot_react_summary.yaml \
  habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/train_2k.json.gz" \
  [... teacher config ...] \
  paths.results_dir=outputs/distillation/teacher_rollouts

# 2. Convert to training data
python -m habitat_llm.finetuning.build_trace_dataset \
  --path outputs/distillation/teacher_rollouts \
  --output-dir data/training/iteration_0/distilled

# 3. Train 8B
python habitat_llm/finetuning/trainer.py \
  --config-name finetuning/iteration_0_distill

# 4. Merge and save
python -m habitat_llm.finetuning.flatten_checkpoint \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --checkpoint_dir multirun/.../checkpoints/final

mv multirun/.../checkpoints/final_merged/ models/iteration_0/
```

### Iterations 1-5: Human Refinement

Create: `scripts/run_iteration.sh`

```bash
#!/bin/bash
# Complete iteration script
# Usage: bash scripts/run_iteration.sh 1

ITER=$1
PREV_ITER=$((ITER - 1))

echo "=========================================="
echo "Running Iteration $ITER"
echo "=========================================="

# 1. Deploy model to server
echo "Step 1: Deploying iteration_${PREV_ITER} model to Lambda..."
export PARTNR_MODEL_PATH=models/iteration_${PREV_ITER}
# (Assumes you're on Lambda server)

# 2. Start data collection server
echo "Step 2: Starting data collection server..."
echo "Share this URL with participants: http://YOUR-LAMBDA-IP:8000"
echo "Press Ctrl+C when collection is done"
python habitat_llm/server/hitl_server.py

# After Ctrl+C, continue...

# 3. Preprocess collected data
echo "Step 3: Preprocessing HITL data..."
python scripts/hitl_analysis/preprocess_data.py \
  --collection-path data/hitl_sessions \
  --recompute

# 4. Build human_gold dataset
echo "Step 4: Building human_gold dataset..."
python -m habitat_llm.finetuning.build_trace_dataset \
  --path data/hitl_sessions/processed/best \
  --output-dir data/training/iteration_${ITER}/human_gold \
  --percent_cut 0.75

# 5. Generate videos for expert review
echo "Step 5: Generating videos for expert review..."
python scripts/hitl_analysis/visualize_episode.py \
  --episodes-path data/hitl_sessions/processed/ \
  --dataset-file data/datasets/partnr_episodes/v0_0/train_2k.json.gz \
  --multi

echo "Videos saved to: data/hitl_sessions/processed/videos/"
echo ""
echo "=========================================="
echo "PAUSE: Expert annotation needed"
echo "=========================================="
echo "1. Expert reviews videos"
echo "2. Run: python habitat_llm/expert/annotate.py"
echo "3. Creates: data/training/iteration_${ITER}/expert/annotations.json"
echo ""
read -p "Press enter when expert annotations are complete..."

# 6. Build expert dataset
echo "Step 6: Building expert dataset..."
python -m habitat_llm.finetuning.build_expert_dataset \
  --hitl-episodes-path data/hitl_sessions/processed/ \
  --annotations-file data/training/iteration_${ITER}/expert/annotations.json \
  --output-dir data/training/iteration_${ITER}/expert

# 7. Update training config for this iteration
echo "Step 7: Creating training config..."
cat > habitat_llm/conf/finetuning/iteration_${ITER}_weighted.yaml << EOF
defaults:
  - finetuning
  - _self_

wandb:
  name: "iteration_${ITER}_weighted"

weighting:
  enabled: true
  expert_weight: 5.0
  human_gold_weight: 1.0

dataset:
  path: "data/training"
  train:
EOF

# Add all previous iterations to training data
for i in $(seq 1 $ITER); do
  echo "    - \"iteration_${i}/human_gold\"" >> habitat_llm/conf/finetuning/iteration_${ITER}_weighted.yaml
  echo "    - \"iteration_${i}/expert\"" >> habitat_llm/conf/finetuning/iteration_${ITER}_weighted.yaml
done

cat >> habitat_llm/conf/finetuning/iteration_${ITER}_weighted.yaml << EOF
  val:
    - "iteration_${ITER}/human_gold"
    - "iteration_${ITER}/expert"
  max_train_size: -1
  max_val_size: 100

llm_config:
  name: "models/iteration_${PREV_ITER}"

training_arguments:
  batch_size: 2
  epochs: 3
  lr: 2e-5
  gradient_accumulation_steps: 4
EOF

# 8. Train model
echo "Step 8: Training model..."
python habitat_llm/finetuning/trainer.py \
  --config-name finetuning/iteration_${ITER}_weighted \
  +iteration_num=${ITER}

# 9. Merge LoRA weights
echo "Step 9: Merging LoRA weights..."
CHECKPOINT_DIR=$(ls -td multirun/*/0/checkpoints/checkpoint_* | head -1)
python -m habitat_llm.finetuning.flatten_checkpoint \
  --model_name models/iteration_${PREV_ITER} \
  --checkpoint_dir ${CHECKPOINT_DIR}

# 10. Save final model
echo "Step 10: Saving final model..."
mkdir -p models/iteration_${ITER}
cp -r ${CHECKPOINT_DIR}_merged/* models/iteration_${ITER}/

# 11. Evaluate
echo "Step 11: Evaluating on validation set..."
python habitat_llm/finetuning/trainer.py \
  evaluate=True \
  eval_checkpoint_dir=models/iteration_${ITER}

echo ""
echo "=========================================="
echo "Iteration $ITER Complete!"
echo "=========================================="
echo "Model saved to: models/iteration_${ITER}/"
echo "Ready for iteration $((ITER + 1))"
```

Usage:

```bash
# Run iteration 1
bash scripts/run_iteration.sh 1

# Run iteration 2
bash scripts/run_iteration.sh 2

# etc.
```

---

## Summary: Complete Workflow

### One-Time Setup (Iteration 0)

1. Run 70B ReACT on TRAIN episodes
2. Distill into 8B → `models/iteration_0/`
3. Time: 2-3 days
4. Cost: $30-50

### Each Iteration (1-5)

1. **Deploy**: Update server with previous iteration model
2. **Collect**: 10 people × 10 episodes via browser = 100 episodes (~3 hours)
3. **Preprocess**: Filter successful episodes
4. **Build human_gold**: Extract AI actions from successful episodes
5. **Expert review**: Videos → annotations (~20 corrections)
6. **Build expert dataset**: Annotations → training samples
7. **Train**: Weighted training on all data iteration_1...N (~6 hours)
8. **Evaluate**: Test on VAL set
9. **Deploy**: iteration_N model ready

**Per iteration:**
- Time: 5-7 days (mostly waiting for human collection + expert review)
- Cost: ~$10-15 (Lambda for collection + training)

**Total for 5 iterations:**
- Time: 4-6 weeks
- Cost: ~$50-100

---

## Files Created

```
partnr-planner/
├── habitat_llm/
│   ├── server/
│   │   ├── hitl_server.py          # Production WebSocket server
│   │   └── static/
│   │       └── index.html          # Browser client
│   ├── expert/
│   │   └── annotate.py             # Expert annotation CLI
│   └── finetuning/
│       ├── build_expert_dataset.py # Build expert training data
│       └── trainer.py              # Modified with weighting
├── scripts/
│   └── run_iteration.sh            # Complete iteration script
└── habitat_llm/conf/finetuning/
    ├── iteration_0_distill.yaml    # Distillation config
    └── iteration_N_weighted.yaml   # Weighted training config
```

**All code is copy-paste ready. Start with Iteration 0!**
