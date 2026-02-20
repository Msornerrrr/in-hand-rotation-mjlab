# Training and Deployment Pipeline

Complete guide for training and deploying in-hand cube rotation policies.

---

## Core Pipeline

### 1. Generate Grasp Cache

Create initial grasp poses for reset randomization:

```bash
uv run python scripts/collect_hand_cube_grasp_cache.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --out-file src/in_hand_rotation_mjlab/tasks/hand_cube/cache/leap_left_grasp_cache.npz \
  --num-envs 256 \
  --num-sizes 32
```

**For other embodiments:**
- `Mjlab-Leap-Right-HandCube-Rotate` → `leap_right_grasp_cache.npz`
- `Mjlab-Leap-Left-Custom-HandCube-Rotate` → `leap_left_custom_grasp_cache.npz`

**Note:** This collects grasps across a grid of cube sizes (default: 32 sizes) with multiple parallel environments (default: 256). Total samples = `num_envs × num_sizes × stability_rate`.

---

### 2. Test Environment (Play Mode)

Verify the robot and environment work correctly:

```bash
# Test with random actions
uv run python scripts/play.py Mjlab-Leap-Left-HandCube-Rotate \
  --agent random

# Test with zero actions (useful for testing physics)
uv run python scripts/play.py Mjlab-Leap-Left-HandCube-Rotate \
  --agent zero

# Test with pre-trained policy (default agent is "trained")
uv run python scripts/play.py Mjlab-Leap-Left-HandCube-Rotate \
  --checkpoint-file /path/to/model.pt
```

**Available agent types:**
- `zero`: Zero actions (gravity/passive dynamics only)
- `random`: Random actions in [-1, 1]
- `trained`: Trained policy (default, requires `--checkpoint-file`)

---

### 3. Train Policy

Start RL training:

```bash
uv run python scripts/train.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --env.scene.num-envs 4096
```

**Training outputs:**
- Checkpoints: `logs/rsl_rl/<experiment_name>/<timestamp>/`
- Best model: `logs/rsl_rl/<experiment_name>/<timestamp>/model_*.pt`

**Monitor training (if using wandb):**
- Configure wandb in the RL config: `agent.logger = "wandb"`
- View training metrics at wandb.ai

**Advanced options:**
```bash
# Multi-GPU training (uses all available GPUs)
uv run python scripts/train.py Mjlab-Leap-Left-HandCube-Rotate --gpu-ids all

# Resume from checkpoint
uv run python scripts/train.py Mjlab-Leap-Left-HandCube-Rotate \
  --agent.resume True \
  --agent.load-run <run_id> \
  --agent.load-checkpoint <checkpoint_name>
```

---

### 4. Evaluate Trained Policy

Play the trained policy in simulation:

```bash
uv run python scripts/play.py Mjlab-Leap-Left-HandCube-Rotate \
  --checkpoint-file logs/rsl_rl/<experiment_name>/<timestamp>/model_*.pt \
  --num-envs 1
```

---

### 5. Deploy with Policy Server

Start the policy server for hardware deployment:

```bash
uv run python scripts/policy_server_zmq.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --checkpoint-file logs/rsl_rl/<experiment_name>/<timestamp>/model_*.pt \
  --bind tcp://0.0.0.0:5555
```

The server:
- Loads the trained policy
- Processes observations from hardware
- Returns joint position commands
- Runs at 20 Hz control frequency

**Client connection (from hardware):**
```python
import zmq
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://your-server-ip:5555")

# Send observation, receive action
socket.send_pyobj(observation_packet)
action_packet = socket.recv_pyobj()
```

---

## Optional Pipelines

### Sim2Sim Deployment Testing

Test deployment pipeline in simulation (mimics hardware):

```bash
uv run python scripts/sim2sim.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --checkpoint-file logs/rsl_rl/<experiment_name>/<timestamp>/model_*.pt \
  --num-envs 1 \
  --render
```

**With external policy server (2 terminals):**

Terminal 1 (server):
```bash
uv run python scripts/policy_server_zmq.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --checkpoint-file logs/rsl_rl/<experiment_name>/<timestamp>/model_*.pt \
  --bind tcp://0.0.0.0:5555
```

Terminal 2 (client):
```bash
uv run python scripts/sim2sim.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --server-endpoint tcp://localhost:5555 \
  --render
```

---

### Collect Policy Rollouts

Record policy rollouts for analysis or replay:

```bash
uv run python scripts/record_replay_trajectory.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --checkpoint-file logs/rsl_rl/<experiment_name>/<timestamp>/model_*.pt \
  --output-file rollouts/trajectory_001.npz \
  --num-steps 1000
```

**Note:** `--num-steps` defaults to the episode length from the task config. The output is saved as `.npz` format containing raw actions and joint position commands.

---

### Replay Recorded Trajectories

Open-loop replay of recorded trajectories:

```bash
# Replay server (for hardware replay)
uv run python scripts/replay_server_zmq.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --trajectory-file rollouts/trajectory_001.npz \
  --bind tcp://0.0.0.0:5556
```

---

### System Identification (CEM)

Fit actuator parameters using collected finger trajectories:

```bash
uv run python scripts/sysid_fit_from_log.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --log-file data/real_robot_finger_traj.npz \
  --output-file sysid/fitted_params.json \
  --optimizer cem \
  --cem-generations 12 \
  --parallel-envs 8192
```

**Input format:** NPZ file with arrays `hand_cmd` (T×16), `hand_state` (T×16), and optionally `time` (T,)

**Output:** JSON with fitted PD gains, friction, damping, armature per joint

**Options:**
- `--finger`: Target finger for calibration (`index`, `middle`, `ring`, `thumb`, or `all`)
- `--optimizer`: Optimization method (`cem` or `random`)
- `--cem-generations`: Number of CEM iterations (default: 12)
- `--parallel-envs`: Number of parallel simulation environments (default: 8192)
