# Quick Start Guide

## Installation

```bash
cd ~/projects/in_hand_rotation_mjlab

# Initialize git submodules (mjlab)
git submodule update --init --recursive

# Option 1: Using uv (recommended)
uv sync

# Option 2: Using pip
pip install -e .
pip install -e third_party/mjlab
```

## Verify Installation

```bash
python3 -c "
from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import make_hand_cube_inhand_rotate_env_cfg
print('✓ Installation successful')
"
```

## Running Examples

### 1. Train a Policy

```python
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.rl import RslRlRunner

# Load configurations
env_cfg = load_env_cfg("hand_cube-leap_left-v0")
agent_cfg = load_rl_cfg("hand_cube-leap_left-v0")

# Create runner and train
runner = RslRlRunner(env_cfg, agent_cfg)
runner.train()
```

### 2. Run Sim2Sim (Test Deployment)

```python
from in_hand_rotation_mjlab.sim2sim import run_native_sim2sim, NativeSim2SimConfig

config = NativeSim2SimConfig(
    task_id="hand_cube-leap_left-v0",
    checkpoint_file="path/to/checkpoint.pt",
    num_envs=1,
    render=True,
)

run_native_sim2sim(config)
```

### 3. Start Policy Server (Hardware Deployment)

```python
from in_hand_rotation_mjlab.policy_server import PolicyServerConfig, MjlabPolicyServer
from in_hand_rotation_mjlab.policy_server.bootstrap import build_server_action_metadata_from_task

# Build metadata
metadata = build_server_action_metadata_from_task(
    task_id="hand_cube-leap_left-v0",
    play=True,
)

# Create server
config = PolicyServerConfig(
    checkpoint_file="checkpoint.pt",
    play=True,
)

server = MjlabPolicyServer(
    task_id="hand_cube-leap_left-v0",
    cfg=config,
    action_term_metadata=metadata,
    step_dt=0.05,  # 20 Hz
    max_episode_steps=400,
)

# Reset and inference
server.reset(obs_packet)
action_packet = server.infer(obs_packet)
```

## Task Details

### Task ID
`hand_cube-leap_left-v0`

### Control Frequency
20 Hz (decimation=10, sim dt=5ms)

### Episode Length
20 seconds (400 steps)

### Action Space
Joint position delta: 16 DoF, clipped to ±1/24 rad/step

### Observation Space (Actor)
- Joint positions (relative to default, noisy + delayed)
- Previous commanded joint positions
- History length: 10 steps

### Observation Space (Critic)
- Joint positions, velocities, tracking error
- Cube pose, velocities (in palm frame)
- Cube properties (size, mass, COM, friction)
- History length: 1 step

## File Locations

- **Task configs**: `src/in_hand_rotation_mjlab/tasks/hand_cube/config/leap_left/`
- **MDP modules**: `src/in_hand_rotation_mjlab/tasks/hand_cube/mdp/`
- **Robot model**: `src/in_hand_rotation_mjlab/robots/leap_hand/`
- **Grasp cache**: `src/in_hand_rotation_mjlab/tasks/hand_cube/cache/leap_left_grasp_cache.npz`
- **Scripts**: `scripts/` (train.py, play.py, sim2sim.py, etc.)

## Useful Commands

```bash
# List all Python files
find src -name "*.py"

# Check imports
python3 -c "from in_hand_rotation_mjlab.tasks.hand_cube import *"

# View task structure
tree src/in_hand_rotation_mjlab/tasks/hand_cube/
```

## Troubleshooting

### ImportError: cannot import mjlab
Make sure mjlab is installed from third_party:
```bash
uv sync
# or
pip install -e third_party/mjlab
```

### Missing default_joint_pos
The default joint positions are defined in:
`src/in_hand_rotation_mjlab/robots/leap_hand/leap_left_constants.py`

### Observation dimension mismatch
Check that your checkpoint was trained with the same observation config.
The policy server will validate dimensions automatically.

## Next Steps

1. Review `README.md` for detailed documentation
2. Check `EXTRACTION_NOTES.md` for what changed from original repo
3. Customize training configs in `src/in_hand_rotation_mjlab/tasks/hand_cube/config/leap_left/`
4. Add your own training scripts to `scripts/` directory
