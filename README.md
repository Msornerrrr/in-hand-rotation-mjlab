# In-Hand Cube Rotation with LEAP Hand

<!-- TODO: Add badges here (License, DOI, etc.) -->

This repository contains Reinforcement Learning (RL) environments for in-hand cube rotation with the LEAP hand. The environments are built using the [MjLab](https://github.com/mujocolab/mjlab) framework and MuJoCo physics engine.

<!-- Videos: GitHub doesn't support <video> tags, use GIF or link to external video -->
<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="assets/sim_demo.gif" width="100%" alt="Simulation Demo"/>
      <br/>
      <em>Simulation</em>
    </td>
    <td align="center" width="50%">
      <img src="assets/real_demo.gif" width="100%" alt="Real Robot Demo"/>
      <br/>
      <em>Real Robot (Video compressed)</em>
    </td>
  </tr>
</table>

The videos above show a trained policy rotating a cube in simulation (left) and on the real LEAP hand (right). The policy is trained using domain randomization and asymmetric actor-critic to enable robust sim-to-real transfer.

---

**Built on prior work:** This implementation is based on the task design and reward formulations from [HORA](https://github.com/haozhiqi/hora) and uses robot models from [LEAP Hand Sim](https://github.com/leap-hand/LEAP_Hand_Sim). We re-implement these methods using the [MjLab](https://github.com/mujocolab/mjlab) framework to provide a modular, extensible codebase for in-hand manipulation research. We thank the authors of these prior works for their excellent contributions to the field!

---

## Overview

This repository implements a complete sim-to-real pipeline for in-hand manipulation:

- **Three LEAP hand embodiments**: right, left, and custom left hand variants
- **Asymmetric actor-critic**: Noisy, delayed proprioception for the actor; full state for the critic
- **Curriculum learning**: Penalties scale with rotation progress to enable gradual skill acquisition
- **Domain randomization**: Cube properties, contact dynamics, and actuator parameters randomized for robust sim-to-real transfer
- **Hardware deployment**: ZMQ-based policy server for real-time inference on physical hardware

The implementation supports **grasp cache generation**, **parallel training** (multi-GPU), **sim2sim testing**, **trajectory replay**, and **system identification** for actuator parameter fitting.

---

## Install

To install the repository, you need the [uv](https://docs.astral.sh/uv/) package manager. If you don't have it yet, install it by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

Clone this repository and run:

```bash
cd ~/projects/in-hand-rotation-mjlab

# Initialize git submodules (mjlab framework)
git submodule update --init --recursive

# Install dependencies
uv sync
```

---

**See [PIPELINE.md](PIPELINE.md) for the complete sim-to-real deployment pipeline**, including:
- Grasp cache generation
- Sim2sim deployment testing
- Policy server setup
- Trajectory recording and replay
- System identification for actuator calibration

---

## Using a Pre-Trained Agent

You can use a pre-trained agent directly in the MjLab environment to see the cube rotation behavior:

```bash
uv run python scripts/play.py Mjlab-Leap-Left-HandCube-Rotate \
  --checkpoint-file logs/rsl_rl/<experiment_name>/<timestamp>/model_*.pt
```

To test the environment with random actions before training:

```bash
# Random actions
uv run python scripts/play.py Mjlab-Leap-Left-HandCube-Rotate --agent random

# Zero actions (observe passive dynamics)
uv run python scripts/play.py Mjlab-Leap-Left-HandCube-Rotate --agent zero
```

---

## Training Your Own Agent

You can customize the environments and train your own policies. Environment configurations are located at `src/in_hand_rotation_mjlab/tasks/hand_cube/config/<embodiment>/`.

### Test the Environment

Before training, verify the environment setup:

```bash
uv run python scripts/play.py Mjlab-Leap-Left-HandCube-Rotate --agent random
```

### Start Training

Train using parallel environments (GPU recommended):

```bash
uv run python scripts/train.py Mjlab-Leap-Left-HandCube-Rotate --env.scene.num-envs 4096
```

This uses 4096 parallel environments for efficient training. Checkpoints are saved to `logs/rsl_rl/<experiment_name>/<timestamp>/`.

### Multi-GPU Training

For multi-GPU training:

```bash
uv run python scripts/train.py Mjlab-Leap-Left-HandCube-Rotate --gpu-ids all
```

### Play Trained Policy

Once training completes, evaluate the trained policy:

```bash
uv run python scripts/play.py Mjlab-Leap-Left-HandCube-Rotate \
  --checkpoint-file logs/rsl_rl/<experiment_name>/<timestamp>/model_*.pt
```

---

## Deploying to Real Hardware

For hardware deployment, we provide a ZMQ-based policy server that runs inference and communicates with the physical LEAP hand controller.

Start the policy server:

```bash
uv run python scripts/policy_server_zmq.py Mjlab-Leap-Left-HandCube-Rotate \
  --checkpoint-file logs/rsl_rl/<experiment_name>/<timestamp>/model_*.pt \
  --bind tcp://0.0.0.0:5555
```

The server processes observations from hardware and returns joint position commands at 20 Hz.

---

## Available Tasks

| Task ID | Robot | Description |
|---------|-------|-------------|
| `Mjlab-Leap-Right-HandCube-Rotate` | LEAP Right | Right-handed cube rotation |
| `Mjlab-Leap-Left-HandCube-Rotate` | LEAP Left | Left-handed cube rotation |
| `Mjlab-Leap-Left-Custom-HandCube-Rotate` | LEAP Left Custom | Custom left hand variant |

Each task can be customized independently via robot-specific parameters (grasp pose, spawn position, cache file) and shared configuration (observations, rewards, domain randomization).

---

## Key Design Choices

### Observations
- **Actor** (deployed on hardware): Noisy, delayed joint positions (history=10) and commanded positions
- **Critic** (training only): Full state including cube pose/velocity and physical parameters

### Actions
- Joint position delta: ±1/24 rad/step at 20 Hz control frequency

### Rewards
- Rotation progress (drift-gated to penalize unwanted axis rotation)
- Penalties: linear velocity, pose deviation, torque, work
- Curriculum: penalties scale with rotation progress for gradual learning

### Domain Randomization
- **Cube**: mass, size, center of mass, friction
- **Contacts**: hand-cube friction coefficients
- **Actuators**: friction, damping, PD gains, effort limits, control delays

This enables robust policies that transfer from simulation to real hardware.

---

## Project Structure

```
in_hand_rotation_mjlab/
├── src/in_hand_rotation_mjlab/
│   ├── tasks/hand_cube/          # Task implementation
│   │   ├── hand_cube_env_cfg.py  # Shared config + helper
│   │   ├── config/               # Per-embodiment configs
│   │   │   ├── leap_left/
│   │   │   ├── leap_right/
│   │   │   └── leap_left_custom/
│   │   ├── mdp/                  # MDP modules (observations, rewards, etc.)
│   │   └── cache/                # Grasp caches
│   ├── robots/leap_hand/         # Robot models & assets
│   ├── sim2sim/                  # Deployment testing infrastructure
│   └── policy_server/            # ZMQ inference server for hardware
├── scripts/                      # Training, evaluation, and deployment scripts
└── third_party/mjlab/            # RL framework (git submodule pointer; source not vendored)
```

---

## Roadmap

**Current Release:**
- ✅ Three LEAP hand embodiments (right, left, left custom)
- ✅ Complete training pipeline with domain randomization
- ✅ Sim2sim deployment testing
- ✅ Policy server infrastructure for hardware deployment
- ✅ System identification tools

**Planned Releases:**
- 🔄 **Real-world deployment code** - Hardware interface code for LEAP hand control
- 🔄 **Hardware setup guide** - Detailed instructions for physical LEAP hand setup

---

## License

mjlab is licensed under the [Apache License, Version 2.0.](LICENSE)

---

## Acknowledgments

This work builds upon several excellent open-source projects:

- **[MjLab](https://github.com/mujocolab/mjlab)** - The RL framework used for training and environment management. We thank the MjLab team for their excellent work on this framework for MJWarp-based robotics tasks.

- **[HORA](https://github.com/haozhiqi/hora)** - Our task design, reward functions, and training methodology are based on the HORA paper. We thank the authors for their pioneering work on in-hand object rotation.

- **[LEAP Hand Sim](https://github.com/leap-hand/LEAP_Hand_Sim)** - The LEAP hand robot models and assets are from this repository. We thank the LEAP hand team for making these resources available to the community.

If you use this codebase, please also consider citing the original HORA and LEAP Hand papers that this work builds upon.
