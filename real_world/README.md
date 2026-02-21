# Real-World Quickstart (Docker)

Use two containers:

1. LEAP ROS controller container from `https://github.com/Msornerrrr/leap_hand_controller`
2. Policy runtime container from this folder (`Dockerfile` + `docker-compose.yml`)

This folder is ROS1 (`rospy`), so Docker with Noetic is the easiest path on Ubuntu 22.04.

## 1) Start LEAP controller

Bring up the controller stack first (from `leap_hand_controller`) so `/leap_hand_state` and `/leaphand_node/cmd_leap` are available.

## 2) Build policy runtime image

From this repo:

```bash
cd real_world
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
docker compose build
```

## 3) Run scripts

Smoke test (no ZMQ policy server required):

```bash
docker compose run --rm policy-runtime python3 real_world/random_leap_policy.py
```

Policy deploy (requires your ZMQ policy server running):

```bash
docker compose run --rm policy-runtime \
  python3 real_world/deploy.py \
  --server-endpoint tcp://0.0.0.0:5555 \
  --policy-hz 20 \
  --init-duration-s 3.0
```

System identification data collection:

```bash
docker compose run --rm policy-runtime \
  python3 real_world/sysid_collect.py
```

Optional custom output dir / finger subset:

```bash
docker compose run --rm policy-runtime \
  python3 real_world/sysid_collect.py \
  --out-dir /workspace/in-hand-rotation-mjlab/data/sysid_run_01 \
  --fingers index middle
```

Default output (when `--out-dir` is omitted):
- `/workspace/in-hand-rotation-mjlab/data/YYYY-MM-DD-sysid/auto/`

On your host, that is:
- `../data/YYYY-MM-DD-sysid/auto/` (from `real_world/`)
- `<repo_root>/data/YYYY-MM-DD-sysid/auto/`

## 4) ROS networking variables

`docker-compose.yml` passes through these vars from your host shell:

- `ROS_MASTER_URI`
- `ROS_HOSTNAME` (optional)
- `ROS_IP` (optional)

Set them before running `docker compose run`, for example:

```bash
export ROS_MASTER_URI=http://<controller_ip>:11311
export ROS_IP=<this_machine_ip>
# or: export ROS_HOSTNAME=<this_machine_hostname>

docker compose run --rm policy-runtime python3 real_world/random_leap_policy.py
```
