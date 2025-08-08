<p align="center">
  <img src="../doc/InternHumaniod.png" alt="InternHumanoid Logo" width="50%">
</p>

# ⚡ Training & Playing

A quick guide to training and evaluating universal motion trackers for humanoid robots using **InternHumanoid**.

For command-conditioned humanoid control and upper-body manipulation, see the reference application [Flomo (Foundation Locomotion)](./flomo_doc/flomo_train_and_play.md).

---

## 📋 Table of Contents

- [Training](#training)
- [Playing](#playing)

---

## Training

Train the universal motion tracker for Unitree G1-29 DoF with the following command:

```bash
python legged_gym/scripts/train.py +algo=ppo +robot=g1/g1_29dof +task=imitation/g1_29dof
```

- To run on CPU, add: `+sim_device=cpu +rl_device=cpu` (sim on CPU and RL on GPU is possible).
- To run headless (no rendering), add: `+headless`.
- **Tip:** To improve performance, once training starts, press `v` to stop rendering. You can enable it later to check progress.
- The trained policy is saved in `issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. `<experiment_name>` and `<run_name>` are defined in the train config.
- Command line arguments override config file values:
    - `+resume`: Resume training from a checkpoint
    - `+experiment_name EXPERIMENT_NAME`: Name of the experiment to run or load
    - `+run_name RUN_NAME`: Name of the run
    - `+load_run LOAD_RUN`: Name of the run to load when resume=True. If -1, loads the last run
    - `+checkpoint CHECKPOINT`: Saved model checkpoint number. If -1, loads the last checkpoint
    - `+num_envs NUM_ENVS`: Number of environments to create
    - `+seed SEED`: Random seed
    - `+max_iterations MAX_ITERATIONS`: Maximum number of training iterations

---

## Playing

After training, play the saved checkpoint with:

```bash
python legged_gym/scripts/motion_imitation/play.py +algo=ppo +robot=g1/g1_29dof +task=imitation/g1_29dof
```

- By default, the last model of the last run in the experiment folder is loaded.
- Other runs/model iterations can be selected by setting `load_run` and `checkpoint` in the train config.