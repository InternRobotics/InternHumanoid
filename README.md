<p align="center">
  <img src="doc/InternHumaniod.png" alt="InternHumanoid Logo" width="50%">
</p>



A **versatile, all-in-one** toolbox for whole-body humanoid robot control—enabling universal motion tracking, upper–lower body split strategies, and accelerated experimentation across simulation and real-world platforms.



## 🚀 Highlights

- **Whole Body Control Mode**: Effortlessly track full-body human motions in a *zero-shot* fashion—generalize, don’t overfit.
- **Upper–Lower Body Split Mode**: Enhanced control strategy like [Homie](https://arxiv.org/abs/2502.13013) with dynamic walking and powerful manipulation—seamless coordination, robust skills.
- **Multi-Robot Ready**: Instantly deploy on `Unitree G1`, `H1`, `H1-2`, and `Fourier GR-1`—with more robots joining the lineup!
- **Lightning-Fast Experimentation**: Tweak everything with flexible Hydra configs—adapt, iterate, and innovate at speed.
- **Sim-to-Real Mastery**: Built-in friction & mass randomization, noisy observations, and Sim2Sim testing—engineered for real-world success.



## 📰 News

- **[2025/07]** First Release for Universal Humanoid Motion Tracking on Unitree G1!


## 🚧 TODO
- \[x\]  Release Whole Body Control Mode on Unitree G1
- \[x\]  Release Upper–Lower Body Split Mode on Unitree G1
- \[ \] Release Pre-trained Checkpoints and Training Data
- \[ \] Release Environments on Different Robots
- \[ \] Release Deployment Codes

---

## 📋 Table of Contents

- [🚀 Highlights](#-highlights)
- [📰 News](#-news)
- [🚧 TODO](#-todo)
- [⚡ Quick Start](#-quick-Start)
- [🛠️ Installation](#-installation)
- [🗂️ Code Structure](#-code-structure)
- [🧩 Adding New Environments](#-adding-new-environments)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

---

## ⚡ Quick Start

The typical workflow for controlling real-world humanoid robots with InternHumanoid:

`Train` → `Play` → `Sim2Sim` → `Sim2Real`

### Training

Train the universal motion tracker for Unitree G1-29 DoF:

```bash
python legged_gym/scripts/train.py +algo=ppo +robot=g1/g1_29dof +task=imitation/g1_29dof
```
- To run on CPU: add `+sim_device=cpu +rl_device=cpu`
- To run headless (no rendering): add `+headless`
- Trained policies are saved in `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

### Playing

After training, play the saved checkpoint:

```bash
python legged_gym/scripts/play.py +algo=ppo +robot=g1/g1_29dof +task=imitation/g1_29dof
```
- By default, loads the last model of the last run in the experiment folder.

### Sim2Sim

Test the saved ONNX model with sim2sim transfer (Mujoco as the testing environment):

```bash
cd sim2sim
python play_im.py --robot g1_29dof
```

More details of training and playing can be found in the [documentation](doc/train_and_play.md).

---

## 🛠️ Installation

Please refer to the [installation guide](doc/setup_en.md) for detailed steps and configuration instructions.

---

## 🗂️ Code Structure

### Simulation Environment (`legged_gym`)
- `envs/`         : Environment/task definitions
- `config/`       : YAML configuration files for tasks, robots, terrains, algorithms
- `utils/`        : Math, logging, motion libraries, terrain helpers, task registry
- `scripts/`      : Entry-point scripts for training, playing, and exporting models

### Reinforcement Learning (`rsl_rl`)
- `algorithms/`   : RL algorithms (e.g., PPO variants)
- `modules/`      : Neural network modules (actor-critic, normalization, etc.)
- `runners/`      : Training and evaluation runners
- `env/`          : Environment wrappers and vectorized interfaces
- `storage/`      : Rollout storage and replay buffers
- `utils/`        : Utility functions and experiment helpers

---

## 🧩 Adding New Environments

To add a new simulation environment or modify configuration files, see [add new experiments.md](doc/add_exps.md) for a step-by-step guide and detailed examples.

---


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@misc{internhumanoid2025,
    title = {InternHumanoid: Universal Whole-Body Control and Imitation for Humanoid Robots},
    author = {InternHumanoid Contributors},
    howpublished={\url{https://github.com/InternRobotics/InternHumanoid}},
    year = {2025}
}
```

---

## 📄 License

InternHumanoid is [MIT licensed](LICENSE).  
Open-sourced data are under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## 👏 Acknowledgements

- [legged_gym](https://github.com/leggedrobotics/legged_gym): Foundation for training and running codes.
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl): Reinforcement learning algorithms.
- [mujoco](https://github.com/google-deepmind/mujoco): Powerful simulation functionalities.
- [unitree_rl](https://github.com/unitreerobotics/unitree_rl_gym): Powerful reinforcement learning framework provided for Unitree Robots.
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python): Hardware communication interface for physical deployment.

---

Let me know if you want to further customize any section, add badges, or include demo images/videos!