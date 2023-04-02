Alpha Zero
================================================================================================
A PyTorch implementation of DeepMind's AlphaZero agent to play Free-style Gomoku board game


# Content
- [Environment and Requirements](#environment-and-requirements)
- [Code Structure](#code-structure)
- [Author's Notes](#authors-notes)
- [Quick Start](#quick-start)
- [Train Agents](#train-agents)
- [Evaluate Agents](#evaluate-agents)
- [Training Progress and Performance](#training-progress-and-performance)
- [References](#references)
- [License](#license)
- [Citing our work](#citing-our-work)


# Environment and Requirements
* Python        3.9.12
* pip           22.3.1
* PyTorch       1.13.1
* gym           0.25.2
* numpy         1.23.4


# Code Structure
* `games` directory contains the custom Gomoku board game env implemented with OpenAI Gym.
* `gomoku` contains the modules to train and play the game
  - `run_training_v1.py` trains the agent following AlphaGo Zero paper
  - `run_training_v2.py` trains the agent following AlphaZero paper (without using evaluation to select 'best' player)
  - `eval_agent.py` evaluate the agents by playing the Gomoku game in terminal mode, only supports AlphaZero vs. AlphaZero mode
  - `eval_agent_gui.py` evaluate the agents by launching a simple GUI program to play Gomoku, supports AlphaZero vs. AlphaZero mode, and Human vs. AlphaZero mode
* `network.py` contains the AlphaZero neural network
* `mcts_v1.py` contains the MCTS node and UCT tree-search algorithm.
* `mcts_v2.py` contains the optimized version of MCTS node and UCT tree-search algorithm.
* `pipeline_v1.py` contains the functions to run self-play, training, and evaluation loops (following AlphaGo Zero paper, evaluation is used to select best player to generate self-play samples)
* `pipeline_v2.py` contains the functions to run training, and evaluation loops (following AlphaZero paper, evaluation is only used to monitoring performance)
* `data_processing.py` contains the functions to argument the training samples through random rotation and mirroring
* `replay.py` contains the functions to a very simple uniform random experience replay


# Author's Notes
* The goal is not to make a strongest player but to study the algorithm, as we stopped the training once the agent have made some progress
* We use scaled down configuration for the training, like using a smaller board size, smaller neural network, lesser simulations per MCTS search etc.
* The elo ratings should not taken too seriously, since we don't set the agent to play against some existing (strong) player


# Quick start
## Install required packages on Ubuntu linux
```
# upgrade pip
python3 -m pip install --upgrade pip setuptools

# Python3 tkinter for GUI
sudo apt-get install python3-tk

pip3 install -r requirements.txt
```

## Install required packages on Mac
```
# install homebrew, skip this step if already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# upgrade pip
python3 -m pip install --upgrade pip setuptools

# Python3 tkinter for GUI
brew install python-tk

pip3 install -r requirements.txt
```


# Train Agents

Trains the agent using the AlphaZero method, which is highly recommended.
```
python3 -m alpha_zero.gomoku.run_training_v2


# resume training
python3 -m alpha_zero.gomoku.run_training_v2 --load_checkpoint_file=checkpoints/gomoku_v2/train_steps_64000 --initial_elo=64


# check training performance
python3 -m alpha_zero.plot --train_csv_file=logs/train_gomoku_v2.csv --eval_csv_file=logs/eval_gomoku_v2.csv
```

Trains the agent the AlphaGo Zero method, which may be much slower depending on the number of evaluation games to play to select the new best player.
```
python3 -m alpha_zero.gomoku.run_training_v1

# check training performance
python3 -m alpha_zero.plot --train_csv_file=logs/train_gomoku_v1.csv --eval_csv_file=logs/eval_gomoku_v1.csv
```

## Training on a single machine

Although the original papers don't explicitly mention it, we believe that AlphaZero utilizes a large number of self-play actors (5000 TPUs) to generate self-play samples while simultaneously training on a separate server (16 TPUs).

When running the pipeline on a single machine, it's important to balance the speed of learning (train sample rate) with the speed of self-play (sample generation rate), and we want the sample generation rate as high as possible. Our experiments have shown that sample generation rate to train sample rate ratio 1:1 or 1:2. This can be achieved by adjusting the batch size, and training delay in combination. The exact combination depends on the hardware and training settings, such as the complexity of the neural network, the number of self-play actors, and the number of MCTS simulations. Therefore, it must be adjusted on a case-by-case basis.

### Example Analysis

Let's say that it takes the actors 5 minutes to generate 1000,000 sample positions, resulting in a sample generation rate of 3333 samples per second. Our goal is to achieve a sample generation rate to train sample rate of 1:2, so we need to determine an appropriate batch size. If each batch takes the learner 0.01 seconds to complete, we can calculate the batch size using the following formula: (3333 x 0.01) x 2 = 66.6, which we can round down to 64.

However, if a single batch can be completed in less than 0.01 seconds which is often the case when using GPU, we'll need to adjust the train delay parameter to ensure that each batch takes 0.01 seconds to complete. This will help us maintain a consistent training pace and ensure that our results are accurate.

### Training Statistics
Gomoku
![Gomoku training performance](/screenshots/gomoku_train_progress.png)


# Evaluate Agents
You can evaluate the agent by running the `eval_agent` script. In addition for the Gomoku game, We also have a very basic GUI program, which supports
* Human vs. AlphaZero
* AlphaZero vs. AlphaZero

To start play the game using the GUI program, run the following command
```
# Human vs. AlphaZero
python3 -m alpha_zero.gomoku.eval_agent_gui

# AlphaZero vs. AlphaZero
python3 -m alpha_zero.gomoku.eval_agent_gui --nohuman_vs_ai
```

![AlphaZero vs AlphaZero](/screenshots/ai_vs_ai.png)

# References

## References Papers
* [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270/)
* [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs//1712.01815)


## Reference Code
* [A Deep Dive into Monte Carlo Tree Search](https://www.moderndescartes.com/essays/deep_dive_mcts/)
* [MCTS basic code](https://github.com/brilee/python_uct)
* [Deep RL Zoo](https://github.com/michaelnny/deep_rl_zoo)


# License

This project is licensed under the Apache License, Version 2.0 (the "License")
see the LICENSE file for details


# Citing our work

If you reference or use our project in your research, please cite:

```
@software{alpha_zero2022github,
  title = {{Alpha Zero}: A PyTorch implementation of DeepMind's AlphaZero agent to play Free-style Gomoku board game},
  author = {Michael Hu},
  url = {https://github.com/michaelnny/alpha_zero},
  version = {1.0.0},
  year = {2022},
}
```
