Alpha Zero
=============================
A PyTorch implementation of DeepMind's AlphaZero agent to play Free-style Gomoku board game


# Content
- [Environment and Requirements](#environment-and-requirements)
- [Supported Games](#supported-games)
- [Code Structure](#code-structure)
- [Author's Notes](#authors-notes)
- [Quick Start](#quick-start)
- [Train Agents](#train-agents)
- [Evaluate Agents](#evaluate-agents)
- [Training Progress and Performance](#training-progress-and-performance)
- [Training Experiments](#training-experiments)
- [TODO](#todo)
- [Reference Papers](#reference-papers)
- [Reference Code](#reference-code)
- [License](#license)
- [Citing our work](#citing-our-work)


# Environment and Requirements
* Python        3.9.12
* pip           22.0.3
* PyTorch       1.11.0
* gym           0.23.1
* numpy         1.21.6


# Supported Games
* Free-style Gomoku
* Tic-Tac-Toe


# Code Structure
* `games` directory contains the custom Gomoku board game env implemented with openAI Gym.
* `gomoku` contains the modules to train and play the game
  - `run_training_v1.py` trains the agent following AlphaGo Zero paper
  - `run_training_v2.py` trains the agent following AlphaZero paper (without using evaluation to select 'best' player)
  - `alpha_zero_vs_alpha_zero.py` launch a GUI program to play in Gomoku AlphaZero vs. AlphaZero mode
  - `human_vs_alpha_zero.py` launch a GUI program to play Gomoku in Human vs. AlphaZero mode
* `mcts.py` contains the MCTS node and UCT tree-search algorithm.
* `pipeline_v1.py` contains the functions to run self-play, training, and evaluation loops (following AlphaGo Zero paper, evaluation is used to select best player to generate self-play samples)
* `pipeline_v2.py` contains the functions to run training, and evaluation loops (following AlphaZero paper, evaluation is only used to monitoring performance)


# Author's Notes
* The goal is not to make a strongest player but to study the algorithm
* We follow the same architecture as mentioned in the AlphaGo Zero paper, use 64 planes and 6 res-blocks
* We use 400 instead of 800 simulations per MCTS search
* We stack most recent 4 board states instead of 8 board states for each player
* We do not implement parallel MCTS search, or batched evaluation during MCTS search
* We do not apply rotation or reflection to the board state during MCTS search
* We stop training once the agent have made some progress


# Quick start
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

## Tic-Tac-Toe

```
python3 -m alpha_zero.tictactoe.run_training_v2

# check training performance
python3 -m alpha_zero.plot --train_csv_file=logs/train_tictactoe_v2.csv --eval_csv_file=logs/eval_tictactoe_v2.csv
```

## Gomoku Game

Training using the AlphaGo Zero method, which may be much slower depending how number of games to play for evaluation to select the new 'best' player.
```
python3 -m alpha_zero.gomoku.run_training_v1

# check training performance
python3 -m alpha_zero.plot --train_csv_file=logs/train_gomoku_v1.csv --eval_csv_file=logs/eval_gomoku_v1.csv
```

Training using the AlphaZero method.
```
python3 -m alpha_zero.gomoku.run_training_v2

# check training performance
python3 -m alpha_zero.plot --train_csv_file=logs/train_gomoku_v2.csv --eval_csv_file=logs/eval_gomoku_v2.csv
```

# Evaluate Agents
We have a very basic GUI program for Gomoku, which supports
* Human vs. AlphaZero
* AlphaZero vs. AlphaZero

To start play the game, make sure you have a valid checkpoint file and run the following command
```
# Human vs. AlphaZero
python3 -m alpha_zero.gomoku.human_vs_alpha_zero

# AlphaZero vs. AlphaZero
python3 -m alpha_zero.gomoku.alpha_zero_vs_alpha_zero
```

# Training Progress and Performance
## Screenshots Gomoku
* Training performance measured in Elo rating
![Training performance](../main/screenshots/gomoku_performance.png)

* Evaluation
![Train steps 501000](../main/screenshots/gomoku_train_steps_501000.png)
![Train steps 502000](../main/screenshots/gomoku_train_steps_502000.png)

* The agent failed to do well in 'edge' case
![Edge case](../main/screenshots/gomoku_edge_case.png)


## Screenshots Tic-Tac-Toe
* Training performance measured in Elo rating
![Training performance](../main/screenshots/tictactoe_performance.png)


# Training Experiments

Although not directly mentioned in the original papers, we believe AlphaZero runs large amount of self-play actors (256 or more) to generate self-play samples,
while training on multiple servers at the same time.

In the case we run the pipeline on a single machine at home, training one batch with batch size 128 only take less than a second on GPU, few seconds on CPU,
but self-play one game could take few minutes (in the early stage one game only lasts 30-50 steps).
If we run training with 8-16 actors, the network could easily over-fitting to existing samples while fail to converge.

One hack is to add some delay to the training loop, to wait for the self-play actors to generate more training samples.
The downside is this will slow down the overall training progress, we also have to 'tune' the train delay hyper-parameters.

We conducted the following experiments:
* For training and self-play on CPU (M1 Mac Mini), we found use `--train_delay=0.25` yields one checkpoint every 10 minutes.
  Took 90-120 minutes for the actors to generate 10k self-play samples.
* For training and self-play on single RTX 2080 Ti GPU, we found use `--train_delay=0.5` yields one checkpoint every 10 minutes.
  Took 80-100 minutes for the actors to generate 10k self-play samples.
* We observed it takes around 400k training steps for the agent to reach a reasonable 'strong' level that can beat a beginner human player, and sometimes beats a 'strong' human player (I'm really not good at this game but my partner is really good, she beats me every single time).

The above experiments were conducted under the same condition (using `run_training_v2.py`) and hyper parameters:
* `--board_size=11`
* `--stack_history=4`
* `--checkpoint_frequency=1000`
* `--num_simulations=400`
* `--num_actors=6`
* `--batch_size=128`
* `--replay_capacity=1000 * 50`
* `--min_replay_size=1000 * 5`

Based on above statistics and observation, it could take 400*10=4000 minutes, which is 66 hours (~3 days) for the agent to converge to a suboptimal policy.
But you can tune the hyper parameters to suit your own environment. For example the number of actors, batch size, train delay.

One additional note is that if we really want to train a strong player, we need to increase the neural network capacity by using larger number of planes in the Conv2d layers and more res-blocks.


# TODO
* Finish grpc module (adapt SEED RL architecture) to minimize GPU resource when running self-play on GPUs,
  as there are some minimum PyTorch resource allocation (we found about 1GB GPU RAM per process)


# Reference Papers
* [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270/)
* [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs//1712.01815)


# Reference Code
* [A Deep Dive into Monte Carlo Tree Search](https://www.moderndescartes.com/essays/deep_dive_mcts/)
* [MCTS basic code](https://github.com/brilee/python_uct)
* [AlphaZeroSimple on MCTS node statistics backup and UCT score](https://github.com/JoshVarty/AlphaZeroSimple)
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
