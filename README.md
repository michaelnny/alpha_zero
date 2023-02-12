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
- [Reference Papers](#reference-papers)
- [Reference Code](#reference-code)
- [License](#license)
- [Citing our work](#citing-our-work)


# Environment and Requirements
* Python        3.9.12
* pip           22.3.1
* PyTorch       1.13.1
* gym           0.25.2
* numpy         1.23.4


# Supported Games
* Tic-Tac-Toe
* Free-style Gomoku


# Code Structure
* `games` directory contains the custom Gomoku board game env implemented with openAI Gym.
* `gomoku` contains the modules to train and play the game
  - `run_training_v1.py` trains the agent following AlphaGo Zero paper
  - `run_training_v2.py` trains the agent following AlphaZero paper (without using evaluation to select 'best' player)
  - `eval_agent.py` evaluate the agents by playing the Gomoku game in terminal mode, only supports AlphaZero vs. AlphaZero mode
  - `eval_agent_gui.py` evaluate the agents by launching a simple GUI program to play Gomoku, supports AlphaZero vs. AlphaZero mode, and Human vs. AlphaZero mode
* `mcts_v1.py` contains the MCTS node and UCT tree-search algorithm.
* `mcts_v2.py` contains the optimized version of MCTS node and UCT tree-search algorithm, which use Numpy arrays to store tree statistics.
* `pipeline_v1.py` contains the functions to run self-play, training, and evaluation loops (following AlphaGo Zero paper, evaluation is used to select best player to generate self-play samples)
* `pipeline_v2.py` contains the functions to run training, and evaluation loops (following AlphaZero paper, evaluation is only used to monitoring performance)


# Author's Notes
* The goal is not to make a strongest player but to study the algorithm, as we stopped the training once the agent have made some progress
* We use scaled down configuration for the training, like using a smaller neural network, lesser simulations per MCTS search, and smaller batch size etc.
* The elo ratings should not taken too seriously, since we don't set the agent to play against some existing (strong) agent


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

## Install required packages on Ubuntu linux
```
# upgrade pip
python3 -m pip install --upgrade pip setuptools

# Python3 tkinter for GUI
sudo apt-get install python3-tk

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

Trains the agent using the AlphaZero method, which is highly recommended.
```
python3 -m alpha_zero.gomoku.run_training_v2


# resume training
python3 -m alpha_zero.gomoku.run_training_v2 --load_samples_file=./samples/gomoku_v2/replay_200000_20230112_102835 --load_checkpoint_file=./checkpoints/gomoku_v2/train_steps_64000 --initial_elo=-2064


# check training performance
python3 -m alpha_zero.plot --train_csv_file=logs/train_gomoku_v2.csv --eval_csv_file=logs/eval_gomoku_v2.csv
```

Trains the agent the AlphaGo Zero method, which may be much slower depending on thr number of evaluation games to play to select the new best player.
```
python3 -m alpha_zero.gomoku.run_training_v1

# check training performance
python3 -m alpha_zero.plot --train_csv_file=logs/train_gomoku_v1.csv --eval_csv_file=logs/eval_gomoku_v1.csv
```



## MCTS performance

We have to implementation of the MCTS search algorithm: `mcts_v1` and `mcts_v2`.
The first one is very basic and easy to understand, however it's very slow due to the nature of massive amount of computations needed to complete the search. The second one is an optimized version, which we use numpy.arrays to store the statistics for the nodes in the search tree.

The following table shows the mean search time (in second) per search for these different MCTS implementations.
We run the experiment with a single thread for 100 search steps on a single RTX 3090 GPU, we use the 400 simulations per MCTS,  for the leaf-parallel search we use a parallel number of 8.
| Module       | Single thread    | Leaf-parallel    |
| ------------ | ---------------- | ---------------- |
| `mcts_v1`    | 0.92             | 1.1              |
| `mcts_v2`    | 0.67             | 0.15             |


## Training on a single machine

Although not directly mentioned in the original papers, we believe AlphaZero runs large amount of self-play actors (5000 TPUs) to generate self-play samples, while training on separate server (16 TPUs) at the same time.

In the case we run the pipeline on a single machine, training one batch of 256 samples (on GPU) takes lesser than a second,
but it could takes much longer time for self-play to generate the same amount of samples. This means the neural network may suffer from the over-fitting issue while fail to converge to an optimal policy.

One solution to mitigate this issue is to add some delay to the learner's training loop. The ideal situation is we want the actors to generate a batch of new samples before starting training on next batch. However, if we delay for too long, more bad samples are generated because the actors are still using the old parameters for neural network. The right value for the train delay parameter would depend on the setup, for example how many actors, what kind of hardware etc.

In our experiment, running 3 actors and a single learner on a single RTX 3090 GPU, using 600 simulations and 8 parallel leaves for MCTS search, it takes ~30 minutes to generate 20000 sample positions. For the learner's training loop, we use batch size 256, and a 0.5 seconds delay per training step, it takes ~10 minutes to run 1000 training steps (one checkpoint). This means every 10 minutes, the actors will start using the new checkpoint to generate new samples.


# Evaluate Agents
You can evaluate the agent by running the `eval_agent` script. In addition for the Gomoku game, We also have a very basic GUI program, which supports
* Human vs. AlphaZero
* AlphaZero vs. AlphaZero

To start play the game, make sure you have a valid checkpoint file and run the following command
```
# Human vs. AlphaZero
python3 -m alpha_zero.gomoku.eval_agent_gui

# AlphaZero vs. AlphaZero
python3 -m alpha_zero.gomoku.eval_agent_gui --nohuman_vs_ai
```


# Training Progress and Performance
## Screenshots Gomoku
* Training performance measured in Elo rating
![Training performance](/screenshots/gomoku_performance.png)

* Evaluation
![Train steps 501000](/screenshots/gomoku_train_steps_501000.png)
![Train steps 502000](/screenshots/gomoku_train_steps_502000.png)

* The agent failed to do well in 'edge' case
![Edge case](/screenshots/gomoku_edge_case.png)


# Reference Papers
* [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270/)
* [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs//1712.01815)


# Reference Code
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
