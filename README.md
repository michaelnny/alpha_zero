Alpha Zero
================================================================================================
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

## Training on a single machine

Although not directly mentioned in the original papers, we believe AlphaZero runs large amount of self-play actors (5000 TPUs) to generate self-play samples, while training on separate server (16 TPUs) at the same time.

In the case we run the pipeline on a single machine, we need to balance between the speed of learning (train sample rate) and sample generation rate.

We can use combination of batch size, learning rate, and training delay to archive that. The exact combination is depend on the hardware and training setting (for example complexity of the neural network, number of self-play actors, number of MCTS simulations), which needs to be adjusted for individual case.

For example, if it takes 5 minutes for the actors to generate a total of 100000 sample positions. Then the sample generation rate is `100000/(5*60) = 333`. This means on average, the actors will generate 333 samples per second.

Now we need to find the right value for batch size, so that in one second, the number of trained samples is roughly the same as the number of samples generated by the actors. For example, if it takes 0.1 second for the learner to finish one batch, then the batch size can be computed as `333/0.1=33.3` or 32. So the train sample rate is `32 * (0.1 / 1) =320`. This gives us an well balanced train sample rate vs sample generation rate.

However, using GPUs to train one batch can be much faster, usually 0.01 second (or lesser) for a single batch. It does not make sense to use `3.3` as a batch size, so we also need to use train delay to adjust this imbalance. In this case with batch size set to 32, we can use a train delay of 0.1 second per batch.

In practice, we can often use a train sample rate that's 10-20x greater than the sample generation rate, since we're using experience replay to store large amount of samples, and we also argument the samples through random rotation and mirroring during training.


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
