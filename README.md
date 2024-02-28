# Alpha Zero

A PyTorch implementation of DeepMind's AlphaZero agent to play Go and Free-style Gomoku board game.
This project is part of my book [**The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python**](https://link.springer.com/book/10.1007/978-1-4842-9606-6)

# Content

- [Environment and Requirements](#environment-and-requirements)
- [Code Structure](#code-structure)
- [Author's Notes](#authors-notes)
- [Quick Start](#quick-start)
- [Train Agents](#train-agents)
- [Evaluate Agents](#evaluate-agents)
- [Limitations](#limitations)
- [References Code](#references-code)
- [License](#license)
- [Citing our work](#citing-our-work)

# Environment and Requirements

- Python 3.10.6
- PyTorch 1.13.1
- gym 0.25.2
- numpy 1.23.4

# Code Structure

- `alpha_zero` directory contains main source code for the project.

  - `core` directory contains core modules for AlphaZero like MCTS search algorithm, selfplay training pipeline, rating etc.

    - `mcts_v1.py` implements the naive implementation of MCTS search algorithm used by AlphaZero
    - `mcts_v2.py` implements the much faster (3x faster than mcts_v1.py) implementation of MCTS search algorithm used by AlphaZero, code adapted from the Minigo project
    - `pipeline.py` implements the core functions for AlphaZero training pipeline, where we can execute self-play actor, learner, and evaluator
    - `eval_dataset.py` implements the code to build an evaluation dataset using professional human play games in sgf format
    - `network.py` implements the neural network class
    - `rating.py` implements the code for compute elo ratings
    - `replay.py` implements the code for uniform random replay

  - `envs` directory contains modules for different board games environment, implemented in openAI Gym API.

    - `base.py` a basic board game environment implemented in Gym API
    - `coords.py` which contains the core logic for board game, code from the Minigo project
    - `go_engine.py` which contains the core logic and scoring functions for board game Go, code from the Minigo project
    - `go.py` implements the board game Go, which uses the core engine from `envs/go_engine.py`
    - `gomoku.py` implements freestyle Gomoku board game (a.k.a five in a row)
    - `gui.py` implements a very basic GUI program for board game

  - `utils` directory contains helper modules like logging, data transformation, sgf wrapper etc.

    - `sgf_wrapper.py` implements the code for reading and replaying Go game records saved as sgf files, code adapted from the Minigo project
    - `transformation.py` implements functions to perform random rotation and mirroring to the training samples

  - `eval_play` directory contains code to evaluate and play against the trained AlphaZero agent.

    - `eval_agent_go.py` contains the code to evaluate the trained agent on the game of Go using a very basic GUI program
    - `eval_agent_go_cmd.py` contains the code to evaluate the trained agent on the game of Go, if you prefer using terminal and (GTP) commands
    - `eval_agent_go_mass_matches.py` contains the code to asses the performance of different models (or checkpoints) by playing mass amount of matches on the game of Go
    - `eval_agent_gomoku.py` contains the code to evaluate the trained agent on freestyle Gomoku board game using a very basic GUI program
    - `eval_agent_gomoku_cmd.py` contains the code to evaluate the trained agent on freestyle Gomoku board game, if you prefer using terminal and (GTP) commands

  - `training_go.py` a driver program initialize the training session on a 9x9 Go board
  - `training_go_jumbo.py` a driver program initialize the training session on a 19x19 Go board, incorporating elements from the original configuration of AlphaZero. Be caution before running this module, as it demands powerful computational resources and is expected to consume a considerable amount of time, possibly weeks or even months.
  - `training_gomoku.py` a driver program initialize the training session on a 13x13 Gomoku board

- `plot_go.py` contains the code to plot training progress for game of Go
- `plot_gomoku.py` contains the code to plot training progress for Gomoku

# Author's Notes

- The goal is not to make a strongest player but to study the algorithm, as we stopped the training once the agent have made some progress.
- We use scaled down configuration for the training, like using a smaller board size, smaller neural network, lesser simulations per MCTS search etc.
- We can't guarantee it's bug free. So bug report and pull request are welcome.

# Quick Start

Please note that the code requires Python 3.10.6 or a higher version to run.

## Important Note:

- The code in this project was not designed to be executed on Jupyter or Google Colab notebooks due to its high computational demands.
- Running the evaluation modules such as 'eval_agent_go.py', on remote servers or Jupyter or Google Colab notebooks is not possible. This is because these environments are 'headless,' lacking a display to launching GUI programs.

Before you get started, ensure that you have the latest version of pip installed on your machine by executing the following command:

```
python3 -m pip install --upgrade pip setuptools

cd <repo_path_on_your_computer>

pip3 install -r requirements.txt

# Required to run the GUI client
apt-get install python3-tk
```

**Using PyTorch with GPUs:**
If you are utilizing Nvidia GPUs, it is highly recommended to install PyTorch with CUDA by following the instructions provided at https://pytorch.org/get-started/locally/.

# Train Agents

## Training AlphaZero on a 9x9 Go board

It is crucial to acknowledge that training an AlphaZero agent can often be a time-consuming process, requiring weeks or even months on a 19x19 Go board. In the original paper published by DeepMind, they mentioned that the original AlphaZero agent was trained over thousands of servers and TPUs. Even with such enormous computational power, it took 72 hours to train the 20 blocks version of AlphaZero, and 40 days for the 40 blocks version. Unfortunately, we do not have access to the same level of computational resources and budget to conduct similar experiments, and our goal is not to build the strongest agent capable of defeating the world champion. Instead, we have made adjustments such as using scaled-down settings (e.g., a smaller neural network, a simpler game with a 9x9 board size, fewer self-play actors, and fewer games played) to train the AlphaZero agent.

For our training, we utilized a single server equipped with 128 CPUs and 8 RTX 3090 GPUs to train the AlphaZero agent to play the game of Go on a 9x9 board. Upon trying different neural network architectures, we ended up chose one which consists of 11 blocks: 1 initial convolutional block + 10 residual blocks, each using 128 filters, and we use 128 unites for fully connected layer inside the value head. The training process took approximately 40 hours (corresponding to 150,000 training steps) to achieve a strong level of play. We trained the agent using over 1 million self-play games and evaluated its performance using 10,000 human-played games, total of 620,000 positions collected from the internet (mostly from CGOS). Additionally, we evaluated the agent by setting it play against the previous model from the last checkpoint. However, these evaluation games were not used for training, and we did not introduce noise to the root node of the Monte Carlo Tree Search (MCTS) search tree.

It is evident that the agent reached a strong level of play around 140,000 training steps, as indicated by the highest Elo rating achieved. However, beyond that point, the agent's performance starts to decline significantly, resulting in a sharp decrease in Elo rating. This decline could be attributed to various factors, such as the learning rate not being reduced quickly enough or the neural network being too powerful for the 9x9 board size. Unfortunately, due to budget constraints, we are unable to rerun the training process from scratch.

However, in the spirit of experimentation, we decided to set our best model (154000 training steps), to compete against other computer engine called CrazyStone (amateur 1 dan level). Due to limitations in our program's integration with third-party online playing services, we have to play the game using two separate GUI clients. In this setup, AlphaZero played as white using our GUI client from this repository, while simultaneously starting another game on mobile app where we played as white and allowed the opponent to play as black. Then we acted as a proxy, making the black moves in our local match and the white moves on mobile app. Our best model won 16 of 20 games, but it fails to score wins when playing against the 2 dan level CrazyStone opponent.

![9x9 Go training](/images/9x9_go_training_progress.png)

### Training on multiple GPUs

By default, the driver program attempts to distribute the self-play actors across all available GPUs in the machine. However, it is crucial to select an appropriate size for the number of actors.

Our findings indicate that a single RTX 3090 GPU, equipped with 24GB of vRAM, can accommodate approximately 30 actors, depending on the complexity of the neural network and the batch size used for training. Going beyond this number may result in the CUDA OUT OF RAM error.

There fore, it's recommended to start a trial run (like the example below) and let it run for a while to generate a few checkpoints. Ensure that everything runs smoothly, such as monitoring vRAM usage and GPU utilization, which should be around 90-95%.

```
# Test run to see GPU vRAM usage and utilization
nohup python3 -m alpha_zero.training_go --num_actors=116 --num_simulations=50 --min_games=100 --games_per_ckpt=100
```

### Analyze self-play games during training

The self-play actors will save the game in the Smart Game Format (SGF) file at regular intervals, while the evaluator will save the game against the previous model in SGF format as well. To review the game, you can open the SGF file using a standard SGF viewer like Sabaki.

When examining an SGF file, the game result property RE is interpreted in the following manner: if it begins with B+, it signifies that black has emerged victorious, and if it starts with W+, it means white has won. For instance, B+1.5 indicates that black has won by 1.5 points. In special cases, B+R or W+R denotes a win by resignation, implying that the opposing player has resigned. To maintain consistency, all game results in the CSV log files also adhere to this notation.

### Resume training

If your training session gets interrupted, such as due to a power outage, you can easily resume it. To do so, you need to provide the location of the last checkpoint and its corresponding Elo rating. You can find the Elo rating for the specific checkpoint in the evaluation.csv file. Furthermore, if you have saved the replay state through option `save_replay_interval`, you can also load the games from the latest replay state to continue training seamlessly.

Here's an example of how to resume training from where we left.

```
python3 -m alpha_zero.training_go --load_ckpt=./checkpoints/go/9x9/training_steps_120000.ckpt --default_rating=350 --load_replay=./checkpoints/go/9x9/replay_state.ckpt
```

## Training AlphaZero on a 13x13 Gomoku board

We offer code for training the AlphaZero agent on freestyle Gomoku, a simpler game that demands fewer computational resources. During our experiment, we trained the AlphaZero agent to excel in 13x13 freestyle Gomoku using a server equipped with 128 CPUs and 8 RTX 3090 GPUs. This training process lasted around 16 hours (200k training steps), and the agent reached a strong level of play at approximately 150k training steps.

![13x13 Gomoku training](/images/13x13_gomoku_training_progress.png)

If you are working with more limited resources, such as a workstation with a single GPU, you can starting with a smaller board size like 9x9 for Gomoku. By running a minimum of 32 actors in parallel, you should be able to observe the agent reaching a strong level of play within a day or two.

```
python3 -m alpha_zero.training_gomoku --board_size=9 --num_stack=4 --num_res_blocks=9 --num_filters=32 --num_fc_units=64 --num_simulations=200 --ckpt_dir=./checkpoints/gomoku/9x9 --logs_dir=./logs/gomoku/9x9 --save_sgf_dir=./games/selfplay_games/gomoku/9x9
```

Training on Gomoku is considerably faster compared to Go, not only because Gomoku is approximately 10 times simpler than Go in terms of complexity, but also due to the faster execution time of the code. In Gomoku, the rules are straightforward: a player can place a stone on an intersection as long as it is empty. However, this simplicity does not apply to Go. In Go, the legality of moves constantly changes step-by-step, making it impossible to construct legal moves solely by counting empty spaces on the board. Additional factors such as KO situations, suicidal moves, and potential captures further complicate the code. Consequently, the code for Go is expected to be 5-10 times slower than that of Gomoku, even with the use of optimized MCTS (Monte Carlo Tree Search) code.

### Edge cases for Gomoku

In Gomoku, the standard neural network architecture of AlphaZero agent faces a limitation despite its strong performance in regular scenarios. It struggles to effectively block the opponent in specific edge cases, such as when the opponent, like a human player, attempts to connect five stones at the edge of the game board during the opening moves (for example, the opponent plays A3,A4,A5,A6,A7, or D13,E13,F13,G13,H13 on a 13x13 board). To overcome this challenge, we implemented a modification by increasing the padding for the neural network from 1 to 3 in the initial convolution block. This adjustment has proven to be effective in resolving the edge cases in Gomoku.

### Gomoku with larger board size

A crucial point to note regarding freestyle Gomoku is that, for larger board sizes like 13x13 or 15x15, the first player (black) can always win if they consistently make optimal moves. This outcome assumes no additional rules are in place, as is the case in our example code. Therefore, it is not surprising to observe the evaluation game typically ends within 20-30 steps. Once the agent has achieved a notably strong level of play, you may consider halting further training, as it has likely reached a highly proficient state.

### Analyzing Gomoku games using SGF viewer

When reviewing SGF files for Gomoku, please note that most SGF viewer programs were designed for Go, so they may enforce rules like detecting suicidal moves and capturing stones, which do not exist in Gomoku. Therefore, you may encounter warning messages such as "illegal move" or see stones disappearing from the board. These are not errors or bugs, they occur because the SGF viewer treats the game as if it were for Go.

# Evaluate Agents

You can evaluate the agent by running the `eval_agent` scripts, we have a very basic GUI program (not optimized for nice UI), which supports

- Human (black) vs. AlphaZero (white)
- AlphaZero vs. AlphaZero

To start play the game of 9x9 Go using the GUI program, run the following command

```
# Human (black) vs. AlphaZero (white)
python3 -m eval_play.eval_agent_go

# AlphaZero vs. AlphaZero
python3 -m eval_play.eval_agent_go --nohuman_vs_ai
```

To start play the game of 13x13 Gomoku using the GUI program, run the following command

```
# Human (black) vs. AlphaZero (white)
python3 -m eval_play.eval_agent_gomoku

# AlphaZero vs. AlphaZero
python3 -m eval_play.eval_agent_gomoku --nohuman_vs_ai
```

In addition, we also have very simple script to run the interactive game in the terminal, incase your machine don't have a GUI environment.
To start play the game of 9x9 Go using the terminal, run the following command

```
# Human (black) vs. AlphaZero (white)
python3 -m eval_play.eval_agent_go_cmd

# AlphaZero vs. AlphaZero
python3 -m eval_play.eval_agent_go_cmd --nohuman_vs_ai
```

To start play the game of 13x13 Gomoku using the terminal, run the following command

```
# Human (black) vs. AlphaZero (white)
python3 -m eval_play.eval_agent_gomoku_cmd

# AlphaZero vs. AlphaZero
python3 -m eval_play.eval_agent_gomoku_cmd --nohuman_vs_ai
```

# Limitations

- The scoring function implemented in `envs.go_engine.py` for the game of Go is a relatively straightforward system that follows the Chinese area scoring rules. However, it occasionally produces different game results when compared to other Go engines or programs. The root cause of the issue is the inability to accurately detect dead stones, which is a major challenge for computer programs when computing the scores at the end of a game. Consequently, this issue will have negative affects on the agent's performance. Unfortunately, we have yet to discover a satisfactory solution to this issue.
- The rules implemented in `envs.go_engine.py` for the game of Go do not support super ko, only basic ko. Basic ko checks the last position, but not the last N positions, which can result in repeated cycles and potentially impact the overall performance of the agent.
- The basic GUI client implemented in `envs.gui.py` may not have a nice looking UI, and may not offer a consistent UI across different OS platforms, especially on Windows. This is due to its limited ability to handle varying screen resolutions and scaling. However, the core features of the client should function properly.
- The basic GUI client implemented in `envs.gui.py` may occasionally hang if you're playing against the agent and clicking too fast or making a large number of clicks in a short amount of time. In such cases, the only resolution is to start a new game or relaunch the GUI client.

## Acknowledgments and References Code

- [A Deep Dive into Monte Carlo Tree Search](https://www.moderndescartes.com/essays/deep_dive_mcts/)
- [Minigo: A minimalist Go engine modeled after AlphaGo Zero](https://github.com/tensorflow/minigo)

# License

This project is licensed under the MIT License, see the LICENSE file for details

# Citing our work

If you reference or use our project in your research, please cite our work:

```bibtex
@book{the_art_of_rl2023,
  author    = {Michael Hu},
  title     = {The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python},
  year      = {2023},
  publisher = {Apress},
  url       = {https://link.springer.com/book/10.1007/978-1-4842-9606-6}
}
```

```bibtex
@software{alpha_zero2022github,
  title = {{Alpha Zero}: A PyTorch implementation of DeepMind's AlphaZero agent},
  author = {Michael Hu},
  url = {https://github.com/michaelnny/alpha_zero},
  version = {2.0.0},
  year = {2022},
}
```
