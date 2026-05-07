# Learning to Play Tak — AlphaZero vs PPO

Two reinforcement learning agents trained from scratch on Tak, a two-player abstract strategy board game. Implemented as part of CS 5180 Reinforcement Learning at Northeastern University.

No capstones. Flats and walls only. 5×5 board.

---

## What is Tak?

Tak is a two-player strategy game where the goal is to build a connected road of your flat stones across the board, either left to right or bottom to top. Players can place flat stones, place walls (which block roads), and move stacks of pieces. First player to complete a road wins. If pieces run out before a road is built, the player with the most flat stones on top wins.

---

## Results

| Matchup | Games | Agent 1 Win% | Agent 2 Win% |
|---|---|---|---|
| AlphaZero vs Random | 1000 | 78.8% | 20.3% |
| PPO vs Random | 1000 | 70.8% | 28.9% |
| AlphaZero vs Minimax | 1000 | 0% | 100% |
| PPO vs Minimax | 1000 | 0% | 100% |
| AlphaZero vs Tiltak | 1000 | 30.4% | 69.6% |
| PPO vs Tiltak | 1000 | 5.9% | 94.1% |
| Minimax vs Tiltak | 1000 | 78.3% | 21.7% |

AlphaZero with 1000 MCTS simulations defeats Minimax in nearly every game.

---

## Project Structure

```
tak-kg/
├── alpha_zero_rl_model.py      # TakEnv, ActionSpace, TakNet, MCTS
├── train_alpha_zero_rl_model.py # AlphaZero training loop
├── train_ppo_rl_model.py        # PPO training loop
├── versus.py                    # Flask web server — human vs AlphaZero
├── rl_vs_minimax.py             # Evaluate RL agent vs Minimax
├── rl_vs_tiltak.py              # Evaluate RL agent vs Tiltak
├── rl_vs_random.py              # Evaluate RL agent vs Random
├── rl_vs_rl.py                  # Evaluate AlphaZero vs PPO
├── minimax_vs_tiltak.py         # Calibration: Minimax vs Tiltak
├── minimax.cpp                  # Alpha-Beta Minimax implementation
├── compile.sh                   # Compiles minimax binary
├── Game.py                      # Game logic and move validation
├── templates/
│   └── index.html               # Browser UI with drag-and-drop
├── tak_checkpoints/
│   └── best.pt                  # Trained AlphaZero weights
└── ppo_checkpoints/
    └── best.pt                  # Trained PPO weights
```

---

## Setup

```bash
pip install flask flask-cors torch numpy matplotlib
```

### Compile Minimax

```bash
bash compile.sh
```

Required for any evaluation against the Minimax agent.

---

## Play Against the Agent

```bash
python3 versus.py
```

Open `http://localhost:5000`. Hit **New Game** and start playing.

Supports drag-and-drop piece placement and text input for stack moves.

**Move format:**

| Move | Meaning |
|---|---|
| `Fa3` | Place flat on a3 |
| `Sa3` | Place wall on a3 |
| `2a3>11` | Pick up 2 from a3, move right, drop 1 then 1 |

Columns: `a b c d e` — Rows: `1 2 3 4 5`

**First move rule:** Your first move places the opponent's piece (standard Tak opening rule).

---

## Evaluate Agents

### AlphaZero vs Random
```bash
python3 rl_vs_random.py --model tak_checkpoints/best.pt --type az --games 100 --first alternate --quiet
```

### AlphaZero vs Minimax
```bash
python3 rl_vs_minimax.py --model tak_checkpoints/best.pt --type az --sims 1000 --games 20 --first alternate --quiet
```

### AlphaZero vs Tiltak
```bash
python3 rl_vs_tiltak.py --model tak_checkpoints/best.pt --type az --sims 1000 --games 20 --first alternate
```

### PPO vs Random
```bash
python3 rl_vs_random.py --model ppo_checkpoints/best.pt --type ppo --games 100 --first alternate --quiet
```

### AlphaZero vs PPO
```bash
python3 rl_vs_rl.py --az tak_checkpoints/best.pt --ppo ppo_checkpoints/best.pt --games 20 --first alternate --quiet
```

---

## Train from Scratch

### AlphaZero
```bash
python3 train_alpha_zero_rl_model.py
```

Or on a SLURM cluster (Specifically for Northeastern's HPC Cluster):
```bash
sbatch train.sh
```

### PPO
```bash
python3 train_ppo_rl_model.py
```

### AlphaZero Training Config

```python
ITERATIONS  = 100     # training iterations
GAMES       = 100     # self-play games per iteration
SIMS        = 50      # MCTS simulations per move
EPOCHS      = 20      # gradient epochs per iteration
HIDDEN      = 256     # network hidden size
LR          = 1e-4    # learning rate
```

### PPO Training Config

```python
ITERATIONS     = 300   # PPO update cycles
GAMES_PER_ITER = 128   # self-play games per iteration
EPOCHS         = 16    # PPO epochs per iteration
HIDDEN         = 256
LR             = 1e-4
GAMMA          = 0.995
GAE_LAMBDA     = 0.97
CLIP_EPS       = 0.15
ENT_COEF       = 0.005
```

---

## Agents

### AlphaZero
- 3-layer MLP, hidden size 256, LayerNorm + ReLU
- Policy head + value head (Tanh)
- MCTS with PUCT selection, C_puct=1.5, Dirichlet noise at root
- Self-play with road progress reward shaping: `r = 0.9 * outcome + 0.1 * delta_road`
- Trained on Northeastern Explorer HPC cluster (16 CPUs, V100)

### PPO
- Same network architecture, shared trunk
- GAE advantage estimation
- Clipped surrogate objective, entropy bonus
- No tree search at inference — greedy argmax over masked logits

### State Encoding (129 floats)
- 5 features per cell × 25 cells: top stone owner, top stone type, stack height, P0 fraction, P1 fraction
- 4 global features: current player, P0 flats remaining, P1 flats remaining, move count

### Action Space (400+ actions)
- Place flat: 25 squares
- Place wall: 25 squares
- Stack moves: all legal pick-up and drop sequences in 4 directions

---

## Requirements

- Python 3.10+
- PyTorch
- Flask, flask-cors
- NumPy
- Matplotlib
- g++ (for minimax compilation)
- Tiltak binary (for Tiltak evaluation) — build from [Tiltak repo](https://github.com/MortenLohne/tiltak.git)
