# Tak 3x3 — AlphaZero RL Agent

A 3×3 Tak game with a trained AlphaZero-style RL agent, playable via a browser UI.

No capstones. Flats and walls only.

---

## What is Tak?

Tak is a two-player abstract strategy game where the goal is to build a connected road of your pieces across the board — either left to right or bottom to top. Players can place flat stones, place walls (which block roads), and move stacks of pieces. First player to complete a road wins. If no road is built and pieces run out, the player with the most flat stones on top wins.

---

## Project Structure

```
tak-kg/
├── human_vs_rl.py              # Flask web server, hooks RL agent to UI vs human
├── minmax_vs_rl.py              # minmax,vs RL agent
├── alpha_zero_rl_model.py      # TakEnv, ActionSpace, TakNet, MCTS
├── Game.py                     # Game logic (rendering + move validation)
├── Board.py                    # Tkinter board renderer (used by Game.py)
├── templates/
│   └── index.html              # Browser UI
├── checkpoints/
│   └── best.pt                 # Trained model weights
└── alpha-zero-rl-model.ipynb   # Training notebook (run on Colab)
```

---

## Setup

```bash
pip install flask flask-cors torch numpy matplotlib
```

### Compile the minimax AI

```bash
bash compile.sh
```

This compiles `minimax.cpp` into a `./minimax` binary using:

```bash
g++ -std=c++11 -O2 -o minimax minimax.cpp
```

The `-O2` flag enables compiler optimization — makes the minimax search noticeably faster. Required for `rl_vs_minimax.py` and the original client/server setup.

---

## Run the Human vs RL Web UI

```bash
python3 huamn_vs_rl.py
```

Open `http://localhost:5000` in your browser. Hit **New Game** and start playing.

**Move format:**

| Move       | Meaning                                          |
| ---------- | ------------------------------------------------ |
| `Fa1`    | Place flat on a1                                 |
| `Sa1`    | Place wall on a1                                 |
| `2a1>11` | Move stack of 2 from a1 rightward, drop 1 then 1 |

Board columns are `a b c`, rows are `1 2 3`.

```
3   a3  b3  c3
2   a2  b2  c2
1   a1  b1  c1
```

**First move rule:** Your first move places the opponent's piece (standard Tak rule).

## Run the Minimax agent vs RL

```bash
python3 minmax_vs_rl.py --games 20 --quiet
```


---

## The RL Agent

The agent uses a simplified AlphaZero architecture:

* **Self-play** — two copies of the network play each other to generate training data
* **MCTS** — Monte Carlo Tree Search guided by the network's policy and value heads
* **Network** — small 3-layer MLP (128 hidden units), policy head + value head
* **Training loop** — self-play → train → evaluate vs previous best → keep better network

### State encoding (49 floats)

* 5 features per square × 9 squares = 45 values (top owner, top type, stack height, ownership fractions)
* 4 global features (current turn, flats remaining for each player, move count)

### Action space (108 actions)

* Place flat: 9 squares
* Place wall: 9 squares
* Stack moves: all legal pick-up and drop distributions in 4 directions

---

## Retrain from Scratch

1. Open `alpha-zero-rl-model.ipynb` in Google Colab
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run all cells top to bottom
4. Download `checkpoints/best.pt` when training finishes
5. Place it in `tak-kg/checkpoints/best.pt`

### Training config (default)

```python
ITERATIONS  = 20      # training iterations
GAMES       = 100     # self-play games per iteration
SIMS        = 50      # MCTS simulations per move
EPOCHS      = 15      # gradient epochs per iteration
HIDDEN      = 128     # network hidden size — don't change between runs
LR          = 1e-3    # learning rate (use 3e-4 for fine-tuning)
```

### Fine-tune existing model

```python
net.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
LR = 3e-4   # lower LR for fine-tuning
```

---

## Swap in a Different RL Model

`app.py` only needs three things from whatever model you use:

```python
# 1. environment
rl_env = TakEnv().reset()

# 2. agent with best_move interface
agent = YourAgent(...)
ai_move, _ = agent.best_move(rl_env, temp=0)

# 3. loadable weights
net.load_state_dict(torch.load("checkpoints/best.pt"))
```

As long as your model exposes these, `app.py`, `index.html`, `Game.py` and `Board.py` stay untouched. Only the notebook and `alpha_zero_rl_model.py` change.

---

## Requirements

* Python 3.10+
* PyTorch
* Flask, flask-cors
* NumPy
* Matplotlib (for training plots)
* Google Colab (recommended for training, free T4 or Pro H100)
