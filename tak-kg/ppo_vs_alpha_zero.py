"""
compare_agents.py
Evaluates PPO and AlphaZero checkpoints over time and plots win rate vs iteration.

Usage:
    python3 compare_agents.py
"""

import os, glob, torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from alpha_zero_rl_model import TakEnv, ActionSpace, encode, HIDDEN

# ── load both model classes ───────────────────────────────────────
ASP = ActionSpace()
OBS = 129  # your OBS value

N            = 5
MAX_GAME_MOVES = 200
EPS          = 1e-8

# ── AlphaZero network ────────────────────────────────────────────
class TakNet(nn.Module):
    def __init__(self, obs=OBS, actions=None, h=HIDDEN):
        super().__init__()
        if actions is None: actions = ASP.size
        self.trunk = nn.Sequential(
            nn.Linear(obs, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h),   nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h),   nn.LayerNorm(h), nn.ReLU(),
        )
        self.pi = nn.Sequential(nn.Linear(h, h//2), nn.ReLU(), nn.Linear(h//2, actions))
        self.v  = nn.Sequential(nn.Linear(h, h//2), nn.ReLU(), nn.Linear(h//2, 1), nn.Tanh())

    def forward(self, x):
        h = self.trunk(x); return self.pi(h), self.v(h)

    @torch.no_grad()
    def act(self, env, device='cpu'):
        self.eval()
        x      = torch.tensor(encode(env), dtype=torch.float32).unsqueeze(0).to(device)
        logits, _ = self.forward(x)
        logits = logits.squeeze(0)
        mask   = torch.tensor(ASP.mask(env), dtype=torch.bool, device=device)
        logits[~mask] = float('-inf')
        idx    = int(torch.argmax(logits))  # greedy at eval time
        return ASP.decode(idx)


# ── PPO network ───────────────────────────────────────────────────
class PPONet(nn.Module):
    def __init__(self, obs=OBS, actions=None, h=HIDDEN):
        super().__init__()
        if actions is None: actions = ASP.size
        self.trunk = nn.Sequential(
            nn.Linear(obs, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h),   nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h),   nn.LayerNorm(h), nn.ReLU(),
        )
        self.pi = nn.Sequential(nn.Linear(h, h//2), nn.ReLU(), nn.Linear(h//2, actions))
        self.v  = nn.Sequential(nn.Linear(h, h//2), nn.ReLU(), nn.Linear(h//2, 1))

    def forward(self, x, mask=None):
        h = self.trunk(x)
        logits = self.pi(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        return logits, self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, env, device='cpu'):
        self.eval()
        x    = torch.tensor(encode(env), dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.tensor(ASP.mask(env), dtype=torch.bool, device=device).unsqueeze(0)
        logits, _ = self.forward(x, mask)
        idx  = int(torch.argmax(logits.squeeze(0)))  # greedy at eval time
        return ASP.decode(idx)


# ── play one game between two agents ─────────────────────────────
def play_game(agent1, agent2, agent1_first=True):
    env = TakEnv().reset()
    p1  = 0 if agent1_first else 1
    p2  = 1 - p1
    agents = {p1: agent1, p2: agent2}

    for _ in range(MAX_GAME_MOVES):
        ag   = agents[env.turn]
        move = ag.act(env)
        _, done = env.step(move)
        if done: break

    if not env.done: env.winner = 2
    return env.winner, p1, p2


# ── evaluate two agents over N games ─────────────────────────────
def evaluate(agent1, agent2, n_games=20):
    a1_wins = a2_wins = draws = 0
    for i in range(n_games):
        agent1_first = (i % 2 == 0)
        w, p1, p2 = play_game(agent1, agent2, agent1_first)
        if w == p1:   a1_wins += 1
        elif w == p2: a2_wins += 1
        else:         draws   += 1
    total = a1_wins + a2_wins + draws
    return a1_wins / total, a2_wins / total, draws / total


# ── load checkpoints ─────────────────────────────────────────────
def load_az_checkpoint(path, device='cpu'):
    net = TakNet().to(device)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    net.eval()
    return net

def load_ppo_checkpoint(path, device='cpu'):
    net = PPONet().to(device)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    net.eval()
    return net


# ── main ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cpu'  # use cpu for eval, fast enough with greedy

    AZ_CKPT_DIR  = "../tak_checkpoints"
    PPO_CKPT_DIR = "../ppo_checkpoints"
    N_EVAL_GAMES = 20  # games per matchup, increase for more reliable results

    # find all checkpoints sorted by iteration
    az_ckpts  = sorted(glob.glob(os.path.join(AZ_CKPT_DIR,  "iter_*.pt")))
    ppo_ckpts = sorted(glob.glob(os.path.join(PPO_CKPT_DIR, "iter_*.pt")))

    print(f"Found {len(az_ckpts)} AlphaZero checkpoints")
    print(f"Found {len(ppo_ckpts)} PPO checkpoints")

    # ── PPO win rate over its own iterations (vs random baseline) ─
    # use the best PPO as fixed opponent to measure AZ progress
    best_ppo = load_ppo_checkpoint(os.path.join(PPO_CKPT_DIR, "best.pt"), device)
    best_az  = load_az_checkpoint(os.path.join(AZ_CKPT_DIR,  "best.pt"), device)

    # ── eval AZ checkpoints vs best PPO ──────────────────────────
    az_iters = []
    az_winrates_vs_ppo = []

    for ckpt in az_ckpts:
        it = int(os.path.basename(ckpt).replace("iter_", "").replace(".pt", ""))
        print(f"Evaluating AZ iter {it} vs best PPO...")
        az_net = load_az_checkpoint(ckpt, device)
        wr, _, _ = evaluate(az_net, best_ppo, N_EVAL_GAMES)
        az_iters.append(it)
        az_winrates_vs_ppo.append(wr * 100)

    # ── eval PPO checkpoints vs best AZ ──────────────────────────
    ppo_iters = []
    ppo_winrates_vs_az = []

    for ckpt in ppo_ckpts:
        it = int(os.path.basename(ckpt).replace("iter_", "").replace(".pt", ""))
        print(f"Evaluating PPO iter {it} vs best AZ...")
        ppo_net = load_ppo_checkpoint(ckpt, device)
        wr, _, _ = evaluate(ppo_net, best_az, N_EVAL_GAMES)
        ppo_iters.append(it)
        ppo_winrates_vs_az.append(wr * 100)

    # ── head to head: best AZ vs best PPO ────────────────────────
    print("\nRunning head-to-head: best AZ vs best PPO...")
    az_wr, ppo_wr, draw_wr = evaluate(best_az, best_ppo, 40)
    print(f"  AlphaZero wins: {az_wr*100:.1f}%")
    print(f"  PPO wins:       {ppo_wr*100:.1f}%")
    print(f"  Draws:          {draw_wr*100:.1f}%")

    # ── plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AlphaZero vs PPO: Training Progression", fontsize=14, fontweight='bold')

    # left: win rates over iterations
    ax = axes[0]
    if az_iters:
        ax.plot(az_iters, az_winrates_vs_ppo, 'o-', color='steelblue',
                label='AlphaZero win rate\n(vs best PPO)', linewidth=2, markersize=6)
    if ppo_iters:
        ax.plot(ppo_iters, ppo_winrates_vs_az, 's--', color='coral',
                label='PPO win rate\n(vs best AZ)', linewidth=2, markersize=6)
    ax.axhline(50, color='gray', linestyle=':', linewidth=1, label='50% baseline')
    ax.set_xlabel("Training Iteration", fontsize=11)
    ax.set_ylabel("Win Rate (%)", fontsize=11)
    ax.set_title("Win Rate vs Iterations", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    # right: head to head bar chart
    ax2 = axes[1]
    labels  = ['AlphaZero', 'PPO', 'Draw']
    values  = [az_wr * 100, ppo_wr * 100, draw_wr * 100]
    colors  = ['steelblue', 'coral', 'gray']
    bars    = ax2.bar(labels, values, color=colors, width=0.5, edgecolor='white')
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{val:.1f}%", ha='center', va='bottom', fontsize=11)
    ax2.set_ylabel("Win Rate (%)", fontsize=11)
    ax2.set_title("Head-to-Head: Best Checkpoints\n(40 games, alternating first)", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig("az_vs_ppo_comparison.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to az_vs_ppo_comparison.png")