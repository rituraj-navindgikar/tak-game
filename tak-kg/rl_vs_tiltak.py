"""
rl_vs_tiltak.py
Pit your RL agent (AlphaZero or PPO) against Tiltak.

Usage:
    python3 rl_vs_tiltak.py --model ../tak_checkpoints/best.pt --type az
    python3 rl_vs_tiltak.py --model ../ppo_checkpoints/iter_0260.pt --type ppo
    python3 rl_vs_tiltak.py --model ../tak_checkpoints/best.pt --type az --games 20 --first alternate
"""

import subprocess, argparse, os, time
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from alpha_zero_rl_model import TakEnv, ActionSpace, encode

np.random.seed(42)

# ── config ────────────────────────────────────────────────────────
MATCH_TYPE  = "alternate"
NUM_WORKERS = 8
N           = 5
MAX_GAME_MOVES = 200
OBS         = N * N * 5 + 4
HIDDEN      = 256

ASP = ActionSpace()

AGENT_NAMES = {
    "rl":     "RL Agent",
    "tiltak": "Tiltak (MCTS)"
}


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
        # use MCTS for AZ if you want, here using greedy for speed
        idx = int(torch.argmax(logits))
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
        idx = int(torch.argmax(logits.squeeze(0)))
        return ASP.decode(idx)


def load_rl_agent(model_path, model_type, device='cpu'):
    if model_type == 'az':
        net = TakNet().to(device)
    else:
        net = PPONet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    net.eval()
    # print(f"Loaded {model_type.upper()} agent from {model_path} -> {sum(p.numel() for p in net.parameters()):,} params")
    return net


# ── notation converters ──────────────────────────────────────────
def env_to_tiltak(move):
    move = move.strip()
    if move.startswith('F') and len(move) == 3:
        return move[1:]
    return move

def tiltak_to_env(move):
    move = move.strip()
    if len(move) == 2 and move[0].isalpha() and move[1].isdigit():
        return 'F' + move
    return move


# ── Tiltak wrapper ───────────────────────────────────────────────
class TiltakPlayer:
    def __init__(self, binary_path, size=5, movetime=1000):
        self.movetime = movetime; self.size = size
        self.proc = subprocess.Popen(
            [binary_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        self._send("tei")
        while True:
            line = self.proc.stdout.readline().decode().strip()
            if line == "teiok": break
        self._send(f"teinewgame {size}")

    def _send(self, cmd):
        self.proc.stdin.write((cmd + "\n").encode())
        self.proc.stdin.flush()

    def new_game(self):
        self._send(f"teinewgame {self.size}")

    def get_move(self, move_history, env, max_retries=5):
        if move_history:
            self._send(f"position startpos moves {' '.join(move_history)}")
        else:
            self._send("position startpos")
        retries = 0
        while True:
            self._send(f"go movetime {self.movetime}")
            while True:
                line = self.proc.stdout.readline().decode().strip()
                if line.startswith("bestmove"):
                    move = line.split()[1]; break
            if move.startswith('C') and len(move) == 3:
                retries += 1
                if retries >= max_retries:
                    import random
                    fallback = random.choice(env.legal_moves())
                    return env_to_tiltak(fallback), fallback
            else:
                return move, tiltak_to_env(move)

    def close(self):
        try: self._send("quit"); self.proc.terminate()
        except: pass


# ── play one game ────────────────────────────────────────────────
def play_game(rl_net, tiltak, rl_first=True, verbose=False):
    rl_player     = 0 if rl_first else 1
    tiltak_player = 1 - rl_player

    env  = TakEnv().reset()
    tiltak.new_game()
    move_history_tiltak = []

    if verbose:
        print("\n" + "="*55)
        label = "RL goes first" if rl_first else "Tiltak goes first"
        print(f"  RL (P{rl_player+1}) vs Tiltak (P{tiltak_player+1})  —  {label}")
        print("="*55)

    while not env.done:
        if env.turn == rl_player:
            move_env = rl_net.act(env)
            if verbose: print(f"  RL plays: {move_env}")
            _, done = env.step(move_env)
            move_history_tiltak.append(env_to_tiltak(move_env))
        else:
            result = tiltak.get_move(move_history_tiltak, env)
            if result is None:
                env.winner = rl_player; break
            move_tiltak, move_env = result
            if verbose: print(f"  Tiltak plays: {move_tiltak}")
            _, done = env.step(move_env)
            move_history_tiltak.append(move_tiltak)

    if not env.done: env.winner = 2
    w = env.winner
    if verbose:
        if w == rl_player:     print("  Result: RL wins!")
        elif w == tiltak_player: print("  Result: Tiltak wins!")
        else:                  print("  Result: Draw!")

    return w, rl_player, tiltak_player


# ── parallel worker ───────────────────────────────────────────────
def run_single_game(game_args):
    tiltak_binary, rl_first, movetime, model_path, model_type = game_args
    tiltak = TiltakPlayer(binary_path=tiltak_binary, movetime=movetime)
    rl_net = load_rl_agent(model_path, model_type)
    w, rp, tp = play_game(rl_net, tiltak, rl_first=rl_first, verbose=False)
    tiltak.close()
    return w, rp, tp


# ── plot ─────────────────────────────────────────────────────────
def plot_results(rl_wins, tiltak_wins, draws, total_games, match_type, rl_label, save_path):
    rl_pct  = 100 * rl_wins     / total_games
    tl_pct  = 100 * tiltak_wins / total_games
    dr_pct  = 100 * draws       / total_games

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    subtitle = {"minimax_first": f"{rl_label} goes first",
                "tiltak_first":  "Tiltak goes first",
                "alternate":     "First move alternates"}.get(match_type, "")
    fig.suptitle(f"{rl_label}  vs  Tiltak (MCTS)\n{subtitle}  |  {total_games} games",
                 fontsize=13, fontweight='bold', y=1.02)

    colors = ["#4C72B0", "#DD8452", "#8C8C8C"]
    ax = axes[0]; bh = 0.4
    ax.barh(0, rl_pct,  bh, color=colors[0], label=rl_label)
    ax.barh(0, tl_pct,  bh, left=rl_pct,          color=colors[1], label="Tiltak (MCTS)")
    ax.barh(0, dr_pct,  bh, left=rl_pct + tl_pct,  color=colors[2], label="Draw")
    for val, left in [(rl_pct, 0), (tl_pct, rl_pct), (dr_pct, rl_pct+tl_pct)]:
        if val > 5:
            ax.text(left + val/2, 0, f"{val:.1f}%", ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 100); ax.set_yticks([])
    ax.set_xlabel("Percentage of games (%)"); ax.set_title("Win/Draw Distribution")
    ax.legend(loc='upper right', fontsize=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)

    ax2 = axes[1]
    counts = [rl_wins, tiltak_wins, draws]
    x = np.arange(3)
    bars = ax2.bar(x, counts, width=0.5, color=colors, edgecolor='white')
    for bar, count, pct in zip(bars, counts, [rl_pct, tl_pct, dr_pct]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{count}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels([rl_label, "Tiltak", "Draw"], fontsize=10)
    ax2.set_ylabel("Number of games"); ax2.set_title("Win Counts")
    ax2.set_ylim(0, max(counts) * 1.3)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    type=str, required=True, help='Path to RL model checkpoint')
    parser.add_argument('--type',     type=str, default='az', choices=['az', 'ppo'],
                        help='Model type: az (AlphaZero) or ppo')
    parser.add_argument('--tiltak',   type=str, default='../other_models/tiltak/target/release/tei')
    parser.add_argument('--games',    type=int, default=20)
    parser.add_argument('--movetime', type=int, default=500)
    parser.add_argument('--first',    type=str, default=MATCH_TYPE,
                        choices=['rl_first', 'tiltak_first', 'alternate'])
    parser.add_argument('--quiet',    action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: model not found at {args.model}"); return
    if not os.path.exists(args.tiltak):
        print(f"ERROR: Tiltak not found at {args.tiltak}"); return

    rl_label = f"AlphaZero" if args.type == 'az' else "PPO"

    game_args = []
    for g in range(args.games):
        if args.first == 'rl_first':        rl_first = True
        elif args.first == 'tiltak_first':  rl_first = False
        else:                               rl_first = (g % 2 == 0)
        game_args.append((args.tiltak, rl_first, args.movetime, args.model, args.type))

    print(f"Running {args.games} games across {NUM_WORKERS} workers...")

    with Pool(processes=NUM_WORKERS) as pool:
        results = list(pool.map(run_single_game, game_args))

    rl_wins = tiltak_wins = draws = 0
    for g, (w, rp, tp) in enumerate(results):
        outcome = rl_label if w == rp else "Tiltak" if w == tp else "Draw"
        print(f"  Game {g+1}: {outcome}")
        if w == rp:   rl_wins     += 1
        elif w == tp: tiltak_wins += 1
        else:         draws       += 1

    total = rl_wins + tiltak_wins + draws
    print("\n" + "="*55)
    print(f"  {rl_label} wins : {rl_wins}  ({100*rl_wins/total:.0f}%)")
    print(f"  Tiltak wins    : {tiltak_wins}  ({100*tiltak_wins/total:.0f}%)")
    print(f"  Draws          : {draws}")
    print("="*55)

    plot_results(
        rl_wins=rl_wins, tiltak_wins=tiltak_wins, draws=draws,
        total_games=args.games, match_type=args.first,
        rl_label=rl_label,
        save_path=f"{rl_label.lower()}_vs_tiltak_{args.first}.png"
    )


if __name__ == "__main__":
    main()