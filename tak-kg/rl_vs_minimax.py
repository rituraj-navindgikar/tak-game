"""
rl_vs_minimax.py
Pit the trained RL agent (AlphaZero or PPO) against the minimax AI.

Usage:
    python3 rl_vs_minimax.py --model ../tak_checkpoints/best.pt --type az
    python3 rl_vs_minimax.py --model ../ppo_checkpoints/iter_0260.pt --type ppo
    python3 rl_vs_minimax.py --model ../tak_checkpoints/best.pt --type az --games 20 --first alternate --quiet
"""

import subprocess, argparse, os, select
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from alpha_zero_rl_model import TakEnv, ActionSpace, encode, MCTS

N          = 5
NUM_WORKERS = 8
OBS        = N * N * 5 + 4
HIDDEN     = 256
MAX_GAME_MOVES = 200

ASP = ActionSpace()


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
    def predict(self, env, device='cpu'):
        self.eval()
        x      = torch.tensor(encode(env), dtype=torch.float32).unsqueeze(0).to(device)
        logits, v = self.forward(x)
        logits = logits.squeeze(0)
        mask   = torch.tensor(ASP.mask(env), dtype=torch.bool, device=device)
        logits[~mask] = float('-inf')
        probs  = F.softmax(logits, dim=0).cpu().numpy()
        return probs, float(v.item())

    @torch.no_grad()
    def act(self, env, device='cpu'):
        self.eval()
        x      = torch.tensor(encode(env), dtype=torch.float32).unsqueeze(0).to(device)
        logits, _ = self.forward(x)
        logits = logits.squeeze(0)
        mask   = torch.tensor(ASP.mask(env), dtype=torch.bool, device=device)
        logits[~mask] = float('-inf')
        return ASP.decode(int(torch.argmax(logits)))


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
        return ASP.decode(int(torch.argmax(logits.squeeze(0))))


def load_rl_agent(model_path, model_type, n_sims=50, device='cpu'):
    if model_type == 'az':
        net = TakNet().to(device)
        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        net.eval()
        return MCTS(net, n_sims=n_sims, device=device)
    else:
        net = PPONet().to(device)
        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        net.eval()
        return net


# ── Minimax wrapper ──────────────────────────────────────────────
class MinimaxPlayer:
    def __init__(self, player_id, n=5, time_limit=1):
        self.proc = subprocess.Popen(
            ["./minimax"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )
        self.proc.stdin.write(f"{player_id} {n} {time_limit}\n".encode())
        self.proc.stdin.flush()

    def send_move(self, move):
        self.proc.stdin.write((move + "\n").encode())
        self.proc.stdin.flush()

    def get_move(self, timeout=15):
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if ready:
            return self.proc.stdout.readline().decode("utf-8", errors="replace").strip()
        return None

    def close(self):
        try: self.proc.kill()
        except: pass


# ── play one game ────────────────────────────────────────────────
def play_game(rl_agent, model_type, rl_player, verbose=True):
    minimax_player = 1 - rl_player
    minimax_id     = minimax_player + 1
    env     = TakEnv().reset()
    minimax = MinimaxPlayer(player_id=minimax_id, n=N, time_limit=1)

    if verbose:
        label = "RL goes first" if rl_player == 0 else "Minimax goes first"
        print(f"\n{'='*50}\n  {label}\n{'='*50}")

    while not env.done:
        if env.turn == rl_player:
            if model_type == 'az':
                move, _ = rl_agent.best_move(env, temp=0)
            else:
                move = rl_agent.act(env)
            if verbose: print(f"  RL plays: {move}")
            _, done = env.step(move)
            minimax.send_move(move)
        else:
            move = minimax.get_move(timeout=15)
            if not move:
                env.winner = rl_player; break
            if verbose: print(f"  Minimax plays: {move}")
            _, done = env.step(move)

    minimax.close()
    w = env.winner
    if w == -1: w = 2
    if verbose:
        if w == rl_player:      print("  RL wins!")
        elif w == minimax_player: print("  Minimax wins!")
        else:                   print("  Draw!")
    return w, rl_player


# ── parallel worker ───────────────────────────────────────────────
def run_single_game(game_args):
    model_path, model_type, rl_player, n_sims = game_args
    agent = load_rl_agent(model_path, model_type, n_sims=n_sims)
    w, rp = play_game(agent, model_type, rl_player=rl_player, verbose=True)
    return w, rp


# ── plot ─────────────────────────────────────────────────────────
def plot_results(rl_wins, minimax_wins, draws, total_games, rl_label, first_mode, save_path):
    rl_pct  = 100 * rl_wins     / total_games
    mm_pct  = 100 * minimax_wins / total_games
    dr_pct  = 100 * draws        / total_games

    subtitle = {"minimax": "Minimax goes first", "rl": "RL goes first",
                "alternate": "First move alternates"}.get(first_mode, "")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{rl_label}  vs  Minimax (Alpha-Beta)\n{subtitle}  |  {total_games} games",
                 fontsize=13, fontweight='bold', y=1.02)

    colors = ["#4C72B0", "#DD8452", "#8C8C8C"]
    ax = axes[0]; bh = 0.4
    ax.barh(0, rl_pct,  bh, color=colors[0], label=rl_label)
    ax.barh(0, mm_pct,  bh, left=rl_pct,          color=colors[1], label="Minimax")
    ax.barh(0, dr_pct,  bh, left=rl_pct + mm_pct,  color=colors[2], label="Draw")
    for val, left in [(rl_pct, 0), (mm_pct, rl_pct), (dr_pct, rl_pct+mm_pct)]:
        if val > 5:
            ax.text(left + val/2, 0, f"{val:.1f}%", ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 100); ax.set_yticks([])
    ax.set_xlabel("Percentage of games (%)"); ax.set_title("Win/Draw Distribution")
    ax.legend(loc='upper right', fontsize=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)

    ax2 = axes[1]
    counts = [rl_wins, minimax_wins, draws]
    x = np.arange(3)
    bars = ax2.bar(x, counts, width=0.5, color=colors, edgecolor='white')
    for bar, count, pct in zip(bars, counts, [rl_pct, mm_pct, dr_pct]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{count}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels([rl_label, "Minimax", "Draw"], fontsize=10)
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
    parser.add_argument('--model',  type=str, required=True)
    parser.add_argument('--type',   type=str, default='az', choices=['az', 'ppo'])
    parser.add_argument('--games',  type=int, default=20)
    parser.add_argument('--sims',   type=int, default=50,
                        help='MCTS sims for AlphaZero (ignored for PPO)')
    parser.add_argument('--first',  type=str, default='alternate',
                        choices=['minimax', 'rl', 'alternate'])
    parser.add_argument('--quiet',  action='store_true')
    args = parser.parse_args()

    if not os.path.exists('./minimax'):
        print("ERROR: ./minimax not found. Run: bash compile.sh"); return
    if not os.path.exists(args.model):
        print(f"ERROR: model not found at {args.model}"); return

    rl_label = "AlphaZero" if args.type == 'az' else "PPO"

    game_args = []
    for g in range(args.games):
        if args.first == 'minimax':   rl_player = 1
        elif args.first == 'rl':      rl_player = 0
        else:                         rl_player = g % 2
        game_args.append((args.model, args.type, rl_player, args.sims))

    print(f"Running {args.games} games across {NUM_WORKERS} workers...")

    with Pool(processes=NUM_WORKERS) as pool:
        results = list(pool.map(run_single_game, game_args))

    rl_wins = minimax_wins = draws = 0
    for g, (w, rp) in enumerate(results):
        outcome = rl_label if w == rp else "Minimax" if w == 1 - rp else "Draw"
        print(f"  Game {g+1}: {outcome}")
        if w == rp:       rl_wins      += 1
        elif w == 1 - rp: minimax_wins += 1
        else:             draws        += 1

    total = rl_wins + minimax_wins + draws
    print("\n" + "="*50)
    print(f"  {rl_label} wins  : {rl_wins}  ({100*rl_wins/total:.0f}%)")
    print(f"  Minimax wins    : {minimax_wins}  ({100*minimax_wins/total:.0f}%)")
    print(f"  Draws           : {draws}")
    print("="*50)

    plot_results(
        rl_wins=rl_wins, minimax_wins=minimax_wins, draws=draws,
        total_games=args.games, rl_label=rl_label,
        first_mode=args.first,
        save_path=f"{rl_label.lower()}_vs_minimax_{args.first}.png"
    )


if __name__ == "__main__":
    main()