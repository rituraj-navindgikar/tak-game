"""
minimax_vs_tiltak.py
Pit your local minimax agent against Tiltak (Rust MCTS bot).

Usage:
    python3 minimax_vs_tiltak.py
    python3 minimax_vs_tiltak.py --games 10 --first tiltak
    python3 minimax_vs_tiltak.py --games 10 --first alternate --quiet
"""

import subprocess, argparse, select, time, os
from multiprocessing import Pool
from alpha_zero_rl_model import TakEnv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

np.random.seed(1234)

# ── config ────────────────────────────────────────────────────────
# Change this to one of: "minimax_first", "tiltak_first", "alternate"
MATCH_TYPE = "alternative"

AGENT_NAMES = {
    "minimax": "Minimax (Alpha-Beta)",
    "tiltak":  "Tiltak (MCTS)"
}

NUM_WORKERS = 12  # 1.5x of 8 cores


def plot_results(minimax_wins, tiltak_wins, draws, total_games, match_type, save_path="comparison_plot.png"):
    mm_pct = 100 * minimax_wins / total_games
    tl_pct = 100 * tiltak_wins  / total_games
    dr_pct = 100 * draws        / total_games

    if match_type == "minimax_first":
        subtitle = f"{AGENT_NAMES['minimax']} goes first in all games"
    elif match_type == "tiltak_first":
        subtitle = f"{AGENT_NAMES['tiltak']} goes first in all games"
    else:
        subtitle = "First move alternates each game"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"{AGENT_NAMES['minimax']}  vs  {AGENT_NAMES['tiltak']}\n"
        f"{subtitle}  |  {total_games} games",
        fontsize=13, fontweight='bold', y=1.02
    )

    ax = axes[0]
    bar_height = 0.4
    colors = ["#4C72B0", "#DD8452", "#8C8C8C"]

    ax.barh(0, mm_pct, bar_height, color=colors[0], label=AGENT_NAMES['minimax'])
    ax.barh(0, tl_pct, bar_height, left=mm_pct, color=colors[1], label=AGENT_NAMES['tiltak'])
    ax.barh(0, dr_pct, bar_height, left=mm_pct + tl_pct, color=colors[2], label="Draw")

    for val, left, color in [
        (mm_pct, 0,              "white"),
        (tl_pct, mm_pct,         "white"),
        (dr_pct, mm_pct+tl_pct, "white"),
    ]:
        if val > 5:
            ax.text(left + val / 2, 0, f"{val:.1f}%",
                    ha='center', va='center', color=color, fontsize=11, fontweight='bold')

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Percentage of games (%)", fontsize=11)
    ax.set_title("Win/Draw Distribution", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)

    ax2 = axes[1]
    categories = [AGENT_NAMES['minimax'], AGENT_NAMES['tiltak'], "Draw"]
    counts     = [minimax_wins, tiltak_wins, draws]
    x          = np.arange(len(categories))
    bars       = ax2.bar(x, counts, width=0.5, color=colors, edgecolor='white', linewidth=1.2)

    for bar, count, pct in zip(bars, counts, [mm_pct, tl_pct, dr_pct]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{count}\n({pct:.1f}%)",
                 ha='center', va='bottom', fontsize=10)

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylabel("Number of games", fontsize=11)
    ax2.set_title("Win Counts", fontsize=11)
    ax2.set_ylim(0, max(counts) * 1.3)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


N = 5


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
    def __init__(self, binary_path, size=5, movetime=2000):
        self.movetime = movetime
        self.size     = size
        self.proc = subprocess.Popen(
            [binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._send("tei")
        while True:
            line = self.proc.stdout.readline().decode().strip()
            if line == "teiok":
                break
        self._send(f"teinewgame {size}")

    def _send(self, cmd):
        self.proc.stdin.write((cmd + "\n").encode())
        self.proc.stdin.flush()

    def new_game(self):
        self._send(f"teinewgame {self.size}")

    def get_move(self, move_history_tiltak, env, max_retries=5):
        if move_history_tiltak:
            self._send(f"position startpos moves {' '.join(move_history_tiltak)}")
        else:
            self._send("position startpos")

        retries = 0
        while True:
            self._send(f"go movetime {self.movetime}")
            while True:
                line = self.proc.stdout.readline().decode().strip()
                if line.startswith("bestmove"):
                    move = line.split()[1]
                    break

            if move.startswith('C') and len(move) == 3:
                retries += 1
                if retries >= max_retries:
                    import random
                    legal = env.legal_moves()
                    fallback = random.choice(legal)
                    print(f"  [Falling back to random legal move: {fallback}]")
                    return env_to_tiltak(fallback), fallback
            else:
                return move, tiltak_to_env(move)

    def close(self):
        try:
            self._send("quit")
            self.proc.terminate()
        except Exception:
            pass


# ── Minimax wrapper ──────────────────────────────────────────────

class MinimaxPlayer:
    def __init__(self, player_id, n=5, time_limit=10):
        self.player_id = player_id
        self.proc = subprocess.Popen(
            ["./minimax"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0
        )
        init = f"{player_id} {n} {time_limit}\n"
        self.proc.stdin.write(init.encode())
        self.proc.stdin.flush()

    def send_move(self, move):
        self.proc.stdin.write((move + "\n").encode())
        self.proc.stdin.flush()

    def get_move(self, timeout=15):
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if ready:
            line = self.proc.stdout.readline().decode("utf-8", errors="replace").strip()
            return line
        return None

    def close(self):
        try:
            self.proc.kill()
        except Exception:
            pass


# ── board print ──────────────────────────────────────────────────

def print_board(env, minimax_player, tiltak_player):
    print()
    for row in range(N - 1, -1, -1):
        line = f"  {row+1} "
        for col in range(N):
            idx = col * N + row
            sq  = env.board[idx]
            if not sq:
                cell = "."
            else:
                cell = "".join(
                    f"{'S' if t=='S' else 'F'}{'M' if p==minimax_player else 'T'}"
                    for p, t in sq
                )
            line += f"  {cell:<6}"
        print(line)
    print("      " + "  ".join(f"{chr(ord('a')+c):<8}" for c in range(N)))
    print(f"  Flats: Minimax(P{minimax_player+1})={env.flats[minimax_player]}  "
          f"Tiltak(P{tiltak_player+1})={env.flats[tiltak_player]}")
    print()


# ── play one game ────────────────────────────────────────────────

def play_game(tiltak, minimax_first=True, verbose=True):
    minimax_player = 0 if minimax_first else 1
    tiltak_player  = 1 - minimax_player
    minimax_id     = minimax_player + 1

    env     = TakEnv().reset()
    minimax = MinimaxPlayer(player_id=minimax_id, n=N, time_limit=1)
    tiltak.new_game()

    move_history_tiltak = []

    if verbose:
        print("\n" + "="*55)
        if minimax_first:
            print("  Minimax (P1) vs Tiltak (P2)  —  Minimax goes first")
        else:
            print("  Tiltak (P1) vs Minimax (P2)  —  Tiltak goes first")
        print("="*55)
        print_board(env, minimax_player, tiltak_player)

    while not env.done:
        if env.turn == minimax_player:
            move_env = minimax.get_move(timeout=15)
            if not move_env:
                print("  Minimax timed out — Tiltak wins by default.")
                env.winner = tiltak_player
                break
            if verbose:
                print(f"  Minimax (P{minimax_player+1}) plays: {move_env}")
            _, done = env.step(move_env)
            move_tiltak = env_to_tiltak(move_env)
            move_history_tiltak.append(move_tiltak)
        else:
            result = tiltak.get_move(move_history_tiltak, env)
            if result is None:
                print("  Tiltak timed out — Minimax wins by default.")
                env.winner = minimax_player
                break
            move_tiltak, move_env = result
            if verbose:
                print(f"  Tiltak (P{tiltak_player+1}) plays: {move_tiltak} (Tiltak notation)")
                print(f"  Tiltak (P{tiltak_player+1}) plays: {move_env} (env notation)")
            _, done = env.step(move_env)
            move_history_tiltak.append(move_tiltak)
            minimax.send_move(move_env)

        if verbose:
            print_board(env, minimax_player, tiltak_player)

    minimax.close()

    w = env.winner
    if w == -1: w = 2

    if w == minimax_player:   print("  Result: Minimax wins!")
    elif w == tiltak_player:  print("  Result: Tiltak wins!")
    else:                     print("  Result: Draw!")

    return w, minimax_player, tiltak_player


# ── parallel worker (must be top-level for pickling) ─────────────

def run_single_game(game_args):
    tiltak_binary, minimax_first, movetime = game_args
    tiltak = TiltakPlayer(binary_path=tiltak_binary, size=N, movetime=movetime)
    w, mp, tp = play_game(tiltak=tiltak, minimax_first=minimax_first, verbose=False)
    tiltak.close()
    return w, mp, tp


# ── main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiltak',   type=str, default='../other_models/tiltak/target/release/tei')
    parser.add_argument('--games',    type=int, default=10)
    parser.add_argument('--movetime', type=int, default=500,
                        help='Tiltak thinking time in ms per move')
    parser.add_argument('--first', type=str, default=MATCH_TYPE,
                        choices=['minimax_first', 'tiltak_first', 'alternate'])
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    if not os.path.exists('./minimax'):
        print("ERROR: ./minimax binary not found. Run: bash compile.sh")
        return
    if not os.path.exists(args.tiltak):
        print(f"ERROR: Tiltak binary not found at {args.tiltak}")
        return

    # build game args list
    game_args = []
    
    for g in range(args.games):
        if args.first == 'minimax_first':
            minimax_first = True
        elif args.first == 'tiltak_first':
            minimax_first = False
        else:
            minimax_first = (g % 2 == 0)
        game_args.append((args.tiltak, minimax_first, args.movetime))

    print(f"Running {args.games} games across {NUM_WORKERS} workers...")

    with Pool(processes=NUM_WORKERS) as pool:
        results = list(pool.map(run_single_game, game_args))

    minimax_wins = 0
    tiltak_wins  = 0
    draws        = 0

    for g, (w, mp, tp) in enumerate(results):
        print(f"  Game {g+1}: {'Minimax' if w==mp else 'Tiltak' if w==tp else 'Draw'}")
        if w == mp:    minimax_wins += 1
        elif w == tp:  tiltak_wins  += 1
        else:          draws        += 1

    total = minimax_wins + tiltak_wins + draws
    print("\n" + "="*55)
    print(f"  Results over {args.games} games (--first {args.first}):")
    print(f"  Minimax wins : {minimax_wins}  ({100*minimax_wins/total:.0f}%)")
    print(f"  Tiltak wins  : {tiltak_wins}  ({100*tiltak_wins/total:.0f}%)")
    print(f"  Draws        : {draws}")
    print("="*55)

    plot_results(
        minimax_wins=minimax_wins,
        tiltak_wins=tiltak_wins,
        draws=draws,
        total_games=args.games,
        match_type=MATCH_TYPE,
        save_path=f"comparison_{MATCH_TYPE}.png"
    )


if __name__ == "__main__":
    main()