"""
rl_vs_minimax.py
Pit the trained RL agent against the minimax AI in the terminal.

Usage:
    python3 rl_vs_minimax.py                          # minimax first (default)
    python3 rl_vs_minimax.py --first rl               # RL first
    python3 rl_vs_minimax.py --first alternate        # alternate each game
    python3 rl_vs_minimax.py --games 20 --quiet
"""

import subprocess, argparse, os
import torch
from alpha_zero_rl_model import TakEnv, TakNet, MCTS, ActionSpace

N = 5

# ── load RL agent ────────────────────────────────────────────────
def load_rl_agent(model_path, hidden=256, n_sims=100):
    net = TakNet(h=hidden)
    net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    net.eval()
    return MCTS(net, n_sims=n_sims, device='cpu')


# ── minimax process wrapper ──────────────────────────────────────
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
        import select
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
def print_board(env, rl_player, minimax_player):
    print()
    for row in range(N - 1, -1, -1):
        line = f"  {row+1} "
        for col in range(N):
            idx = col * N + row
            sq  = env.board[idx]
            if not sq:
                cell = "."
            else:
                cell = "".join(f"{'W' if t=='S' else 'F'}{'1' if p==0 else '2'}" for p, t in sq)
            line += f"  {cell:<6}"
        print(line)
    print("      " + "  ".join(f"{chr(ord('a')+c):<8}" for c in range(N)))
    rl_label      = f"RL(P{rl_player+1})"
    minimax_label = f"Minimax(P{minimax_player+1})"
    print(f"  Flats: {rl_label}={env.flats[rl_player]}  {minimax_label}={env.flats[minimax_player]}")
    print()


# ── play one game ────────────────────────────────────────────────
def play_game(rl_agent, rl_player, verbose=True):
    """
    rl_player: 0 = RL goes first (P1), 1 = RL goes second (P2)
    """
    minimax_player = 1 - rl_player
    minimax_id     = minimax_player + 1   # minimax expects 1-indexed player id

    env     = TakEnv().reset()
    minimax = MinimaxPlayer(player_id=minimax_id, n=N, time_limit=1)

    if verbose:
        print("\n" + "="*50)
        if rl_player == 0:
            print("  RL Agent (P1) vs Minimax (P2) — RL goes first")
        else:
            print("  Minimax (P1) vs RL Agent (P2) — Minimax goes first")
        print("="*50)
        print_board(env, rl_player, minimax_player)

    while not env.done:
        if env.turn == rl_player:
            # RL agent's turn
            move, _ = rl_agent.best_move(env, temp=0)
            if verbose:
                print(f"  RL Agent (P{rl_player+1}) plays: {move}")
            _, done = env.step(move)
            minimax.send_move(move)

        else:
            # minimax's turn
            move = minimax.get_move(timeout=15)
            if not move:
                print("  Minimax timed out.")
                env.winner = rl_player
                break
            if verbose:
                print(f"  Minimax (P{minimax_player+1}) plays: {move}")
            _, done = env.step(move)

        if verbose:
            print_board(env, rl_player, minimax_player)

    minimax.close()

    w = env.winner
    if w == -1: w = 2

    if verbose:
        if w == rl_player:      print("  Result: RL Agent wins!")
        elif w == minimax_player: print("  Result: Minimax wins!")
        else:                   print("  Result: Draw!")

    return w, rl_player


# ── main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  type=str,   default='../checkpoints/best.pt')
    parser.add_argument('--games',  type=int,   default=10)
    parser.add_argument('--sims',   type=int,   default=100)
    parser.add_argument('--hidden', type=int,   default=256)
    parser.add_argument('--first',  type=str,   default='minimax',
                        choices=['minimax', 'rl', 'alternate'],
                        help='Who goes first: minimax, rl, or alternate each game')
    parser.add_argument('--quiet',  action='store_true')
    args = parser.parse_args()

    if not os.path.exists('./minimax'):
        print("ERROR: ./minimax binary not found. Run: bash compile.sh")
        return
    if not os.path.exists(args.model):
        print(f"ERROR: model not found at {args.model}")
        return

    print(f"Loading RL agent from {args.model}...")
    agent = load_rl_agent(args.model, hidden=args.hidden, n_sims=args.sims)

    rl_wins      = 0
    minimax_wins = 0
    draws        = 0

    for g in range(args.games):
        print(f"\n--- Game {g+1}/{args.games} ---")

        if args.first == 'minimax':
            rl_player = 1
        elif args.first == 'rl':
            rl_player = 0
        else:   # alternate
            rl_player = g % 2   # even games RL first, odd games minimax first

        w, rp = play_game(agent, rl_player=rl_player, verbose=not args.quiet)

        if w == rp:               rl_wins      += 1
        elif w == 1 - rp:         minimax_wins += 1
        else:                     draws        += 1

    total = rl_wins + minimax_wins + draws
    print("\n" + "="*50)
    print(f"  Results over {args.games} games (--first {args.first}):")
    print(f"  RL Agent wins  : {rl_wins}  ({100*rl_wins/total:.0f}%)")
    print(f"  Minimax wins   : {minimax_wins}  ({100*minimax_wins/total:.0f}%)")
    print(f"  Draws          : {draws}")
    print("="*50)


if __name__ == "__main__":
    main()