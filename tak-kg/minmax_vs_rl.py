"""
rl_vs_minimax.py
Pit the trained RL agent against the minimax AI in the terminal.

Minimax always plays as Player 1 (goes first).
RL agent plays as Player 2.

Usage:
    python3 rl_vs_minimax.py
    python3 rl_vs_minimax.py --games 20 --sims 100 --model checkpoints/best.pt
"""

import subprocess, argparse, time, os
from alpha_zero_rl_model import TakEnv, TakNet, MCTS, ActionSpace
import torch

N = 3

# ── load RL agent ────────────────────────────────────────────────
def load_rl_agent(model_path, hidden=128, n_sims=100):
    asp = ActionSpace()
    net = TakNet(h=hidden)
    net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    net.eval()
    return MCTS(net, n_sims=n_sims, device='cpu')


# ── minimax process wrapper ──────────────────────────────────────
class MinimaxPlayer:
    def __init__(self, n=3, time_limit=10):
        self.proc = subprocess.Popen(
            ["./minimax"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0
        )
        # send init: player=1, n=3, time_limit
        init = f"1 {n} {time_limit}\n"
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
def print_board(env):
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
    print(f"  Flats: P1(minimax)={env.flats[0]}  P2(RL)={env.flats[1]}")
    print()


# ── play one game ────────────────────────────────────────────────
def play_game(rl_agent, verbose=True):
    env     = TakEnv().reset()
    minimax = MinimaxPlayer(n=N, time_limit=10)

    if verbose:
        print("\n" + "="*50)
        print("  Minimax (P1) vs RL Agent (P2)")
        print("  Minimax goes first")
        print("="*50)
        print_board(env)

    while not env.done:
        if env.turn == 0:
            # minimax's turn — just read its move
            move = minimax.get_move(timeout=15)
            if not move:
                print("  Minimax timed out.")
                env.winner = 1   # RL wins by default
                break
            if verbose:
                print(f"  Minimax plays: {move}")
            _, done = env.step(move)

        else:
            # RL agent's turn
            move, _ = rl_agent.best_move(env, temp=0)
            if verbose:
                print(f"  RL Agent plays: {move}")
            _, done = env.step(move)
            # now tell minimax what RL played so it can think next turn
            minimax.send_move(move)

        if verbose:
            print_board(env)

    minimax.close()

    w = env.winner
    if w == -1: w = 2   # draw if game hit move limit
    if verbose:
        if w == 0:   print("  Result: Minimax (P1) wins!")
        elif w == 1: print("  Result: RL Agent (P2) wins!")
        else:        print("  Result: Draw!")
    return w


# ── main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  type=str, default='../checkpoints/alpha-zero-best.pt')
    parser.add_argument('--games',  type=int, default=5)
    parser.add_argument('--sims',   type=int, default=100)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--quiet',  action='store_true', help='suppress board prints')
    args = parser.parse_args()

    if not os.path.exists('./minimax'):
        print("ERROR: ./minimax binary not found. Run: bash compile.sh")
        return

    if not os.path.exists(args.model):
        print(f"ERROR: model not found at {args.model}")
        return

    print(f"Loading RL agent from {args.model}...")
    agent = load_rl_agent(args.model, hidden=args.hidden, n_sims=args.sims)

    results = {0: 0, 1: 0, 2: 0}   # minimax wins, rl wins, draws

    for g in range(args.games):
        print(f"\n--- Game {g+1}/{args.games} ---")
        winner = play_game(agent, verbose=not args.quiet)
        if winner == -1: winner = 2
        results[winner] += 1

    print("\n" + "="*50)
    print(f"  Results over {args.games} games:")
    print(f"  Minimax (P1) wins : {results[0]}")
    print(f"  RL Agent (P2) wins: {results[1]}")
    print(f"  Draws             : {results[2]}")
    print("="*50)


if __name__ == "__main__":
    main()