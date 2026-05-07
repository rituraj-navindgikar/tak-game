"""
play.py — master runner for any vs any Tak games

Player types:
  random    — picks a random legal move
  minimax   — uses the compiled ./minimax binary
  rl        — uses a trained RL model (requires --p1-model or --p2-model)
  human     — opens the web UI in your browser (requires Flask app running)

Usage:
  python3 play.py random minimax
  python3 play.py rl minimax --p1-model checkpoints/best.pt
  python3 play.py rl rl --p1-model checkpoints/v1.pt --p2-model checkpoints/v2.pt
  python3 play.py human rl --p2-model checkpoints/best.pt
  python3 play.py random random --games 50 --quiet
  python3 play.py minimax minimax --games 5
"""

import argparse, os, sys, subprocess, random, webbrowser, time
import torch
from alpha_zero_rl_model import TakEnv, TakNet, MCTS

N          = 5
MAX_MOVES  = 300


# ══════════════════════════════════════════════════════════════════
# Player classes
# ══════════════════════════════════════════════════════════════════

class RandomPlayer:
    def __init__(self, player_id):
        self.player_id = player_id

    def get_move(self, env, **kwargs):
        moves = env.legal_moves()
        return random.choice(moves)

    def notify(self, move):
        pass  # random doesn't care about opponent moves

    def close(self):
        pass


class MinimaxPlayer:
    def __init__(self, player_id, n=N, time_limit=10):
        self.player_id = player_id
        self.proc = subprocess.Popen(
            ["./minimax"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0
        )
        self.proc.stdin.write(f"{player_id} {n} {time_limit}\n".encode())
        self.proc.stdin.flush()

    def get_move(self, env, **kwargs):
        import select
        ready, _, _ = select.select([self.proc.stdout], [], [], 15)
        if ready:
            line = self.proc.stdout.readline().decode("utf-8", errors="replace").strip()
            return line
        return None

    def notify(self, move):
        self.proc.stdin.write((move + "\n").encode())
        self.proc.stdin.flush()

    def close(self):
        try:
            self.proc.kill()
        except Exception:
            pass


class RLPlayer:
    def __init__(self, player_id, model_path, hidden=256, n_sims=100):
        self.player_id = player_id
        net = TakNet(h=hidden)
        net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
        net.eval()
        self.mcts = MCTS(net, n_sims=n_sims, device='cpu')

    def get_move(self, env, **kwargs):
        move, _ = self.mcts.best_move(env, temp=0)
        return move

    def notify(self, move):
        pass  # RL reads state from env directly

    def close(self):
        pass


class HumanPlayer:
    def __init__(self, player_id, port=5000):
        self.player_id = player_id
        self.port = port
        print(f"\n  Human player (P{player_id}) — opening browser at http://localhost:{port}")
        print("  Make sure app.py is running first.")
        time.sleep(1)
        webbrowser.open(f"http://localhost:{port}")

    def get_move(self, env, verbose=True):
        legal = env.legal_moves()
        while True:
            sys.stderr.write(f"\n  Your move (P{self.player_id}): ")
            sys.stderr.flush()
            move = sys.stdin.readline().strip()
            if not move:
                continue
            if move in legal:
                return move
            sys.stderr.write(f"  Invalid. Sample legal moves: {legal[:5]}\n")
            sys.stderr.flush()

    def notify(self, move):
        print(f"  Opponent played: {move}")

    def close(self):
        pass


# ══════════════════════════════════════════════════════════════════
# Board print
# ══════════════════════════════════════════════════════════════════

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
            line += f"  {cell:<5}"
        print(line)
    print("     " + "  ".join(f"{chr(ord('a')+c):<7}" for c in range(N)))
    print(f"  Flats: P1={env.flats[0]}  P2={env.flats[1]}  Turn: P{env.turn+1}  Move#: {env.moves}")
    print()


# ══════════════════════════════════════════════════════════════════
# Build player from args
# ══════════════════════════════════════════════════════════════════

def build_player(ptype, player_id, args):
    if ptype == 'random':
        return RandomPlayer(player_id)

    elif ptype == 'minimax':
        if not os.path.exists('./minimax'):
            print("ERROR: ./minimax not found. Run: bash compile.sh")
            sys.exit(1)
        return MinimaxPlayer(player_id, n=N, time_limit=args.time_limit)

    elif ptype == 'rl':
        model_path = args.p1_model if player_id == 1 else args.p2_model
        if not model_path:
            print(f"ERROR: --p{player_id}-model required for rl player")
            sys.exit(1)
        if not os.path.exists(model_path):
            print(f"ERROR: model not found at {model_path}")
            sys.exit(1)
        return RLPlayer(player_id, model_path, hidden=args.hidden, n_sims=args.sims)

    elif ptype == 'human':
        return HumanPlayer(player_id, port=args.port)

    else:
        print(f"ERROR: unknown player type '{ptype}'")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# Play one game
# ══════════════════════════════════════════════════════════════════

def play_game(p1, p2, verbose=True):
    env     = TakEnv().reset()
    players = {0: p1, 1: p2}

    if verbose:
        print("\n" + "="*50)
        print(f"  {type(p1).__name__} (P1) vs {type(p2).__name__} (P2)")
        print("="*50)
        print_board(env)

    for _ in range(MAX_MOVES):
        if env.done:
            break

        current = players[env.turn]
        opponent = players[1 - env.turn]

        move = current.get_move(env, verbose=verbose)

        if move is None:
            print(f"  P{env.turn+1} failed to produce a move.")
            env.winner = 1 - env.turn
            env.done   = True
            break

        if verbose:
            print(f"  P{env.turn+1} ({type(current).__name__}) plays: {move}")

        _, done = env.step(move)

        # tell opponent what was played (needed by minimax)
        opponent.notify(move)

        if verbose:
            print_board(env)

        if done:
            break

    if not env.done:
        env.winner = 2   # draw if MAX_MOVES hit

    w = env.winner
    if verbose:
        if w == 0:   print("  Result: P1 wins!")
        elif w == 1: print("  Result: P2 wins!")
        else:        print("  Result: Draw!")

    return w


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Tak — any vs any runner',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
player types: random | minimax | rl | human

examples:
  python3 play.py random minimax
  python3 play.py rl minimax --p1-model checkpoints/best.pt
  python3 play.py rl rl --p1-model checkpoints/v1.pt --p2-model checkpoints/v2.pt
  python3 play.py human rl --p2-model checkpoints/best.pt
  python3 play.py random random --games 50 --quiet
        """
    )
    parser.add_argument('p1',        type=str, choices=['random','minimax','rl','human'], help='Player 1 type (goes first)')
    parser.add_argument('p2',        type=str, choices=['random','minimax','rl','human'], help='Player 2 type (goes second)')
    parser.add_argument('--p1-model',type=str, default=None,  help='Model path for P1 if rl')
    parser.add_argument('--p2-model',type=str, default=None,  help='Model path for P2 if rl')
    parser.add_argument('--games',   type=int, default=1,     help='Number of games to play')
    parser.add_argument('--sims',    type=int, default=100,   help='MCTS simulations for rl players')
    parser.add_argument('--hidden',  type=int, default=256,   help='Network hidden size for rl players')
    parser.add_argument('--time-limit', type=int, default=100, help='Time limit per move for minimax (seconds)')
    parser.add_argument('--port',    type=int, default=5000,  help='Flask port for human player')
    parser.add_argument('--quiet',   action='store_true',     help='Suppress board prints')
    args = parser.parse_args()

    p1_wins = 0
    p2_wins = 0
    draws   = 0

    for g in range(args.games):
        print(f"\n--- Game {g+1}/{args.games} ---")

        # rebuild players each game so minimax gets a fresh process
        p1 = build_player(args.p1, 1, args)
        p2 = build_player(args.p2, 2, args)

        w = play_game(p1, p2, verbose=not args.quiet)

        if w == 0:   p1_wins += 1
        elif w == 1: p2_wins += 1
        else:        draws   += 1

        p1.close()
        p2.close()

    total = p1_wins + p2_wins + draws
    print("\n" + "="*50)
    print(f"  Results over {args.games} game(s):")
    print(f"  P1 ({args.p1}) wins : {p1_wins}  ({100*p1_wins/total:.0f}%)")
    print(f"  P2 ({args.p2}) wins : {p2_wins}  ({100*p2_wins/total:.0f}%)")
    print(f"  Draws              : {draws}")
    print("="*50)


if __name__ == "__main__":
    main()