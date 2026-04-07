"""
rl_vs_rl.py
Pit two different RL models against each other in the terminal.

Model 1 (P1) always goes first.
Model 2 (P2) goes second.

Usage:
    python3 rl_vs_rl.py --model1 checkpoints/best_v1.pt --model2 checkpoints/best_v2.pt
    python3 rl_vs_rl.py --model1 checkpoints/best_v1.pt --model2 checkpoints/best_v2.pt --games 20 --quiet
"""

import argparse, os
import torch
from alpha_zero_rl_model import TakEnv, TakNet, MCTS, ActionSpace

N = 3


# ── load agent ───────────────────────────────────────────────────
def load_agent(model_path, hidden=128, n_sims=100, label="agent"):
    asp = ActionSpace()
    net = TakNet(h=hidden)
    net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    net.eval()
    print(f"Loaded {label} from {model_path} -> {sum(p.numel() for p in net.parameters()):,} params")
    return MCTS(net, n_sims=n_sims, device='cpu')


# ── board print ──────────────────────────────────────────────────
def print_board(env, label1="M1", label2="M2"):
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
    print(f"  Flats: {label1}(P1)={env.flats[0]}  {label2}(P2)={env.flats[1]}")
    print()


# ── play one game ────────────────────────────────────────────────
def play_game(agent1, agent2, label1="Model1", label2="Model2", verbose=True):
    env    = TakEnv().reset()
    agents = {0: agent1, 1: agent2}
    labels = {0: label1, 1: label2}

    if verbose:
        print("\n" + "="*50)
        print(f"  {label1} (P1) vs {label2} (P2)")
        print(f"  {label1} goes first")
        print("="*50)
        print_board(env, label1, label2)

    while not env.done:
        current = env.turn
        move, _ = agents[current].best_move(env, temp=0)

        if verbose:
            print(f"  {labels[current]} (P{current+1}) plays: {move}")

        _, done = env.step(move)

        if verbose:
            print_board(env, label1, label2)

    w = env.winner
    if w == -1:
        w = 2

    if verbose:
        if w == 0:   print(f"  Result: {label1} (P1) wins!")
        elif w == 1: print(f"  Result: {label2} (P2) wins!")
        else:        print(f"  Result: Draw!")

    return w


# ── main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1',  type=str, required=True,             help='Path to model 1 (plays as P1)')
    parser.add_argument('--model2',  type=str, required=True,             help='Path to model 2 (plays as P2)')
    parser.add_argument('--label1',  type=str, default='Model1',          help='Label for model 1')
    parser.add_argument('--label2',  type=str, default='Model2',          help='Label for model 2')
    parser.add_argument('--games',   type=int, default=10)
    parser.add_argument('--sims',    type=int, default=100)
    parser.add_argument('--sims2',   type=int, default=None,              help='Override sims for model 2 (default: same as --sims)')
    parser.add_argument('--hidden',  type=int, default=128)
    parser.add_argument('--quiet',   action='store_true',                 help='Suppress board prints')
    args = parser.parse_args()

    sims2 = args.sims2 if args.sims2 else args.sims

    for path in [args.model1, args.model2]:
        if not os.path.exists(path):
            print(f"ERROR: model not found at {path}")
            return

    agent1 = load_agent(args.model1, hidden=args.hidden, n_sims=args.sims,  label=args.label1)
    agent2 = load_agent(args.model2, hidden=args.hidden, n_sims=sims2,      label=args.label2)

    results = {0: 0, 1: 0, 2: 0}

    for g in range(args.games):
        print(f"\n--- Game {g+1}/{args.games} ---")
        w = play_game(agent1, agent2,
                      label1=args.label1,
                      label2=args.label2,
                      verbose=not args.quiet)
        results[w] += 1

    print("\n" + "="*50)
    print(f"  Results over {args.games} games:")
    print(f"  {args.label1} (P1) wins: {results[0]}")
    print(f"  {args.label2} (P2) wins: {results[1]}")
    print(f"  Draws              : {results[2]}")
    print("="*50)


if __name__ == "__main__":
    main()