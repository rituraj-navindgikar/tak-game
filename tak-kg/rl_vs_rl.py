"""
rl_vs_rl.py
Pit two different RL models against each other in the terminal.

Usage:
    # AZ vs AZ
    python3 rl_vs_rl.py --az ../tak_checkpoints/best.pt --az2 ../tak_checkpoints/iter_0050.pt

    # AZ vs PPO
    python3 rl_vs_rl.py --az ../tak_checkpoints/best.pt --ppo ../ppo_checkpoints/best.pt

    # PPO vs PPO
    python3 rl_vs_rl.py --ppo ../ppo_checkpoints/best.pt --ppo2 ../ppo_checkpoints/iter_0260.pt

    # Common options
    python3 rl_vs_rl.py --az ../tak_checkpoints/best.pt --ppo ../ppo_checkpoints/best.pt \
                        --games 20 --first alternate --sims 50 --quiet
"""

import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Pool
from alpha_zero_rl_model import TakEnv, TakNet, MCTS, ActionSpace, encode

N           = 5
NUM_WORKERS = 8
OBS         = N * N * 5 + 4
HIDDEN      = 256

ASP = ActionSpace()


# ── PPO network ──────────────────────────────────────────────────
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


# ── load agent ───────────────────────────────────────────────────
def load_agent(model_path, model_type, n_sims=100, device='cpu', label="agent"):
    if model_type == 'az':
        net = TakNet(h=HIDDEN).to(device)
        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        net.eval()
        return ('az', MCTS(net, n_sims=n_sims, device=device))
    else:
        net = PPONet().to(device)
        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        net.eval()
        return ('ppo', net)


# ── agent move helper ────────────────────────────────────────────
def agent_move(agent_tuple, env):
    kind, agent = agent_tuple
    if kind == 'az':
        move, _ = agent.best_move(env, temp=0)
    else:
        move = agent.act(env)
    return move


# ── board print ──────────────────────────────────────────────────
def print_board(env, label1="M1", label2="M2"):
    print()
    for row in range(N - 1, -1, -1):
        line = f"  {row+1} "
        for col in range(N):
            idx = col * N + row
            sq  = env.board[idx]
            cell = "." if not sq else "".join(
                f"{'W' if t=='S' else 'F'}{'1' if p==0 else '2'}" for p, t in sq)
            line += f"  {cell:<6}"
        print(line)
    print("      " + "  ".join(f"{chr(ord('a')+c):<8}" for c in range(N)))
    print(f"  Flats: {label1}(P1)={env.flats[0]}  {label2}(P2)={env.flats[1]}")
    print()


# ── play one game ────────────────────────────────────────────────
def play_game(agent1, agent2, p1_is_model1, label1, label2, verbose=True):
    env    = TakEnv().reset()
    agents = {0: agent1, 1: agent2} if p1_is_model1 else {0: agent2, 1: agent1}
    labels = {0: label1, 1: label2} if p1_is_model1 else {0: label2, 1: label1}

    if verbose:
        first  = label1 if p1_is_model1 else label2
        second = label2 if p1_is_model1 else label1
        print(f"\n{'='*50}")
        print(f"  {first} (P1, first) vs {second} (P2)")
        print("="*50)
        print_board(env, label1, label2)

    while not env.done:
        cur  = env.turn
        move = agent_move(agents[cur], env)
        if verbose:
            print(f"  {labels[cur]} (P{cur+1}) plays: {move}")
        env.step(move)
        if verbose:
            print_board(env, label1, label2)

    w = env.winner
    if w == -1:
        return 2  # draw
    # remap board winner (0=P1, 1=P2) -> model winner (0=model1, 1=model2)
    if p1_is_model1:
        return w
    else:
        return 1 - w


# ── parallel worker ──────────────────────────────────────────────
def run_single_game(game_args):
    path1, type1, path2, type2, sims1, sims2, p1_is_model1, label1, label2, verbose = game_args
    a1 = load_agent(path1, type1, n_sims=sims1, label=label1)
    a2 = load_agent(path2, type2, n_sims=sims2, label=label2)
    return play_game(a1, a2, p1_is_model1=p1_is_model1, label1=label1, label2=label2, verbose=verbose)


# ── parse model args ─────────────────────────────────────────────
def parse_models(args):
    """
    Accepts these flag combos and returns (path1, type1, label1, path2, type2, label2):
      --az PATH --ppo PATH       -> AZ vs PPO
      --az PATH --az2 PATH       -> AZ vs AZ
      --ppo PATH --ppo2 PATH     -> PPO vs PPO
      --ppo PATH --az PATH       -> PPO vs AZ  (ppo is model1)
    """
    entries = []
    if args.az:   entries.append((args.az,   'az',  'AlphaZero'))
    if args.ppo:  entries.append((args.ppo,  'ppo', 'PPO'))
    if args.az2:  entries.append((args.az2,  'az',  'AlphaZero-2'))
    if args.ppo2: entries.append((args.ppo2, 'ppo', 'PPO-2'))

    if len(entries) != 2:
        raise ValueError("Provide exactly two model flags: e.g. --az PATH --ppo PATH, or --az PATH --az2 PATH")

    (path1, type1, label1), (path2, type2, label2) = entries

    # allow label overrides
    if args.label1: label1 = args.label1
    if args.label2: label2 = args.label2

    return path1, type1, label1, path2, type2, label2


# ── main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    # model flags
    parser.add_argument('--az',    type=str, default=None, help='AlphaZero checkpoint (model 1)')
    parser.add_argument('--ppo',   type=str, default=None, help='PPO checkpoint (model 1 if no --az)')
    parser.add_argument('--az2',   type=str, default=None, help='Second AlphaZero checkpoint (model 2)')
    parser.add_argument('--ppo2',  type=str, default=None, help='Second PPO checkpoint (model 2)')

    # optional label overrides
    parser.add_argument('--label1', type=str, default=None, help='Override label for model 1')
    parser.add_argument('--label2', type=str, default=None, help='Override label for model 2')

    # game settings
    parser.add_argument('--games',   type=int, default=10)
    parser.add_argument('--sims',    type=int, default=100,  help='MCTS sims for AZ model 1')
    parser.add_argument('--sims2',   type=int, default=None, help='MCTS sims for AZ model 2 (default: same as --sims)')
    parser.add_argument('--first',   type=str, default='alternate',
                        choices=['model1', 'model2', 'alternate'],
                        help='Who goes first: model1, model2, or alternate each game')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--quiet',   action='store_true', help='Suppress board prints')
    args = parser.parse_args()

    try:
        path1, type1, label1, path2, type2, label2 = parse_models(args)
    except ValueError as e:
        print(f"ERROR: {e}"); return

    for path in [path1, path2]:
        if not os.path.exists(path):
            print(f"ERROR: model not found at {path}"); return

    sims2 = args.sims2 if args.sims2 else args.sims

    def sims_label(t, s): return f"sims={s}" if t == 'az' else "greedy"

    print(f"Running {args.games} games across {args.workers} workers...")
    print(f"  First-move mode : {args.first}")
    print(f"  {label1} ({type1.upper()}, {sims_label(type1, args.sims)}) vs "
          f"{label2} ({type2.upper()}, {sims_label(type2, sims2)})")

    game_args = []
    for g in range(args.games):
        if args.first == 'model1':   p1_is_model1 = True
        elif args.first == 'model2': p1_is_model1 = False
        else:                        p1_is_model1 = (g % 2 == 0)
        game_args.append((path1, type1, path2, type2, args.sims, sims2,
                          p1_is_model1, label1, label2, not args.quiet))

    with Pool(processes=args.workers) as pool:
        results = list(pool.map(run_single_game, game_args))

    m1_wins = m2_wins = draws = 0
    for g, w in enumerate(results):
        if   w == 0: m1_wins += 1; outcome = f"{label1} wins"
        elif w == 1: m2_wins += 1; outcome = f"{label2} wins"
        else:        draws   += 1; outcome = "Draw"
        first_label = label1 if game_args[g][6] else label2
        print(f"  Game {g+1:>3}: {outcome}  (first: {first_label})")

    total = m1_wins + m2_wins + draws
    print("\n" + "="*50)
    print(f"  Results over {args.games} games:")
    print(f"  {label1} wins : {m1_wins}  ({100*m1_wins/total:.0f}%)")
    print(f"  {label2} wins : {m2_wins}  ({100*m2_wins/total:.0f}%)")
    print(f"  Draws         : {draws}  ({100*draws/total:.0f}%)")
    print("="*50)


if __name__ == "__main__":
    main()