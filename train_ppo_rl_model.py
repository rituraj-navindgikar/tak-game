import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque
import copy, time, os
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp

# ── env constants ─────────────────────────────────────────────────
N         = 5
MAX_FLATS = 21
MAX_STACK = N
OBS       = N * N * 5 + 4   # 129
MAX_GAME_MOVES = 200

# ── PPO config ────────────────────────────────────────────────────
ITERATIONS    = 300          # number of PPO update cycles
GAMES_PER_ITER = 128         # self-play games per iteration
EPOCHS        = 16           # PPO epochs per iteration
BATCH         = 512
LR            = 1e-4
GAMMA         = 0.995        # discount factor
GAE_LAMBDA    = 0.97        # GAE lambda
CLIP_EPS      = 0.15         # PPO clip epsilon
ENT_COEF      = 0.005        # entropy bonus
VF_COEF       = 0.5         # value loss coefficient
MAX_GRAD_NORM = 0.5
HIDDEN        = 256
NUM_CPUS      = 4

# ── multiprocessing ───────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ── TakEnv ───────────────────────────────────────────────────────
class TakEnv:
    def reset(self):
        self.board  = [[] for _ in range(N * N)]
        self.flats  = [MAX_FLATS, MAX_FLATS]
        self.turn   = 0
        self.moves  = 0
        self.done   = False
        self.winner = -1
        return self

    def clone(self):
        e        = TakEnv.__new__(TakEnv)
        e.board  = [list(s) for s in self.board]
        e.flats  = self.flats[:]
        e.turn   = self.turn
        e.moves  = self.moves
        e.done   = self.done
        e.winner = self.winner
        return e

    def step(self, move):
        assert not self.done
        ok = self._apply(move.strip())
        if not ok:
            self.done = True; self.winner = 1 - self.turn
            return -1.0, True
        return self._check_terminal()

    def legal_moves(self):
        moves = []; opening = self.moves < 2
        if self.moves == 0:             owner = 1 - self.turn
        elif self.moves == 1 and self.turn == 1: owner = 1 - self.turn
        else:                           owner = self.turn
        for idx in range(N * N):
            sq = self._i2s(idx)
            if not self.board[idx]:
                if self.flats[owner] > 0:
                    moves.append('F' + sq)
                    if not opening: moves.append('S' + sq)
            if not opening and self.board[idx] and self.board[idx][-1][0] == self.turn:
                moves += self._stack_moves(idx)
        return moves

    def _apply(self, move):
        if not move: return False
        if move[0].isalpha(): return self._place(move)
        if move[0].isdigit(): return self._move_stack(move)
        return False

    def _place(self, move):
        t = move[0]; idx = self._s2i(move[1:])
        if idx == -1 or self.board[idx] or t not in ('F', 'S'): return False
        if t == 'S' and self.moves < 2: return False
        if self.moves == 0:             owner = 1 - self.turn
        elif self.moves == 1 and self.turn == 1: owner = 1 - self.turn
        else:                           owner = self.turn
        if self.flats[owner] <= 0: return False
        self.board[idx].append((owner, t))
        self.flats[owner] -= 1
        if self.turn == 0: self.moves += 1
        self.turn = 1 - self.turn
        return True

    def _move_stack(self, move):
        if len(move) < 5: return False
        try:
            count = int(move[0]); drops = [int(c) for c in move[4:]]
        except ValueError: return False
        direction = move[3]
        if direction not in ('+', '-', '<', '>'): return False
        src = self._s2i(move[1:3])
        if src == -1 or len(self.board[src]) < count or count < 1 or count > MAX_STACK: return False
        if self.board[src][-1][0] != self.turn: return False
        if sum(drops) != count: return False
        chg = self._dc(direction); cur = src
        for d in drops:
            nxt = cur + chg
            if not self._vs(cur, nxt, direction): return False
            if self.board[nxt] and self.board[nxt][-1][1] == 'S': return False
            cur = nxt
        hand = self.board[src][-count:]
        self.board[src] = self.board[src][:-count]
        cur = src; taken = 0
        for d in drops:
            nxt = cur + chg
            self.board[nxt] += hand[taken:taken + d]
            taken += d; cur = nxt
        if self.turn == 0: self.moves += 1
        self.turn = 1 - self.turn
        return True

    def _stack_moves(self, src):
        moves = []; sq = self._i2s(src)
        for count in range(1, min(len(self.board[src]), MAX_STACK) + 1):
            for d in ('+', '-', '<', '>'):
                chg = self._dc(d); pl = 0; cur = src
                for _ in range(N - 1):
                    nxt = cur + chg
                    if not self._vs(cur, nxt, d): break
                    if self.board[nxt] and self.board[nxt][-1][1] == 'S': break
                    pl += 1; cur = nxt
                if not pl: continue
                for drops in self._partitions(count, pl):
                    moves.append(str(count) + sq + d + ''.join(str(x) for x in drops))
        return moves

    def _partitions(self, total, max_steps):
        res = []
        def _go(rem, ml, cur):
            if rem == 0: res.append(list(cur)); return
            if ml == 0:  return
            for d in range(1, rem + 1):
                cur.append(d); _go(rem - d, ml - 1, cur); cur.pop()
        _go(total, max_steps, [])
        return res

    def _check_terminal(self):
        just = 1 - self.turn
        if self._road(just):
            self.done = True; self.winner = just;      return  1.0, True
        if self._road(self.turn):
            self.done = True; self.winner = self.turn; return -1.0, True
        full = all(self.board); out = self.flats[0] == 0 or self.flats[1] == 0
        if full or out:
            self.done = True; w = self._flat_win(); self.winner = w
            if w == just: return  1.0, True
            if w == 2:    return  0.0, True
            return -1.0, True
        return 0.0, False

    def _road(self, p):
        def dfs(starts, ends):
            vis = set(); stk = []
            for s in starts:
                if self.board[s] and self.board[s][-1] == (p, 'F'):
                    vis.add(s); stk.append(s)
            while stk:
                c = stk.pop()
                if c in ends: return True
                for nb in self._nb(c):
                    if nb not in vis and self.board[nb] and self.board[nb][-1] == (p, 'F'):
                        vis.add(nb); stk.append(nb)
            return False
        return (dfs([i * N for i in range(N)], {(i + 1) * N - 1 for i in range(N)}) or
                dfs(list(range(N)),             {N * N - 1 - i   for i in range(N)}))

    def _flat_win(self):
        c = [0, 0]
        for sq in self.board:
            if sq and sq[-1][1] == 'F': c[sq[-1][0]] += 1
        if c[0] > c[1]: return 0
        if c[1] > c[0]: return 1
        if self.flats[0] > self.flats[1]: return 0
        if self.flats[1] > self.flats[0]: return 1
        return 2

    def _s2i(self, s):
        if len(s) != 2: return -1
        col = ord(s[0]) - ord('a')
        try: row = int(s[1]) - 1
        except: return -1
        if not (0 <= col < N and 0 <= row < N): return -1
        return col * N + row

    def _i2s(self, i): return chr(i // N + ord('a')) + str(i % N + 1)
    def _dc(self, d):  return {'+': 1, '-': -1, '>': N, '<': -N}[d]

    def _vs(self, src, dst, d):
        if dst < 0 or dst >= N * N:          return False
        if d == '>' and src % N == N - 1:    return False
        if d == '<' and src % N == 0:        return False
        if d == '+' and src % N == N - 1:    return False
        if d == '-' and src % N == 0:        return False
        return True

    def _nb(self, i):
        r, c = i % N, i // N; nb = []
        if r > 0:     nb.append(i - 1)
        if r < N - 1: nb.append(i + 1)
        if c > 0:     nb.append(i - N)
        if c < N - 1: nb.append(i + N)
        return nb


# ── ActionSpace ──────────────────────────────────────────────────
class ActionSpace:
    def __init__(self):
        seen = set(); env = TakEnv().reset()
        for col in range(N):
            for row in range(N):
                sq = chr(col + ord('a')) + str(row + 1)
                seen.add('F' + sq); seen.add('S' + sq)
        for src in range(N * N):
            sq = env._i2s(src)
            for count in range(1, N + 1):
                for d in ('+', '-', '<', '>'):
                    chg = env._dc(d); pl = 0; cur = src
                    for _ in range(N - 1):
                        nxt = cur + chg
                        if not env._vs(cur, nxt, d): break
                        pl += 1; cur = nxt
                    for drops in env._partitions(count, pl):
                        seen.add(str(count) + sq + d + ''.join(str(x) for x in drops))
        self.moves    = sorted(seen)
        self.move2idx = {m: i for i, m in enumerate(self.moves)}

    @property
    def size(self): return len(self.moves)
    def encode(self, m): return self.move2idx[m]
    def decode(self, i): return self.moves[i]
    def mask(self, env):
        legal = set(env.legal_moves())
        return [m in legal for m in self.moves]

ASP = ActionSpace()


# ── state encoder ────────────────────────────────────────────────
def road_progress(env, player):
    visited = set(); best = 0
    for start in range(N * N):
        if start in visited: continue
        if not env.board[start]: continue
        if env.board[start][-1] != (player, 'F'): continue
        queue = [start]; component = {start}
        while queue:
            cur = queue.pop()
            for nb in env._nb(cur):
                if nb not in component and env.board[nb] and env.board[nb][-1] == (player, 'F'):
                    component.add(nb); queue.append(nb)
        visited |= component
        best = max(best, len(component))
    return best / (N * N)

def encode(env):
    v = np.zeros(OBS, dtype=np.float32)
    for i in range(N * N):
        b = i * 5; sq = env.board[i]; h = len(sq)
        if h:
            top    = sq[-1]
            v[b]   = 1. if top[0] == 0 else -1.
            v[b+1] = 1. if top[1] == 'F' else -1.
            v[b+2] = h / MAX_STACK
            p0     = sum(1 for p, _ in sq if p == 0)
            v[b+3] = p0 / h; v[b+4] = (h - p0) / h
    v[125] = 1. if env.turn == 0 else -1.
    v[126] = env.flats[0] / MAX_FLATS
    v[127] = env.flats[1] / MAX_FLATS
    v[128] = min(env.moves, 50) / 50.
    return v


# ── PPO network (actor + critic shared trunk) ────────────────────
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        h = self.trunk(x)
        logits = self.pi(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        return logits, self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, env, device='cpu'):
        """Sample action and return (action_idx, log_prob, value, move_str)."""
        self.eval()
        x    = torch.tensor(encode(env), dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.tensor(ASP.mask(env), dtype=torch.bool, device=device).unsqueeze(0)
        logits, val = self.forward(x, mask)
        dist  = torch.distributions.Categorical(logits=logits.squeeze(0))
        idx   = dist.sample()
        return idx.item(), dist.log_prob(idx).item(), val.item(), ASP.decode(idx.item())


# ── self-play to collect rollout data ────────────────────────────
def collect_game(net_state, device='cpu'):
    """
    Play one full game using current policy.
    Returns list of (obs, action_idx, log_prob, reward, value, done, player) tuples.
    """
    net = PPONet().to(device)
    net.load_state_dict(net_state)
    net.eval()

    env  = TakEnv().reset()
    traj = []   # (obs, act, logp, rew, val, done, player)

    for _ in range(MAX_GAME_MOVES):
        obs  = encode(env)
        player = env.turn
        prog0  = road_progress(env, 0)
        prog1  = road_progress(env, 1)

        act, logp, val, move = net.act(env, device)
        _, done = env.step(move)

        traj.append((obs, act, logp, 0.0, val, done, player, prog0, prog1))
        if done: break

    if not env.done: env.winner = 2

    w = env.winner
    # assign terminal rewards with road progress shaping
    examples = []
    for (obs, act, logp, _, val, done, player, p0, p1) in traj:
        if w == 2:        z = 0.
        elif w == player: z = 1.
        else:             z = -1.
        progress = (p0 - p1) if player == 0 else (p1 - p0)
        z = 0.9 * z + 0.1 * progress
        examples.append((obs, act, logp, z, val, done))

    return examples, w


def collect_game_worker(args):
    net_state, device = args
    return collect_game(net_state, device)


# ── GAE returns computation ───────────────────────────────────────
def compute_gae(rewards, values, dones, last_val=0.0):
    """Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    last_gae   = 0.0
    for t in reversed(range(len(rewards))):
        next_val  = last_val if t == len(rewards) - 1 else values[t + 1]
        next_done = dones[t]
        delta     = rewards[t] + GAMMA * next_val * (1 - next_done) - values[t]
        last_gae  = delta + GAMMA * GAE_LAMBDA * (1 - next_done) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ── PPO update ───────────────────────────────────────────────────
def ppo_update(net, opt, batch_obs, batch_acts, batch_logps_old,
               batch_returns, batch_advs, device):
    net.train()
    # normalize advantages
    batch_advs = (batch_advs - batch_advs.mean()) / (batch_advs.std() + 1e-8)

    total_pl, total_vl, total_el = [], [], []

    dataset = torch.utils.data.TensorDataset(
        batch_obs, batch_acts, batch_logps_old, batch_returns, batch_advs
    )
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    for _ in range(EPOCHS):
        for obs, acts, logps_old, rets, advs in loader:
            obs      = obs.to(device)
            acts     = acts.to(device)
            logps_old = logps_old.to(device)
            rets     = rets.to(device)
            advs     = advs.to(device)

            # get legal masks - skip masking in batch update for speed
            logits, vals = net(obs)
            dist    = torch.distributions.Categorical(logits=logits)
            logps   = dist.log_prob(acts)
            entropy = dist.entropy().mean()

            # PPO clipped objective
            ratio   = (logps - logps_old).exp()
            surr1   = ratio * advs
            surr2   = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * advs
            p_loss  = -torch.min(surr1, surr2).mean()

            # value loss
            v_loss  = F.mse_loss(vals, rets)

            loss = p_loss + VF_COEF * v_loss - ENT_COEF * entropy

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            opt.step()

            total_pl.append(p_loss.item())
            total_vl.append(v_loss.item())
            total_el.append(entropy.item())

    return np.mean(total_pl), np.mean(total_vl), np.mean(total_el)


# ── eval: PPO vs previous snapshot ───────────────────────────────
def pit(net_new, net_old, n_games=20, device='cpu'):
    nw = ow = d = 0
    for i in range(n_games):
        env    = TakEnv().reset()
        agents = {i % 2: net_new, (i + 1) % 2: net_old}
        new_p  = i % 2
        for _ in range(MAX_GAME_MOVES):
            ag  = agents[env.turn]
            _, _, _, move = ag.act(env, device)
            _, done = env.step(move)
            if done: break
        w = env.winner
        if w == 2:      d  += 1
        elif w == new_p: nw += 1
        else:            ow += 1
    return nw, ow, d


# ── main training loop ───────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs("ppo_checkpoints", exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    net     = PPONet().to(device)
    
    # load existing checkpoint if available
    net.load_state_dict(torch.load("ppo_checkpoints/best.pt", map_location=device, weights_only=False))

    opt     = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)
    best_net = copy.deepcopy(net)

    log = {"iter": [], "p_loss": [], "v_loss": [], "entropy": [],
           "p0_wins": [], "p1_wins": [], "draws": [], "avg_reward": []}

    iter_bar = tqdm(range(1, ITERATIONS + 1), desc="iterations")

    with mp.Pool(processes=NUM_CPUS) as pool:
        for it in iter_bar:
            t0 = time.time()

            # ── collect rollouts ──────────────────────────────────
            net_state = {k: v.cpu() for k, v in net.state_dict().items()}
            args = [(net_state, 'cpu')] * GAMES_PER_ITER

            results = list(tqdm(
                pool.imap_unordered(collect_game_worker, args, chunksize=1),
                total=GAMES_PER_ITER,
                desc=f"[{it:2d}/{ITERATIONS}] collect",
                leave=False
            ))

            # ── flatten rollouts and compute GAE ──────────────────
            all_obs, all_acts, all_logps = [], [], []
            all_rews, all_vals, all_dones = [], [], []
            w_count = [0, 0, 0]; rewards = []

            for traj, winner in results:
                if winner >= 0 and winner <= 2:
                    w_count[winner] += 1
                obs_t  = np.array([e[0] for e in traj])
                acts_t = np.array([e[1] for e in traj])
                logp_t = np.array([e[2] for e in traj])
                rew_t  = np.array([e[3] for e in traj])
                val_t  = np.array([e[4] for e in traj])
                don_t  = np.array([float(e[5]) for e in traj])

                advs, rets = compute_gae(rew_t, val_t, don_t)
                all_obs.append(obs_t); all_acts.append(acts_t)
                all_logps.append(logp_t); all_rews.append(rew_t)
                all_vals.append(val_t); all_dones.append(don_t)
                rewards.append(rew_t.mean())

            batch_obs     = torch.tensor(np.concatenate(all_obs),   dtype=torch.float32)
            batch_acts    = torch.tensor(np.concatenate(all_acts),  dtype=torch.long)
            batch_logps   = torch.tensor(np.concatenate(all_logps), dtype=torch.float32)
            batch_rews    = torch.tensor(np.concatenate(all_rews),  dtype=torch.float32)
            batch_vals    = torch.tensor(np.concatenate(all_vals),  dtype=torch.float32)
            batch_dones   = torch.tensor(np.concatenate(all_dones), dtype=torch.float32)

            # recompute GAE on full batch
            advs_np, rets_np = compute_gae(
                batch_rews.numpy(), batch_vals.numpy(), batch_dones.numpy()
            )
            batch_advs  = torch.tensor(advs_np, dtype=torch.float32)
            batch_rets  = torch.tensor(rets_np, dtype=torch.float32)

            # ── PPO update ────────────────────────────────────────
            pl, vl, ent = ppo_update(
                net, opt,
                batch_obs, batch_acts, batch_logps,
                batch_rets, batch_advs, device
            )

            # ── eval ──────────────────────────────────────────────
            nw, ow, d = pit(net, best_net, n_games=10, device=device)
            best_net = copy.deepcopy(net)  # always update

            # ── log ───────────────────────────────────────────────
            log["iter"].append(it)
            log["p_loss"].append(pl)
            log["v_loss"].append(vl)
            log["entropy"].append(ent)
            log["p0_wins"].append(w_count[0])
            log["p1_wins"].append(w_count[1])
            log["draws"].append(w_count[2])
            log["avg_reward"].append(np.mean(rewards))

            iter_bar.set_postfix(
                p_loss  = f"{pl:.4f}",
                v_loss  = f"{vl:.4f}",
                entropy = f"{ent:.3f}",
                avg_r   = f"{np.mean(rewards):.3f}",
                eval    = f"{nw}-{ow}-{d}",
                elapsed = f"{time.time()-t0:.0f}s"
            )

            if it % 10 == 0:
                torch.save(best_net.state_dict(), f"ppo_checkpoints/best.pt")
                torch.save(best_net.state_dict(), f"ppo_checkpoints/iter_{it:04d}.pt")
                print(f"  checkpoint saved at iter {it}")

    # ── final save ───────────────────────────────────────────────
    torch.save(best_net.state_dict(), "ppo_checkpoints/best.pt")
    print("Saved ppo_checkpoints/best.pt")

    # ── plots ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(log["iter"], log["p_loss"],  label="policy loss", color="steelblue")
    axes[0].plot(log["iter"], log["v_loss"],  label="value loss",  color="coral")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Iteration")

    axes[1].plot(log["iter"], log["entropy"], color="purple")
    axes[1].set_title("Entropy"); axes[1].set_xlabel("Iteration")

    axes[2].plot(log["iter"], log["avg_reward"], color="seagreen")
    axes[2].set_title("Avg Reward per Game"); axes[2].set_xlabel("Iteration")
    axes[2].axhline(0, color='gray', linestyle='--', linewidth=0.8)

    its = np.array(log["iter"])
    axes[3].bar(its - 0.25, log["p0_wins"], 0.25, label="P0 wins", color="steelblue")
    axes[3].bar(its,        log["p1_wins"], 0.25, label="P1 wins", color="coral")
    axes[3].bar(its + 0.25, log["draws"],   0.25, label="Draws",   color="gray")
    axes[3].set_title("Game Outcomes"); axes[3].legend(); axes[3].set_xlabel("Iteration")

    plt.tight_layout()
    plt.savefig("ppo_training_curves.png", dpi=150)
    print("Plot saved to ppo_training_curves.png")