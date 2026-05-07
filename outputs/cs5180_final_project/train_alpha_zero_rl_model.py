import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque
import copy, math, random, time
import os
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as mp

N = 5
MAX_FLATS = 21
MAX_STACK = N

OBS = N*N*5 + 4   # 5 * 5 * 5 + 4 = 129

## 5. MCTS
C_PUCT = 1.5
EPS    = 1e-8

# number of cpus
NUM_CPUS = 16

MAX_GAME_MOVES = 150


# ---------- config ----------
ITERATIONS  = 100
GAMES       = 100
SIMS        = 50
EPOCHS      = 20
BATCH       = 512
EVAL_GAMES  = 10
LR          = 1e-4
BUFFER_MAX  = 100000
ACCEPT_THR  = 0.0
HIDDEN      = 256


# multi processing config
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

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
        e = TakEnv.__new__(TakEnv)
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
        moves = []
        opening = self.moves < 2   # moves==0 or moves==1, no walls or stacks

        if self.moves == 0:
            owner = 1 - self.turn
        elif self.moves == 1 and self.turn == 1:
            owner = 1 - self.turn
        else:
            owner = self.turn

        for idx in range(N * N):
            sq = self._i2s(idx)
            if not self.board[idx]:
                if self.flats[owner] > 0:
                    moves.append('F' + sq)
                    if not opening:
                        moves.append('S' + sq)
            if not opening and self.board[idx] and self.board[idx][-1][0] == self.turn:
                moves += self._stack_moves(idx)
        return moves

    def _apply(self, move):
        if not move: return False
        if move[0].isalpha():  return self._place(move)
        if move[0].isdigit():  return self._move_stack(move)
        return False

    def _place(self, move):
        t = move[0]; idx = self._s2i(move[1:])
        if idx == -1 or self.board[idx] or t not in ('F', 'S'):
            return False
        opening = self.moves < 2
        if t == 'S' and opening:
            return False
        if self.moves == 0:
            owner = 1 - self.turn
        elif self.moves == 1 and self.turn == 1:
            owner = 1 - self.turn
        else:
            owner = self.turn
        if self.flats[owner] <= 0:
            return False
        self.board[idx].append((owner, t))
        self.flats[owner] -= 1
        if self.turn == 0:       # only increment when P0 moves
            self.moves += 1
        self.turn = 1 - self.turn
        return True

    def _move_stack(self, move):
        if len(move) < 5: return False
        try:
            count = int(move[0]); drops = [int(c) for c in move[4:]]
        except ValueError: return False
        direction = move[3]
        if direction not in ('+','-','<','>'): return False
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
            self.board[nxt] += hand[taken:taken+d]
            taken += d; cur = nxt
        if self.turn == 0:
            self.moves += 1
        self.turn = 1 - self.turn
        return True

    def _stack_moves(self, src):
        moves = []; sq = self._i2s(src)
        for count in range(1, min(len(self.board[src]), MAX_STACK) + 1):
            for d in ('+','-','<','>'):
                chg = self._dc(d); pl = 0; cur = src
                for _ in range(N-1):
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
            for d in range(1, rem+1):
                cur.append(d); _go(rem-d, ml-1, cur); cur.pop()
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
            if w == just:  return  1.0, True
            if w == 2:     return  0.0, True
            return -1.0, True
        return 0.0, False

    def _road(self, p):
        def dfs(starts, ends):
            vis = set(); stk = []
            for s in starts:
                if self.board[s] and self.board[s][-1] == (p,'F'):
                    vis.add(s); stk.append(s)
            while stk:
                c = stk.pop()
                if c in ends: return True
                for nb in self._nb(c):
                    if nb not in vis and self.board[nb] and self.board[nb][-1] == (p,'F'):
                        vis.add(nb); stk.append(nb)
            return False
        return dfs([i*N for i in range(N)], {(i+1)*N-1 for i in range(N)}) or \
               dfs(list(range(N)),           {N*N-1-i    for i in range(N)})

    def _flat_win(self):
        c = [0,0]
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

    def _i2s(self, i):
        return chr(i // N + ord('a')) + str(i % N + 1)

    def _dc(self, d):
        return {'+':1, '-':-1, '>':N, '<':-N}[d]

    def _vs(self, src, dst, d):
        if dst < 0 or dst >= N*N: return False
        if d == '>' and src % N == N-1: return False
        if d == '<' and src % N == 0:   return False
        if d == '+' and src % N == N-1: return False
        if d == '-' and src % N == 0:   return False
        return True

    def _nb(self, i):
        r,c = i%N, i//N; nb = []
        if r > 0:   nb.append(i-1)
        if r < N-1: nb.append(i+1)
        if c > 0:   nb.append(i-N)
        if c < N-1: nb.append(i+N)
        return nb

"""## 2. Action Space  ( around 144 moves)"""

class ActionSpace:
    def __init__(self):
        seen = set(); env = TakEnv().reset()
        for col in range(N):
            for row in range(N):
                sq = chr(col+ord('a')) + str(row+1)
                seen.add('F'+sq); seen.add('S'+sq)
        for src in range(N*N):
            sq = env._i2s(src)
            for count in range(1, N+1):
                for d in ('+','-','<','>'):
                    chg = env._dc(d); pl = 0; cur = src
                    for _ in range(N-1):
                        nxt = cur+chg
                        if not env._vs(cur,nxt,d): break
                        pl += 1; cur = nxt
                    for drops in env._partitions(count, pl):
                        seen.add(str(count)+sq+d+''.join(str(x) for x in drops))
        self.moves    = sorted(seen)
        self.move2idx = {m:i for i,m in enumerate(self.moves)}

    @property
    def size(self): return len(self.moves)
    def encode(self, m): return self.move2idx[m]
    def decode(self, i): return self.moves[i]
    def mask(self, env):
        legal = set(env.legal_moves())
        return [m in legal for m in self.moves]

"""## 3. State Encoder  (49 floats)"""

def encode(env):
    v = np.zeros(OBS, dtype=np.float32)
    for i in range(N*N):
        b = i*5; sq = env.board[i]; h = len(sq)
        if h:
            top = sq[-1]
            v[b]   = 1. if top[0]==0 else -1.
            v[b+1] = 1. if top[1]=='F' else -1.
            v[b+2] = h / MAX_STACK
            p0 = sum(1 for p,_ in sq if p==0)
            v[b+3] = p0/h; v[b+4] = (h-p0)/h
    v[125] = 1. if env.turn==0 else -1.
    v[126] = env.flats[0]/MAX_FLATS
    v[127] = env.flats[1]/MAX_FLATS
    # v[128] = min(env.moves, 20) / 20.
    v[128] = min(env.moves, 50) / 50.
    return v

# road progress helper ─────────────────────────────────────────

def road_progress(env, player):
    """Largest connected component of player's flat stones, normalized 0-1."""
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

"""## 4. Network"""

# Call action space
ASP = ActionSpace()

class TakNet(nn.Module):
    def __init__(self, obs=OBS, actions=ASP.size, h=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h),   nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h),   nn.LayerNorm(h), nn.ReLU(),
        )
        self.pi = nn.Sequential(nn.Linear(h,h//2), nn.ReLU(), nn.Linear(h//2, actions))
        self.v  = nn.Sequential(nn.Linear(h,h//2), nn.ReLU(), nn.Linear(h//2,1), nn.Tanh())
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h)

    @torch.no_grad()
    def predict(self, env, device='cpu'):
        self.eval()
        x = torch.tensor(encode(env), dtype=torch.float32).unsqueeze(0).to(device)
        logits, v = self.forward(x)
        logits = logits.squeeze(0)
        mask   = torch.tensor(ASP.mask(env), dtype=torch.bool, device=device)
        logits[~mask] = float('-inf')
        probs  = F.softmax(logits, dim=0).cpu().numpy()
        return probs, float(v.item())

class Node:
    __slots__ = ('env','parent','move','children','N','W','P','expanded','terminal','value')
    def __init__(self, env, parent=None, move=None, prior=0.):
        self.env      = env
        self.parent   = parent
        self.move     = move
        self.children = {}
        self.N        = 0
        self.W        = 0.
        self.P        = prior
        self.expanded = False
        self.terminal = env.done
        self.value    = None

    @property
    def Q(self): return self.W / (self.N + EPS)

class MCTS:
    def __init__(self, net, n_sims=50, device='cpu'):
        self.net    = net
        self.n_sims = n_sims
        self.device = device

    def run(self, env, add_noise=False):
        root = Node(env.clone())
        self._expand(root)

        if add_noise and root.children:
            # add dirichlet noise to root priors for exploration
            moves = list(root.children.keys())
            noise = np.random.dirichlet([0.3] * len(moves))
            for mv, n in zip(moves, noise):
                root.children[mv].P = 0.75 * root.children[mv].P + 0.25 * n

        for _ in range(self.n_sims):
            self._sim(root)

        pi = np.zeros(ASP.size, dtype=np.float32)
        for mv, ch in root.children.items():
            pi[ASP.encode(mv)] = ch.N
        s = pi.sum()
        if s > 0: pi /= s
        return pi

    def best_move(self, env, temp=1.0, add_noise=False):
        pi = self.run(env, add_noise=add_noise)
        if temp == 0:
            idx = int(np.argmax(pi))
        else:
            pt  = pi ** (1./temp)
            pt /= pt.sum() + EPS
            idx = int(np.random.choice(len(pi), p=pt))
        return ASP.decode(idx), pi

    def _sim(self, node):
        path = [node]; cur = node
        while cur.expanded and not cur.terminal:
            cur = self._select(cur); path.append(cur)
        if cur.terminal:
            self._expand(cur)
            value = cur.value if cur.value is not None else 0.
        else:
            self._expand(cur)
            _, value = self.net.predict(cur.env, self.device)
        for n in reversed(path):
            n.N += 1; n.W += value; value = -value

    def _expand(self, node):
        if node.terminal:
            w    = node.env.winner
            last = 1 - node.env.turn
            node.value    = 0. if w==2 else (1. if w==last else -1.)
            node.expanded = True
            return
        probs, _ = self.net.predict(node.env, self.device)
        for mv in node.env.legal_moves():
            ce = node.env.clone(); ce.step(mv)
            node.children[mv] = Node(ce, parent=node, move=mv, prior=float(probs[ASP.encode(mv)]))
        node.expanded = True

    def _select(self, node):
        sqN = math.sqrt(node.N + EPS)
        best_s, best_c = -float('inf'), None
        for ch in node.children.values():
            s = ch.Q + C_PUCT * ch.P * sqN / (1 + ch.N)
            if s > best_s: best_s = s; best_c = ch
        return best_c

def play_game(net, n_sims=50, temp_cutoff=10, device='cpu'):
    mcts = MCTS(net, n_sims, device=device)
    env  = TakEnv().reset()
    hist = []
    for step in range(MAX_GAME_MOVES):
        temp = 1.0 if step < temp_cutoff else 0.0
        mv, pi = mcts.best_move(env, temp=temp, add_noise=True)
        prog0 = road_progress(env, 0)
        prog1 = road_progress(env, 1)
        hist.append((encode(env), pi, env.turn, prog0, prog1))
        _, done = env.step(mv)
        if done: break
    if not env.done: env.winner = 2

    w = env.winner
    examples = []
    for (vec, pi, player, p0, p1) in hist:
        if w == 2:        z = 0.
        elif w == player: z = 1.
        else:             z = -1.
        # use snapshotted progress at that moment in the game
        progress = (p0 - p1) if player == 0 else (p1 - p0)
        z = 0.9 * z + 0.1 * progress
        examples.append((vec, pi, z))
    return examples, w

def play_game_worker(args):
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

    net_state, n_sims, device = args
    net = TakNet(h=HIDDEN).to(device)
    net.load_state_dict(net_state)
    net.eval()
    return play_game(net, n_sims=n_sims, device=device)

class Buffer:
    def __init__(self, maxlen=BUFFER_MAX):
        self.buf = deque(maxlen=maxlen)
    def add(self, ex): self.buf.extend(ex)
    def __len__(self):  return len(self.buf)
    def all(self):      return list(self.buf)

class TakDS(Dataset):
    def __init__(self, ex): self.ex = ex
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        v,pi,z = self.ex[i]
        return torch.tensor(v,dtype=torch.float32), \
               torch.tensor(pi,dtype=torch.float32), \
               torch.tensor([z],dtype=torch.float32)

def train_step(net, opt, examples, batch=64):
    net.train()
    loader = DataLoader(TakDS(examples), batch_size=batch, shuffle=True)
    pl, vl = [], []
    for s, pi, z in loader:
        s, pi, z = s.to(device), pi.to(device), z.to(device)
        logits, val = net(s)
        lp     = F.log_softmax(logits, dim=1)
        p_loss = -(pi * lp).sum(1).mean()
        v_loss = F.mse_loss(val, z)
        loss   = p_loss + v_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()
        pl.append(p_loss.item()); vl.append(v_loss.item())
    return float(np.mean(pl)), float(np.mean(vl))

def pit(net_new, net_old, n_games=20, n_sims=50, device='cpu'):
    nw, ow, d = 0, 0, 0
    m_new = MCTS(net_new, n_sims, device=device)
    m_old = MCTS(net_old, n_sims, device=device)
    for i in range(n_games):
        env = TakEnv().reset()
        agents = {i%2: m_new, (i+1)%2: m_old}
        new_p  = i % 2
        for _ in range(MAX_GAME_MOVES):
            mv, _ = agents[env.turn].best_move(env, temp=0)
            _, done = env.step(mv)
            if done: break
        w = env.winner
        if w == 2:     d  += 1
        elif w==new_p: nw += 1
        else:          ow += 1
    return nw, ow, d


os.makedirs("tak_checkpoints", exist_ok=True)

"""## 7. Training Loop"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# ----------------------------

net = TakNet(h=HIDDEN).to(device)

# prev checkpoints
# net.load_state_dict(torch.load("checkpoints/alpha-zero-v1.pt", map_location=device))

opt      = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERATIONS, eta_min=1e-4)
buf      = Buffer(BUFFER_MAX)


best_net = copy.deepcopy(net)

log = {"iter":[], "p_loss":[], "v_loss":[], "p0_wins":[], "p1_wins":[], "draws":[], "avg_reward":[]}

iter_bar = tqdm(range(1, ITERATIONS+1), desc="iterations")

with mp.Pool(processes=NUM_CPUS) as pool:
    for it in iter_bar:
        t0 = time.time()

        # --- self-play ---
        rewards = []
        w_count = [0, 0, 0]
        w_count = [0, 0, 0]
        
        net_state = {k: v.cpu() for k, v in net.state_dict().items()}
        args = [(net_state, SIMS, 'cpu') for _ in range(GAMES)]


        results = list(
            tqdm(
                pool.imap_unordered(play_game_worker, args, chunksize=1),
                total=GAMES,
                desc=f"[{it:2d}/{ITERATIONS}] self-play",
                leave=False
            )
        )

        for ex, winner in results:
            buf.add(ex)
            w_count[winner] += 1
            rewards.append(sum(e[2] for e in ex) / len(ex))

        # --- train ---
        p_losses, v_losses = [], []
        for _ in tqdm(range(EPOCHS), desc=f"[{it:2d}/{ITERATIONS}] training ", leave=False):
            pl, vl = train_step(net, opt, buf.all(), BATCH)
            p_losses.append(pl); v_losses.append(vl)

        # --- eval ---
        nw, ow, d = pit(net, best_net, EVAL_GAMES, SIMS, device=device)
        total  = nw + ow + d
        # accept = total > 0 and (nw / total) >= ACCEPT_THR
        # if accept:
        #     best_net = copy.deepcopy(net)

        best_net = copy.deepcopy(net)
        accept = True
        # --- log ---
        log["iter"].append(it)
        log["p_loss"].append(np.mean(p_losses))
        log["v_loss"].append(np.mean(v_losses))
        log["p0_wins"].append(w_count[0])
        log["p1_wins"].append(w_count[1])
        log["draws"].append(w_count[2])
        log["avg_reward"].append(np.mean(rewards))

        iter_bar.set_postfix(
            p_loss  = f"{log['p_loss'][-1]:.4f}",
            v_loss  = f"{log['v_loss'][-1]:.4f}",
            avg_r   = f"{log['avg_reward'][-1]:.3f}",
            eval    = f"{nw}-{ow}-{d}",
            status  = "yes" if accept else "no",
            elapsed = f"{time.time()-t0:.0f}s"
        )
        scheduler.step()

        # Save checkpoints
        if it % 2 == 0:
            torch.save(best_net.state_dict(), f"tak_checkpoints/best.pt")
            torch.save(best_net.state_dict(), f"tak_checkpoints/iter_{it:04d}.pt")
            print(f"  checkpoint saved at iter {it}")

"""## 8. Save Model + Plot"""

os.makedirs("tak_checkpoints", exist_ok=True)
torch.save(best_net.state_dict(), "tak_checkpoints/best.pt")
print("Saved tak_checkpoints/best.pt")



fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(log["iter"], log["p_loss"],  label="policy loss", color="steelblue")
axes[0].plot(log["iter"], log["v_loss"],  label="value loss",  color="coral")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Iteration")

axes[1].plot(log["iter"], log["avg_reward"], color="seagreen")
axes[1].set_title("Avg Reward per Game"); axes[1].set_xlabel("Iteration")
axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)

its = np.array(log["iter"])
axes[2].bar(its - 0.25, log["p0_wins"], 0.25, label="P0 wins", color="steelblue")
axes[2].bar(its,        log["p1_wins"], 0.25, label="P1 wins", color="coral")
axes[2].bar(its + 0.25, log["draws"],   0.25, label="Draws",   color="gray")
axes[2].set_title("Game Outcomes per Iteration")
axes[2].legend(); axes[2].set_xlabel("Iteration")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("Plot saved to training_curves.png")
