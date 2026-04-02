import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

N         = 3
MAX_FLATS = 10
MAX_STACK = N
OBS       = N * N * 5 + 4   # 49
EPS       = 1e-8
C_PUCT    = 1.5


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


# ── ActionSpace ──────────────────────────────────────────────────

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


# ── State encoder ────────────────────────────────────────────────

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
    v[45] = 1. if env.turn==0 else -1.
    v[46] = env.flats[0]/MAX_FLATS
    v[47] = env.flats[1]/MAX_FLATS
    v[48] = min(env.moves, 50) / 50.
    return v


# ── TakNet ───────────────────────────────────────────────────────

ASP = ActionSpace()

class TakNet(nn.Module):
    def __init__(self, obs=OBS, actions=ASP.size, h=128):
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


# ── MCTS ─────────────────────────────────────────────────────────

## 5. MCTS
C_PUCT = 1.5
EPS    = 1e-8

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

    def run(self, env):
        root = Node(env.clone())
        self._expand(root)
        for _ in range(self.n_sims):
            self._sim(root)
        pi = np.zeros(ASP.size, dtype=np.float32)
        for mv, ch in root.children.items():
            pi[ASP.encode(mv)] = ch.N
        s = pi.sum()
        if s > 0: pi /= s
        return pi

    def best_move(self, env, temp=1.0):
        pi   = self.run(env)
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


# ── load helper ──────────────────────────────────────────────────

def load_agent(model_path, hidden=128, n_sims=100, device='cpu'):
    """Load trained model and return a ready-to-use MCTS agent."""
    net = TakNet(h=hidden).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return MCTS(net, n_sims=n_sims, device=device)