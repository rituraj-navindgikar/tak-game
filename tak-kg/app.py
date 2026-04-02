from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from Game import Game
from alpha_zero_rl_model import TakEnv, TakNet, MCTS
import torch

app = Flask(__name__)
CORS(app)

# ── config ───────────────────────────────────────────────────────
HIDDEN = 128
SIMS   = 100
device = 'cpu'

# ── load model once at startup ───────────────────────────────────
rl_net = TakNet(h=HIDDEN).to(device)
rl_net.load_state_dict(torch.load("../checkpoints/alpha-zero-best.pt", map_location=device, weights_only=False))
rl_net.eval()
agent = MCTS(rl_net, n_sims=SIMS, device=device)
print(f"RL model loaded -> {sum(p.numel() for p in rl_net.parameters()):,} params")

# ── game state ───────────────────────────────────────────────────
game         = None
move_history = []
game_over    = False
winner_msg   = ""


# ── helpers ──────────────────────────────────────────────────────

def rebuild_rl_env(history):
    """Replay all moves on a fresh TakEnv to get current agent state."""
    env = TakEnv().reset()
    for mv in history:
        if not env.done:
            env.step(mv)
    return env


def board_to_json(g):
    cells = []
    for idx in range(g.total_squares):
        stack = g.board[idx]
        cells.append({
            "idx":   idx,
            "col":   chr(idx % g.n + 97),
            "row":   idx // g.n + 1,
            "stack": [{"player": p, "type": t} for p, t in stack],
            "top":   {"player": stack[-1][0], "type": stack[-1][1]} if stack else None
        })
    return {
        "cells":      cells,
        "turn":       g.turn,
        "moves":      g.moves,
        "p1_flats":   g.players[0].flats,
        "p2_flats":   g.players[1].flats,
        "game_over":  game_over,
        "winner_msg": winner_msg
    }


def _winner(result):
    if result == 2: return "Player 1 (You) wins!"
    if result == 3: return "Player 2 (AI) wins!"
    return "Draw!"


def print_board(g, label=""):
    """Debug print board state to terminal."""
    print(f"\n{'='*40}")
    if label:
        print(f"  {label}")
    print(f"  Turn: P{g.turn+1}  |  Move#: {g.moves}  |  Flats: P1={g.players[0].flats} P2={g.players[1].flats}")
    print(f"{'='*40}")
    for row in range(g.n - 1, -1, -1):
        line = f"  {row+1} "
        for col in range(g.n):
            sq_str = chr(ord('a') + col) + str(row + 1)
            idx    = g.square_to_num(sq_str)   # use Game.py's own indexing
            sq     = g.board[idx]
            if not sq:
                cell = "."
            else:
                cell = "".join(f"{'W' if t=='S' else 'F'}{'1' if p==0 else '2'}" for p, t in sq)
            line += f"  {cell:<6}"
        print(line)
    cols = "     " + "  ".join(f"{chr(ord('a')+c):<8}" for c in range(g.n))
    print(cols)
    print()


# ── routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/new_game", methods=["POST"])
def new_game():
    global game, move_history, game_over, winner_msg
    game         = Game(3, "None")
    move_history = []
    game_over    = False
    winner_msg   = ""
    print("\n" + "="*40)
    print("  NEW GAME STARTED")
    print("="*40)
    return jsonify({"status": "ok", "board": board_to_json(game)})


@app.route("/move", methods=["POST"])
def move():
    global game_over, winner_msg, move_history

    if game_over:
        return jsonify({"error": "Game over. Start a new game."}), 400
    if game is None:
        return jsonify({"error": "Start a new game first."}), 400

    human_move = request.json.get("move", "").strip()

    # --- human move ---
    result = game.execute_move(human_move)
    if result == 0:
        print(f"  INVALID move by human: {human_move}")
        return jsonify({"error": "Invalid move."}), 400

    move_history.append(human_move)
    print(f"\n>>> HUMAN (P1) played: {human_move}")
    print_board(game, "After human move")

    if result in (2, 3, 4):
        game_over  = True
        winner_msg = _winner(result)
        print(f"  GAME OVER: {winner_msg}")
        return jsonify({"board": board_to_json(game), "ai_move": None})

    # --- rebuild TakEnv from full history, ask agent ---
    rl_env = rebuild_rl_env(move_history)
    print(f"  TakEnv: turn={rl_env.turn} moves={rl_env.moves} flats={rl_env.flats}")

    ai_move, _ = agent.best_move(rl_env, temp=0)

    # --- apply agent move on Game ---
    result = game.execute_move(ai_move)
    if result == 0:
        print(f"  INVALID move by agent: {ai_move}  (this is a bug)")
        return jsonify({"error": f"Agent produced invalid move: {ai_move}"}), 500

    move_history.append(ai_move)
    print(f">>> AI    (P2) played: {ai_move}")
    print_board(game, "After AI move")

    if result in (2, 3, 4):
        game_over  = True
        winner_msg = _winner(result)
        print(f"  GAME OVER: {winner_msg}")

    return jsonify({"board": board_to_json(game), "ai_move": ai_move})


if __name__ == "__main__":
    app.run(debug=True, port=5000)