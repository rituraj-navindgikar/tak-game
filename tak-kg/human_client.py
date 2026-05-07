import socket
import sys
import json
import argparse
from Game import Game


def main():
    parser = argparse.ArgumentParser(description='Human Tak client')
    parser.add_argument('ip', type=str)
    parser.add_argument('port', type=int)
    parser.add_argument('-n', dest='n', type=int, default=5, help='Board size (default: 5)')
    args = parser.parse_args()

    s = socket.socket()
    s.connect((args.ip, args.port))
    s.settimeout(500)

    def recv():
        data = s.recv(4096)
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8", errors="replace")
        return json.loads(data)

    def send(action, data, meta=""):
        msg = json.dumps({"action": action, "data": data, "meta": meta})
        s.sendall(msg.encode("utf-8"))

    # receive init
    msg = recv()
    init = msg['data'].strip().split()
    player_id = init[0]
    n = int(init[1])
    time_limit = int(init[2])
    print(f"You are player {player_id} on a {n}x{n} board. Time limit: {time_limit}s")

    game = Game(n, 'CUI')

    # if player 2, wait for player 1's first move
    if player_id == '2':
        msg = recv()
        if msg['action'] == 'KILLPROC' or msg['action'] == 'FINISH':
            print("Game over before it started.")
            s.close()
            return
        opp_move = msg['data'].strip()
        print(f"Opponent played: {opp_move}")
        game.execute_move(opp_move)

    while True:
        # your turn
        while True:
            sys.stderr.write("Your move: ")
            sys.stderr.flush()
            move = sys.stdin.readline().strip()
            if not move:
                continue
            result = game.execute_move(move)
            if result == 0:
                sys.stderr.write("Invalid move, try again.\n")
                sys.stderr.flush()
                # undo is not supported, re-ask
                continue
            break

        print(f"You played: {move}")

        if result in (2, 3, 4):
            if result == 2:
                print("Player 1 wins!")
            elif result == 3:
                print("Player 2 wins!")
            else:
                print("Draw!")
            send("FINISH", move)
            break
        else:
            send("NORMAL", move)

        # opponent's turn
        msg = recv()
        if msg['action'] == 'KILLPROC':
            print("Opponent error: " + msg.get('meta', ''))
            break
        if msg['action'] == 'FINISH':
            opp_move = msg['data'].strip()
            print(f"Opponent played: {opp_move}")
            game.execute_move(opp_move)
            print("Game over.")
            break

        opp_move = msg['data'].strip()
        print(f"Opponent played: {opp_move}")
        result = game.execute_move(opp_move)
        if result in (2, 3, 4):
            print("Game over.")
            break

    s.close()


if __name__ == '__main__':
    main()