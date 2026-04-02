import sys

sys.stdin.readline()

while True:
    try:
        sys.stderr.write("Your move: ")
        sys.stderr.flush()
        move = sys.stdin.readline().strip()
        if not move:
            break
        print(move)
        sys.stdout.flush()
    except:
        break