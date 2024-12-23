import math

# 黒と白の石を定義
BLACK = 1
WHITE = 2

# 初期盤面を定義（8x8）
board = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 2, 0, 0, 0],
    [0, 0, 0, 2, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]

# 石を置けるか判定
def can_place_x_y(board, stone, x, y):
    if board[y][x] != 0:
        return False

    opponent = 3 - stone
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        found_opponent = False

        while 0 <= nx < len(board[0]) and 0 <= ny < len(board) and board[ny][nx] == opponent:
            nx += dx
            ny += dy
            found_opponent = True

        if found_opponent and 0 <= nx < len(board[0]) and 0 <= ny < len(board) and board[ny][nx] == stone:
            return True

    return False

# 有効な手を取得
def valid_moves(board, stone):
    moves = []
    for y in range(len(board)):
        for x in range(len(board[0])):
            if can_place_x_y(board, stone, x, y):
                moves.append((x, y))
    return moves

# 評価関数
def count_stable_stones(board, stone):
    stable_stones = 0
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(len(board)):
        for x in range(len(board[0])):
            if board[y][x] == stone:
                is_stable = True
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    while 0 <= nx < len(board[0]) and 0 <= ny < len(board):
                        if board[ny][nx] == 0:  # 空きマスがあると不安定
                            is_stable = False
                            break
                        nx += dx
                        ny += dy
                    if not is_stable:
                        break
                if is_stable:
                    stable_stones += 1
    return stable_stones

def evaluate_board_with_stability(board, stone):
    value_table = [
        [100, -20,  10,   5,   5,  10, -20, 100],
        [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
        [ 10,  -2,   1,   1,   1,   1,  -2,  10],
        [  5,  -2,   1,   0,   0,   1,  -2,   5],
        [  5,  -2,   1,   0,   0,   1,  -2,   5],
        [ 10,  -2,   1,   1,   1,   1,  -2,  10],
        [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
        [100, -20,  10,   5,   5,  10, -20, 100],
    ]
    base_score = 0
    for y in range(len(board)):
        for x in range(len(board[0])):
            if board[y][x] == stone:
                base_score += value_table[y][x]
            elif board[y][x] == (3 - stone):
                base_score -= value_table[y][x]

    # 安定石を加味
    stable_score = count_stable_stones(board, stone) * 10
    return base_score + stable_score

# 改良されたEagerAI4
class EagerAI4:
    def face(self):
        return "🧠⚡"

    def place(self, board, stone):
        moves = valid_moves(board, stone)
        if not moves:
            return None

        best_move = None
        best_score = -math.inf

        for x, y in moves:
            new_board = [row[:] for row in board]
            new_board[y][x] = stone
            self.flip_stones(new_board, stone, x, y)

            # 改良版評価関数を使用
            score = evaluate_board_with_stability(new_board, stone)
            if score > best_score:
                best_score = score
                best_move = (x, y)

        return best_move

    def flip_stones(self, board, stone, x, y):
        opponent = 3 - stone
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            stones_to_flip = []

            while 0 <= nx < len(board[0]) and 0 <= ny < len(board):
                if board[ny][nx] == opponent:
                    stones_to_flip.append((nx, ny))
                elif board[ny][nx] == stone:
                    for flip_x, flip_y in stones_to_flip:
                        board[flip_y][flip_x] = stone
                    break
                else:
                    break
                nx += dx
                ny += dy

# 実行: EagerAI4を動かす
from kogi_canvas import play_othello

ai4 = EagerAI4()
play_othello(ai4, board=board)
