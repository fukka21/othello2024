import math

BLACK = 1
WHITE = 2

board = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 0, 0],
    [0, 0, 2, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

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

def valid_moves(board, stone):
    moves = []
    for y in range(len(board)):
        for x in range(len(board[0])):
            if can_place_x_y(board, stone, x, y):
                moves.append((x, y))
    return moves

def evaluate_board(board, stone):
    # 評価関数
    value_table = [
        [100, -20, 10, 10, -20, 100],
        [-20, -50,  1,  1, -50, -20],
        [ 10,   1,  5,  5,   1,  10],
        [ 10,   1,  5,  5,   1,  10],
        [-20, -50,  1,  1, -50, -20],
        [100, -20, 10, 10, -20, 100],
    ]

    score = 0
    for y in range(len(board)):
        for x in range(len(board[0])):
            if board[y][x] == stone:
                score += value_table[y][x]
            elif board[y][x] == (3 - stone):
                score -= value_table[y][x]
    return score

class StrategicAI:
    def face(self):
        return "🤖"

    def place(self, board, stone):
        # 有効な手を取得
        moves = valid_moves(board, stone)
        if not moves:
            return None

        # 各手を評価し、最高スコアの手を選ぶ
        best_move = None
        best_score = -math.inf
        for x, y in moves:
            # 仮に置いてみる
            new_board = [row[:] for row in board]
            new_board[y][x] = stone

            # この盤面を評価
            score = evaluate_board(new_board, stone)
            if score > best_score:
                best_score = score
                best_move = (x, y)

        return best_move
