import math

BLACK = 1
WHITE = 2

# 初期盤面
board = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 0, 0],
    [0, 0, 2, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

# 位置価値表
value_table = [
    [100, -20, 10, 10, -20, 100],
    [-20, -50,  1,  1, -50, -20],
    [ 10,   1,  5,  5,   1,  10],
    [ 10,   1,  5,  5,   1,  10],
    [-20, -50,  1,  1, -50, -20],
    [100, -20, 10, 10, -20, 100],
]

# 有効手の判定
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

# 評価関数の要素
def count_stones(board, stone):
    my_stones = sum(row.count(stone) for row in board)
    opponent_stones = sum(row.count(3 - stone) for row in board)
    return my_stones - opponent_stones

def count_stable_stones(board, stone):
    # 簡易的な安定石カウント（角のみ）
    stable_count = 0
    corners = [(0, 0), (0, 5), (5, 0), (5, 5)]
    for x, y in corners:
        if board[y][x] == stone:
            stable_count += 1
    return stable_count

def evaluate_board(board, stone):
    position_score = sum(
        value_table[y][x] if board[y][x] == stone else -value_table[y][x]
        for y in range(len(board))
        for x in range(len(board[0]))
    )
    stone_diff = count_stones(board, stone)
    stable_stone_diff = count_stable_stones(board, stone)
    return position_score + stone_diff * 10 + stable_stone_diff * 50

# ミニマックスアルゴリズム + アルファベータ枝刈り
def alpha_beta(board, stone, depth, alpha, beta, maximizing_player):
    if depth == 0:
        return evaluate_board(board, stone)
    
    moves = valid_moves(board, stone)
    if not moves:
        return evaluate_board(board, stone)
    
    if maximizing_player:
        max_eval = -math.inf
        for x, y in moves:
            new_board = [row[:] for row in board]
            new_board[y][x] = stone
            eval = alpha_beta(new_board, 3 - stone, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for x, y in moves:
            new_board = [row[:] for row in board]
            new_board[y][x] = stone
            eval = alpha_beta(new_board, 3 - stone, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# 探索深さを決定
def determine_depth(board):
    total_stones = sum(row.count(1) + row.count(2) for row in board)
    if total_stones <= 10:
        return 2  # 序盤
    elif total_stones <= 24:
        return 4  # 中盤
    else:
        return 6  # 終盤

# 改良AIクラス
class AdvancedAI:
    def face(self):
        return "🌟"

    def place(self, board, stone):
        moves = valid_moves(board, stone)
        if not moves:
            return None
        
        best_move = None
        best_score = -math.inf
        depth = determine_depth(board)
        for x, y in moves:
            new_board = [row[:] for row in board]
            new_board[y][x] = stone
            score = alpha_beta(new_board, 3 - stone, depth, -math.inf, math.inf, False)
            if score > best_score:
                best_score = score
                best_move = (x, y)

        return best_move
