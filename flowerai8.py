import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

# ----------------------------
# 1. データ生成
# ----------------------------
def get_valid_moves(board, stone):
    moves = []
    opponent = 3 - stone
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(6):
        for x in range(6):
            if board[y][x] != 0:
                continue
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                found_opponent = False
                while 0 <= nx < 6 and 0 <= ny < 6 and board[ny][nx] == opponent:
                    nx += dx
                    ny += dy
                    found_opponent = True
                if found_opponent and 0 <= nx < 6 and 0 <= ny < 6 and board[ny][nx] == stone:
                    moves.append((x, y))
                    break
    return moves

# データ生成の高速化 (簡易AIに変更)
def generate_data_with_simple_ai(num_samples=200):
    X_train = []
    Y_train = []
    for _ in range(num_samples):
        board = np.zeros((6, 6), dtype=int)
        board[2][2], board[2][3], board[3][2], board[3][3] = 1, 2, 2, 1
        stone = random.choice([1, 2])
        valid_moves = get_valid_moves(board, stone)
        if not valid_moves:
            continue
        move = random.choice(valid_moves)
        X_train.append(board)
        Y_train.append(move[1] * 6 + move[0])
    return np.array(X_train), np.array(Y_train)

def augment_data(X, Y):
    augmented_X = []
    augmented_Y = []
    for board, move in zip(X, Y):
        for _ in range(4):  # 90度ずつ回転
            board = np.rot90(board)
            # moveは数値形式（y * 6 + x）で保持しているので再計算は不要
            augmented_X.append(board)
            augmented_Y.append(move)
        # 水平方向反転
        flipped_board = np.fliplr(board)
        augmented_X.append(flipped_board)
        # moveのx座標を反転
        x, y = move % 6, move // 6
        flipped_move = y * 6 + (5 - x)
        augmented_Y.append(flipped_move)
    return np.array(augmented_X), np.array(augmented_Y)

# ----------------------------
# 2. モデル構築
# ----------------------------
def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

# 簡略化したモデル構築
def create_lightweight_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(36, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ----------------------------
# 3. データ準備と前処理
# ----------------------------
def preprocess_data(X, Y):
    # X: 入力データを6×6×1に変換
    X = np.array(X).reshape(-1, 6, 6, 1)

    # Y: 出力データをone-hotエンコーディング
    Y = np.array(Y)
    Y = tf.keras.utils.to_categorical(Y, num_classes=36)

    return X, Y

# ----------------------------
# 4. モデルの訓練
# ----------------------------
def lr_schedule(epoch):
    initial_lr = 0.001
    decay_rate = 0.5
    decay_step = 10
    return initial_lr * (decay_rate ** (epoch // decay_step))

# 学習率スケジューラーと早期終了を設定
lr_scheduler = LearningRateScheduler(lr_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# ----------------------------
# メイン処理
# ----------------------------
import numpy as np
import random

class MCTS_AI:
    def __init__(self, model=None):
        self.model = model  # Optional: Trained model for enhanced evaluation

    def evaluate(self, board, player):
        """
        Evaluate the board position using either a neural network model or a manual heuristic.
        """
        if self.model:
            # Use the neural network model if provided
            input_data = self.prepare_input(board, player)
            score = self.model.predict(input_data)
            return score
        else:
            # Fallback to manual heuristic evaluation
            return self.manual_evaluation(board, player)

    def prepare_input(self, board, player):
        """
        Prepare the board and player state for the neural network input.
        """
        return np.array([board.flatten(), player])  # Example placeholder

    def manual_evaluation(self, board, player):
        """
        Manual heuristic evaluation of the board position.
        """
        opponent = 3 - player
        player_score = np.sum(board == player)
        opponent_score = np.sum(board == opponent)
        return player_score - opponent_score

    def get_valid_moves(self, board, player):
        """
        Generate a list of valid moves for the current player.
        """
        valid_moves = []
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if self.is_valid_move(board, x, y, player):
                    valid_moves.append((x, y))
        return valid_moves

    def is_valid_move(self, board, x, y, player):
        """
        Check if placing a piece at (x, y) is valid for the current player.
        """
        return board[x, y] == 0  # Example: Assume 0 indicates an empty spot

    def apply_move(self, board, move, player):
        """
        Apply the player's move to the board.
        """
        new_board = board.copy()
        x, y = move
        new_board[x, y] = player
        return new_board

    def simulate(self, board, player):
        """
        Simulate a random game from the current position until the end.
        """
        current_player = player
        while not self.is_terminal(board):
            valid_moves = self.get_valid_moves(board, current_player)
            if not valid_moves:
                break
            move = random.choice(valid_moves)
            board = self.apply_move(board, move, current_player)
            current_player = 3 - current_player
        return self.evaluate(board, player)

    def is_terminal(self, board):
        """
        Check if the game has reached a terminal state.
        """
        return not np.any(board == 0)  # Example: Terminal if no empty spots

    def mcts_search(self, board, player, iterations=100):
        """
        Perform MCTS to select the best move.
        """
        scores = {}
        counts = {}
        valid_moves = self.get_valid_moves(board, player)

        for move in valid_moves:
            scores[move] = 0
            counts[move] = 0

        for _ in range(iterations):
            move = random.choice(valid_moves)
            new_board = self.apply_move(board, move, player)
            score = self.simulate(new_board, 3 - player)
            scores[move] += score
            counts[move] += 1

        best_move = max(valid_moves, key=lambda m: scores[m] / counts[m])
        return best_move

# Example usage
if __name__ == "__main__":
    board = np.zeros((3, 3))  # Example: 3x3 board for simplicity
    ai = MCTS_AI()
    player = 1
    best_move = ai.mcts_search(board, player)
    print("Best move:", best_move)



    def evaluate_winner(self, board):
        # 最も石が多いプレイヤーを勝者とする
        flat_board = np.array(board).flatten()
        return 1 if np.sum(flat_board == 1) > np.sum(flat_board == 2) else 2


# データ生成
ai = MCTS_AI()
X_train, Y_train = generate_data_with_simple_ai(num_samples=200)

# データ拡張
X_train, Y_train = augment_data(X_train, Y_train)

# データ前処理
X_train, Y_train = preprocess_data(X_train, Y_train)

# モデル構築
input_shape = (6, 6, 1)
model = create_lightweight_model(input_shape)

# 訓練データと検証データに分割
split_index = int(len(X_train) * 0.8)
X_train, X_val = X_train[:split_index], X_train[split_index:]
Y_train, Y_val = Y_train[:split_index], Y_train[split_index:]

# モデル訓練
model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_split=0.2)

# モデル保存
model.save('strong_othello_model_6x6.h5')

import numpy as np
from tensorflow.keras.models import load_model

class FlowerAI8:
    def __init__(self, model_path='strong_othello_model_6x6.h5'):
        # 訓練済みモデルをロード
        self.model = load_model(model_path)

    def face(self):
        return "🌼"

    def place(self, board, stone):
        # ボードデータをモデルが処理できる形式に変換
        input_board = np.array(board).reshape(-1, 6, 6, 1)  # 6x6の入力形式
        predictions = self.model.predict(input_board)

        # 全ての手をスコアリングし、最適な手を選ぶ
        valid_moves = self.valid_moves(board, stone)
        if not valid_moves:
            return None  # 有効な手がない場合

        best_move = None
        best_score = -float('inf')  # スコアの初期値を低い値に設定
        for move in valid_moves:
            x, y = move
            move_index = y * 6 + x
            score = predictions[0][move_index] + self.evaluate(board, move, stone)  # モデルのスコアと評価関数を足す
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def valid_moves(self, board, stone):
        # 有効な手をリストアップ
        moves = []
        opponent = 3 - stone
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for y in range(6):
            for x in range(6):
                if board[y][x] != 0:
                    continue
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    found_opponent = False
                    while 0 <= nx < 6 and 0 <= ny < 6 and board[ny][nx] == opponent:
                        nx += dx
                        ny += dy
                        found_opponent = True
                    if found_opponent and 0 <= nx < 6 and 0 <= ny < 6 and board[ny][nx] == stone:
                        moves.append((x, y))
                        break
        return moves

    def evaluate(self, board, move, stone):
        x, y = move
        opponent = 3 - stone
        value = 0

        # 評価マトリックス（角・辺・中央を重視）
        evaluation_matrix = np.array([
            [100, -10, 10, 10, -10, 100],
            [-10, -50, 5, 5, -50, -10],
            [10, 5, 1, 1, 5, 10],
            [10, 5, 1, 1, 5, 10],
            [-10, -50, 5, 5, -50, -10],
            [100, -10, 10, 10, -10, 100],
        ])
        value += evaluation_matrix[y][x]

        # 安定石の評価
        stable_bonus = 0
        if (x, y) in [(0, 0), (0, 5), (5, 0), (5, 5)]:
            stable_bonus += 50  # 角の安定石は特に高評価
        value += stable_bonus

        # リスク領域のペナルティ（角の隣）
        risk_positions = [(0, 1), (1, 0), (0, 4), (1, 5), (4, 0), (5, 1), (5, 4), (4, 5)]
        if (x, y) in risk_positions:
            value -= 20  # リスクのある場所は評価を下げる

        # 次の手の自由度を考慮
        board_copy = np.array(board)
        board_copy[y][x] = stone
        next_valid_moves = self.valid_moves(board_copy, opponent)
        value -= len(next_valid_moves) * 10  # 相手の手数が増えると評価を下げる

        return value
