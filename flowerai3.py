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

def generate_data_with_strong_ai(ai, num_samples=1000):
    X_train = []
    Y_train = []
    for _ in range(num_samples):
        # ランダムに初期盤面を生成
        board = [[random.choice([0, 1, 2]) for _ in range(6)] for _ in range(6)]
        board[2][2], board[2][3], board[3][2], board[3][3] = 1, 2, 2, 1  # 中央は固定

        stone = random.choice([1, 2])  # ランダムに黒か白を選ぶ

        # 強いAIに最適な手を計算させる
        move = ai.place(board, stone)
        if move is None:
            continue

        # データに追加
        X_train.append(board)
        Y_train.append(move[1] * 6 + move[0])  # (y * 6 + x)形式で記録
    return X_train, Y_train

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

def create_resnet_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(36, activation='softmax')(x)  # 6×6盤用
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
class MCTS_AI:
    def __init__(self, simulations=100):
        self.simulations = simulations

    def place(self, board, stone):
        valid_moves = get_valid_moves(board, stone)
        if not valid_moves:
            return None

        # シミュレーションごとに勝率を評価
        move_scores = {move: 0 for move in valid_moves}
        for move in valid_moves:
            for _ in range(self.simulations):
                result = self.simulate(board, stone, move)
                move_scores[move] += result

        # 最も勝率が高い手を選択
        return max(move_scores, key=move_scores.get)

    def simulate(self, board, stone, move):
        # ボードをコピーして手を試す
        board_copy = np.array(board)
        x, y = move
        board_copy[y][x] = stone

        # ランダムに試合を進行し、勝利したら1を返す（単純化した例）
        current_stone = stone
        while get_valid_moves(board_copy, current_stone):
            moves = get_valid_moves(board_copy, current_stone)
            random_move = random.choice(moves)
            x, y = random_move
            board_copy[y][x] = current_stone
            current_stone = 3 - current_stone
        return 1 if self.evaluate_winner(board_copy) == stone else 0

    def evaluate_winner(self, board):
        # 最も石が多いプレイヤーを勝者とする
        flat_board = np.array(board).flatten()
        return 1 if np.sum(flat_board == 1) > np.sum(flat_board == 2) else 2


# データ生成
ai = MCTS_AI()
X_train, Y_train = generate_data_with_strong_ai(ai, num_samples=1000)

# データ拡張
X_train, Y_train = augment_data(X_train, Y_train)

# データ前処理
X_train, Y_train = preprocess_data(X_train, Y_train)

# モデル構築
input_shape = (6, 6, 1)
model = create_resnet_model(input_shape)

# 訓練データと検証データに分割
split_index = int(len(X_train) * 0.8)
X_train, X_val = X_train[:split_index], X_train[split_index:]
Y_train, Y_val = Y_train[:split_index], Y_train[split_index:]

# モデル訓練
model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=32,
    callbacks=[lr_scheduler, early_stopping]
)

# モデル保存
model.save('strong_othello_model_6x6.h5')

import numpy as np
from tensorflow.keras.models import load_model

class FlowerAI2:
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
        # 評価関数：角や辺を重視し、中央を少し優遇する
        x, y = move
        value = 0

        # 角のスコア
        if (x, y) in [(0, 0), (0, 5), (5, 0), (5, 5)]:
            value += 10  # 角は非常に強力

        # 辺のスコア
        elif x == 0 or x == 5 or y == 0 or y == 5:
            value += 5  # 辺は強い

        # 中央のスコア
        elif 2 <= x <= 3 and 2 <= y <= 3:
            value += 3  # 中央はそこそこ強い

        return value
