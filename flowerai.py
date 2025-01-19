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
# 強いAIの代わりにランダムAIを模倣した例を使用（強いAIに置き換え可）
class RandomAI:
    def place(self, board, stone):
        moves = get_valid_moves(board, stone)
        return random.choice(moves) if moves else None

# データ生成
ai = RandomAI()
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

class FlowerAI:
    def __init__(self, model_path='strong_othello_model_6x6.h5'):
        # 訓練済みモデルをロード
        self.model = load_model(model_path)

    def face(self):
        return "🌼"

    def place(self, board, stone):
        # ボードデータをモデルが処理できる形式に変換
        input_board = np.array(board).reshape(-1, 6, 6, 1)  # 6x6の入力形式
        predictions = self.model.predict(input_board)

        # 最適な手のインデックスを取得
        move_index = np.argmax(predictions)
        x, y = move_index % 6, move_index // 6  # インデックスを(x, y)形式に変換

        # 有効な手であることを確認
        valid_moves = self.valid_moves(board, stone)
        if (x, y) in valid_moves:
            return x, y
        else:
            # 有効な手がない場合は最初の有効な手を選ぶ
            return valid_moves[0] if valid_moves else None

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
