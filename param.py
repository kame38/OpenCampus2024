# constat parameters
import numpy as np

""" CASE1: ペルチェ(peltie)霧箱 """

# ================= take_photo.py ==============================
GRAY_THR = 40  # 濃度変化の閾値
FILTER_SIZE = 5  # medianフィルタのサイズ
CUT_MODE = True  # True:検出物体を切り取って保存, False:画像全体をそのまま保存
RHO_HOUGH = 10  # Hough変換の距離解像度
THETA_HOUGH = 10 * np.pi / 180  # Hough変換の角度解像度
COUNT_HOUGH = 300  # Hough変換の閾値
MIN_LEN_HOUGH = 50  # 検出する直線の最小長さ
MAX_GAP_HOUGH = 200  # 直線として認識する最大の間隔
PADDING = 50  # 枠の大きさに余裕を持たせる
RAY_COUNT_MAX = 2  # バッチサイズ(一度に検出する物体の数)

# ================= label_img.py ==============================
SRC_DIR = "image_data/tmp"
DEST_ALPHA = "image_data/all/alpha"
DEST_COSMIC = "image_data/all/cosmic"
DEST_NOISE = "image_data/all/noise"

# ================= train_net.py ==============================
DATA_DIR = "image_data/all"  # 画像フォルダ名
CKPT_PROCESS = "train_process.ckpt"  # 学習経過保存ファイル名
CKPT_NET = "trained_net.ckpt"  # 学習済みパラメータファイル名
NUM_CLASSES = 3  # クラス数
NUM_EPOCHS = 100  # 学習回数
LEARNING_RATE = 0.01  # 学習率

# ================= realtime_classification.py ==============================
# CKPT_NET
OBJ_NAMES = ["alpha", "cosmic", "noise"]  # 各クラスの表示名
SHOW_COLOR = (255, 191, 0)  # 枠の色(B,G,R) green
CHANNELS = 1  # 色のチャンネル数(BGR:3, グレースケール:1)
# RHO_HOUGH
# THETA_HOUGH
# COUNT_HOUGH
# MIN_LEN_HOUGH
# MAX_GAP_HOUGH
# GRAY_THR
# RAY_COUNT_MAX
# NUM_CLASSES
# PADDING
# FILTER_SIZE

# ============== conf_matrix.py =============================
TEST_DATA_DIR = "image_data/test"  # テストデータのフォルダ名
