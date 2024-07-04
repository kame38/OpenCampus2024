# coding: utf-8
import os
from PIL import Image
from time import sleep
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torchvision import transforms
import numpy as np

# ペルチェ霧箱用のリアルタイム飛跡検出プログラム
# 背景画像ははじめに設定し、固定する
# 背景画像の更新が少ないので、飛跡が多く、ノイズが少ない場合の物体検出に最適
# 液チ霧箱を使う場合は、260行目以降を使う


# from param import (
#     CKPT_NET,
#     OBJ_NAMES,
#     RHO_HOUGH,
#     THETA_HOUGH,
#     COUNT_HOUGH,
#     MIN_LEN_HOUGH,
#     MAX_GAP_HOUGH,
#     GRAY_THR,
#     RAY_COUNT_MAX,
#     SHOW_COLOR,
#     NUM_CLASSES,
#     CHANNELS,
#     PADDING,
#     FILTER_SIZE,
# )  # param.pyを確認!!

# from train_net import NeuralNet

# # 画像データ変換定義
# # Resizeと, classifierの最初のLinear入力が関連
# data_transforms = transforms.Compose(
#     [
#         transforms.Resize((112, 112)),
#         transforms.Grayscale(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5]),
#     ]
# )


# def detect_ray(back, target):
#     """
#     OpenCVの背景差分処理で, 検出された物体のタプルを作成
#     引数:
#     back: 入力背景画像(カラー)
#     target: 背景差分対象の画像(カラー) <-- 複数の物体を切り抜き, カラー画像タプルにまとめる
#     """
#     # 2値化
#     b_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
#     t_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
#     # 差分を計算
#     diff = cv2.absdiff(t_gray, b_gray)
#     # medianフィルターの適用
#     filter = cv2.medianBlur(diff, FILTER_SIZE)
#     # 閾値に従ってマスクを作成, 直線を抽出
#     mask = cv2.threshold(filter, GRAY_THR, 255, cv2.THRESH_BINARY)[1]
#     cv2.imshow("mask", mask)

#     # Hough変換を使った直線検出
#     lines = cv2.HoughLinesP(
#         mask,
#         RHO_HOUGH,
#         THETA_HOUGH,
#         COUNT_HOUGH,
#         MIN_LEN_HOUGH,
#         MAX_GAP_HOUGH,
#     )

#     # 検出された直線から2本選んで位置情報を抽出
#     pt_list = []
#     if lines is not None:
#         valid_rects = []
#         for line in lines:
#             for x1, y1, x2, y2 in line:
#                 valid_rects.append([x1, y1, x2, y2])

#         # 重なっている枠をまとめる(20%の重なりを許容)
#         rects, _ = cv2.groupRectangles(valid_rects, groupThreshold=1, eps=0.2)

#         for x1, y1, x2, y2 in rects[:2]:  # 2本までの直線を選ぶ
#             x, y, w, h = cv2.boundingRect(np.array([[x1, y1], [x2, y2]]))
#             pt_list.append((x, y, w, h))

#     pt_list = pt_list[:RAY_COUNT_MAX]

#     # 位置情報に従ってフレーム切り抜き, PIL画像のタプルに変換して返す
#     obj_images = tuple(
#         map(
#             lambda x: Image.fromarray(target[x[1] : x[1] + x[3], x[0] : x[0] + x[2]]),
#             pt_list,
#         )
#     )
#     return (obj_images, pt_list)


# def batch_maker(tuple_images, transform):
#     """
#     PIL形式の画像のタプルをtransformし, ネットワークで処理可能なtensorバッチに変換
#     引数:
#     tuple_images: PIL画像タプル
#     transform: torchvision画像変換定義
#     """
#     return torch.cat([transform(img) for img in tuple_images]).view(
#         -1, CHANNELS, 112, 112
#     )


# def judge_what(img, probs_list, pos_list):
#     """
#     各クラスに属する確率から物体を決定し, その位置に枠と名前を表示, クラスのインデックスを返す
#     引数:
#     probs_list: 確率の二次配列. バッチ形式
#     pos_list: 位置の二次配列. バッチ形式
#     """
#     # 最も高い確率とそのインデックスのリストに変換
#     ip_list = list(
#         map(
#             lambda x: max(enumerate(x), key=lambda y: y[1]),
#             F.softmax(probs_list, dim=-1),
#         )
#     )

#     # インデックスを物体名に変換, 物体の位置に物体名と確信度を書き込み表示
#     for (idx, prob), pos in zip(ip_list, pos_list):
#         x, y, w, h = pos
#         x -= PADDING
#         y -= PADDING
#         w += 2 * PADDING
#         h += 2 * PADDING
#         cv2.rectangle(img, (x, y), (x + w, y + h), SHOW_COLOR, 2)
#         cv2.putText(
#             img,
#             "%s:%.1f%%" % (OBJ_NAMES[idx], prob * 100),
#             (x + 5, y + 20),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             SHOW_COLOR,
#             thickness=2,
#         )
#     return ip_list


# def set_background(cap):
#     """
#     カメラのプレビューを表示し、キー操作で背景画像を設定する関数
#     引数:
#     cap: カメラキャプチャオブジェクト
#     戻り値:
#     背景画像
#     """
#     print("=============== Set background ===============\n", end="", flush=True)
#     print("+------------- Key Instructions -------------+")
#     print("|              p : take Picture              |")
#     print("|              q : Quit                      |")
#     print("+--------------------------------------------+")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame.")
#             break
#         cv2.imshow("Preview", frame)
#         cv2.moveWindow("Preview", 0, 0)

#         wkey = cv2.waitKey(5) & 0xFF  # キー入力受付 5ms

#         if wkey == ord("q"):
#             cv2.destroyAllWindows()
#             cap.release()
#             return None
#         elif wkey == ord("p"):
#             cv2.imwrite("background.jpg", frame)
#             print("done!")
#             cv2.destroyAllWindows()
#             return frame


# def realtime_classify():
#     """
#     学習済みモデル読み込み
#     -> テストデータ読み込み
#     -> 分類
#     -> 結果を画像に重ねて表示
#     """
#     # デバイス設定
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # ネットワーク設定
#     net = NeuralNet(NUM_CLASSES).to(device)

#     # 訓練済みデータ取得
#     if os.path.isfile(CKPT_NET):
#         checkpoint = torch.load(CKPT_NET)
#         net.load_state_dict(checkpoint)
#     else:
#         raise FileNotFoundError("No trained network file: {}".format(CKPT_NET))

#     # 評価モード
#     net.eval()

#     # Webカメラを初期化
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     img_back = set_background(cap)
#     if img_back is None:
#         print("Background setting cancelled...")
#         return

#     print("Start! Enter q to quit...")
#     with torch.no_grad():
#         while True:
#             ret, img_target = cap.read()
#             if not ret:
#                 print("Failed to grab frame.")
#                 break

#             # 物体とその位置を検出
#             obj_imgs, positions = detect_ray(img_back, img_target)
#             if obj_imgs:
#                 # 検出物体をネットワークの入力形式に変換
#                 obj_batch = batch_maker(obj_imgs, data_transforms)
#                 # 分類
#                 outputs = net(obj_batch)
#                 # 判定
#                 result = judge_what(img_target, outputs, positions)
#                 # print("  Result:", result)

#             # 表示
#             cv2.imshow("detection", img_target)

#             if cv2.waitKey(2000) == ord("q"):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     try:
#         realtime_classify()
#     except KeyboardInterrupt:
#         print("Exit...")
#     finally:
#         cv2.destroyAllWindows()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 液チ霧箱用のリアルタイム飛跡検出プログラム
# 0.5sごとにフレームを更新し、１つ前のフレームを背景画像に適用
# 背景画像の更新が多いので、飛跡が少なく、ノイズが多い場合に最適
# ペルチェ霧箱を使う場合は、飛跡が多いので上のコードを使う

CKPT_NET = "trained_net.ckpt"  # 学習済みパラメータファイル
OBJ_NAMES = ["alpha", "cosmic"]  # 各クラスの表示名
GRAY_THR = 5  # 濃度変化の閾値
FILTER_SIZE = 5  # medianフィルタのサイズ
RHO_HOUGH = 5  # Hough変換の距離解像度
THETA_HOUGH = 5 * np.pi / 180  # Hough変換の角度解像度
COUNT_HOUGH = 50  # Hough変換の閾値
MIN_LEN_HOUGH = 10  # 検出する直線の最小長さ
MAX_GAP_HOUGH = 50  # 直線として認識する最大の間隔
PADDING = 50  # 枠の大きさに余裕を持たせる
RAY_COUNT_MAX = 2  # バッチサイズ(一度に検出する物体の数)の上限
SHOW_COLOR = (255, 191, 0)  # 枠の色(B,G,R) green
NUM_CLASSES = 2  # クラス数
CHANNELS = 1  # 色のチャンネル数(BGR:3, グレースケール:1)

from train_net import NeuralNet

# 画像データ変換定義
# Resizeと, classifierの最初のLinear入力が関連
data_transforms = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


def detect_ray(back, target):
    """
    OpenCVの背景差分処理で, 検出された物体のタプルを作成
    引数:
    back: 入力背景画像(カラー)
    target: 背景差分対象の画像(カラー) <-- 複数の物体を切り抜き, カラー画像タプルにまとめる
    """
    # 2値化
    b_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    t_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # 差分を計算
    diff = cv2.absdiff(t_gray, b_gray)
    # medianフィルターの適用
    filter = cv2.medianBlur(diff, FILTER_SIZE)
    # 閾値に従ってマスクを作成, 直線を抽出
    mask = cv2.threshold(filter, GRAY_THR, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("mask", mask)

    lines = cv2.HoughLinesP(
        mask,
        RHO_HOUGH,
        THETA_HOUGH,
        COUNT_HOUGH,
        MIN_LEN_HOUGH,
        MAX_GAP_HOUGH,
    )

    # 検出された直線から2本選んで位置情報を抽出
    pt_list = []
    if lines is not None:
        valid_rects = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                valid_rects.append([x1, y1, x2, y2])

        rects, _ = cv2.groupRectangles(valid_rects, groupThreshold=1, eps=0.2)

        for x1, y1, x2, y2 in rects[:5]:
            x, y, w, h = cv2.boundingRect(np.array([[x1, y1], [x2, y2]]))
            pt_list.append((x, y, w, h))

    pt_list = pt_list[:RAY_COUNT_MAX]

    obj_images = tuple(
        map(
            lambda x: Image.fromarray(target[x[1] : x[1] + x[3], x[0] : x[0] + x[2]]),
            pt_list,
        )
    )
    return (obj_images, pt_list)


def batch_maker(tuple_images, transform):
    """
    PIL形式の画像のタプルをtransformし, ネットワークで処理可能なtensorバッチに変換
    引数:
    tuple_images: PIL画像タプル
    transform: torchvision画像変換定義
    """
    return torch.cat([transform(img) for img in tuple_images]).view(
        -1, CHANNELS, 112, 112
    )


def judge_what(img, probs_list, pos_list):
    """
    各クラスに属する確率から物体を決定し, その位置に枠と名前を表示, クラスのインデックスを返す
    引数:
    probs_list: 確率の二次配列. バッチ形式
    pos_list: 位置の二次配列. バッチ形式
    """
    # 最も高い確率とそのインデックスのリストに変換
    ip_list = list(
        map(
            lambda x: max(enumerate(x), key=lambda y: y[1]),
            F.softmax(probs_list, dim=-1),
        )
    )

    # インデックスを物体名に変換, 物体の位置に物体名と確信度を書き込み表示
    results = []
    for (idx, prob), pos in zip(ip_list, pos_list):
        x, y, w, h = pos
        x -= PADDING
        y -= PADDING
        w += 2 * PADDING
        h += 2 * PADDING
        cv2.rectangle(img, (x, y), (x + w, y + h), SHOW_COLOR, 2)
        cv2.putText(
            img,
            "%s:%.1f%%" % (OBJ_NAMES[idx], prob * 100),
            (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            SHOW_COLOR,
            thickness=2,
        )
    results.append(f"{OBJ_NAMES[idx]}: {prob * 100:.1f}% at ({x}, {y})")
    return results


def realtime_classify():
    """
    学習済みモデル読み込み
    -> テストデータ読み込み
    -> 分類
    -> 結果を画像に重ねて表示
    """
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ネットワーク設定
    net = NeuralNet(NUM_CLASSES).to(device)

    # 訓練済みデータ取得
    if os.path.isfile(CKPT_NET):
        checkpoint = torch.load(CKPT_NET)
        net.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError("No trained network file: {}".format(CKPT_NET))

    # 評価モード
    net.eval()

    # Webカメラを初期化
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Start! Enter q to quit...")

    last_background_frame = None

    with torch.no_grad():
        while True:
            ret, img_target = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            if last_background_frame is not None:
                img_back = last_background_frame  # 前のフレームを背景画像に適用
            else:
                img_back = img_target  # 初回は背景画像を初期化

            last_background_frame = img_target.copy()

            # 物体とその位置を検出
            obj_imgs, positions = detect_ray(img_back, img_target)
            if obj_imgs:
                # 検出物体をネットワークの入力形式に変換
                obj_batch = batch_maker(obj_imgs, data_transforms)
                # 分類
                outputs = net(obj_batch)
                # 判定
                result = judge_what(img_target, outputs, positions)
                for res in result:
                    print(f" Result: ", result)

            # 表示
            cv2.imshow("detection", img_target)

            if cv2.waitKey(500) == ord("q"):  # 0.5sでフレーム更新
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        realtime_classify()
    except KeyboardInterrupt:
        print("Exit...")
    finally:
        cv2.destroyAllWindows()
