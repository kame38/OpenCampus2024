# coding: utf-8
import os
from PIL import Image
from time import sleep
import cv2
import torch

# pytorchディレクトリで "export OMP_NUM_THREADS=4"(デフォルト)
# 並列処理コア数は "print(torch.__config__.parallel_info())" で確認
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torchvision import transforms
import numpy as np

from param import (
    CKPT_NET,
    OBJ_NAMES,
    RHO_HOUGH,
    THETA_HOUGH,
    COUNT_HOUGH,
    MIN_LEN_HOUGH,
    MAX_GAP_HOUGH,
    GRAY_THR,
    RAY_COUNT_MAX,
    SHOW_COLOR,
    NUM_CLASSES,
    PIXEL_LEN,
    CHANNELS,
    PADDING,
)  # param.pyを確認!!

from train_net import NeuralNet

# 画像データ変換定義
# Resizeと, classifierの最初のLinear入力が関連
data_transforms = transforms.Compose(
    [
        transforms.Resize((PIXEL_LEN, PIXEL_LEN)),
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
    print("Detecting Rays ...")
    # 2値化
    b_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    t_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # 差分を計算
    diff = cv2.absdiff(t_gray, b_gray)
    # medianフィルターの適用
    filter = cv2.medianBlur(diff, 5)
    # 閾値に従ってマスクを作成, 直線を抽出
    mask = cv2.threshold(filter, GRAY_THR, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("mask", mask)

    # Hough変換を使った直線検出
    lines = cv2.HoughLinesP(
        mask,
        RHO_HOUGH,
        THETA_HOUGH,
        COUNT_HOUGH,
        MIN_LEN_HOUGH,
        MAX_GAP_HOUGH,
    )

    # 検出された直線から位置情報を抽出
    pt_list = []
    if lines is not None:
        valid_rects = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                valid_rects.append((x1, y1, x2, y2))

        filtered_rects = []
        for rect1 in valid_rects:
            x1, y1, x2, y2 = rect1
            rect1_area = (x2 - x1) * (y2 - y1)
            is_contained = False
            for rect2 in valid_rects:
                if rect1 == rect2:
                    continue
                x3, y3, x4, y4 = rect2
                rect2_area = (x4 - x3) * (y4 - y3)
                if (
                    x1 >= x3
                    and y1 >= y3
                    and x2 <= x4
                    and y2 <= y4
                    and rect1_area <= rect2_area
                ):
                    is_contained = True
                    break
            if not is_contained:
                filtered_rects.append((x1, y1, x2, y2))

        pt_list = [
            (x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in filtered_rects
        ]

    pt_list = pt_list[:RAY_COUNT_MAX]

    # 位置情報に従ってフレーム切り抜き, PIL画像のタプルに変換して返す
    obj_imgaes = tuple(
        map(
            lambda x: Image.fromarray(target[x[1] : x[1] + x[3], x[0] : x[0] + x[2]]),
            pt_list,
        )
    )
    return (obj_imgaes, pt_list)


def batch_maker(tuple_images, transform):
    """
    PIL形式の画像のタプルをtransformし, ネットワークで処理可能なtensorバッチに変換
    引数:
    tuple_images: PIL画像タプル
    transform: torchvision画像変換定義
    """
    return torch.cat([transform(img) for img in tuple_images]).view(
        -1, CHANNELS, PIXEL_LEN, PIXEL_LEN
    )


def judge_what(img, probs_list, pos_list):
    """
    各クラスに属する確率から物体を決定し, その位置に枠と名前を表示, クラスのインデックスを返す
    引数:
    probs_list: 確率の二次配列. バッチ形式
    pos_list: 位置の二次配列. バッチ形式
    """
    print("Judging objects ...")
    # 最も高い確率とそのインデックスのリストに変換
    ip_list = list(
        map(
            lambda x: max(enumerate(x), key=lambda y: y[1]),
            F.softmax(probs_list, dim=-1),
        )
    )

    # インデックスを物体名に変換, 物体の位置に物体名と確信度を書き込み表示
    for (idx, prob), pos in zip(ip_list, pos_list):
        cv2.rectangle(
            img, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), SHOW_COLOR, 2
        )
        cv2.putText(
            img,
            "%s:%.1f%%" % (OBJ_NAMES[idx], prob * 100),
            (pos[0] + 5, pos[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            SHOW_COLOR,
            thickness=2,
        )
    return ip_list


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
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Setting background ...")
    sleep(2)

    # 背景に設定
    ret, img_back = cap.read()
    if not ret:
        print("Failed to grab frame.")
        cap.release()
        cv2.destroyAllWindows()
        return

    print("Start! Enter q to quit...")
    with torch.no_grad():
        while True:
            ret, img_target = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # 物体とその位置を検出
            obj_imgs, positions = detect_ray(img_back, img_target)
            if obj_imgs:
                # 検出物体をネットワークの入力形式に変換
                obj_batch = batch_maker(obj_imgs, data_transforms)
                # 分類
                outputs = net(obj_batch)
                # 判定
                result = judge_what(img_target, outputs, positions)
                print("  Result:", result)

            # 表示
            cv2.imshow("detection", img_target)

            if cv2.waitKey(200) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        realtime_classify()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
