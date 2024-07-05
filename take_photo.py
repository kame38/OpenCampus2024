# ペルチェ(peltie)/液体窒素(n2)霧箱用のリアルタイム飛跡検出プログラム
# ペルチェ/液チ 霧箱を使う場合は、277/278 行目以降を使う

import os
import numpy as np
import cv2
from datetime import datetime

from param import (
    GRAY_THR,
    FILTER_SIZE,
    CUT_MODE,
    RHO_HOUGH,
    THETA_HOUGH,
    COUNT_HOUGH,
    MIN_LEN_HOUGH,
    MAX_GAP_HOUGH,
    PADDING,
    RAY_COUNT_MAX,
)  # param.pyを確認!!


def imshow_rect(img, lines, padding=PADDING):
    """
    取得画像中の直線から2本選んで四角枠で囲む
    枠が重なっている場合は枠をまとめる
    引数:
    img: カメラ画像
    lines: 検出された直線
    padding: 枠の大きさに余裕を持たせる
    """
    if lines is not None:
        valid_rects = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                valid_rects.append([x1, y1, x2, y2])

        # 重なっている枠をまとめる(20%の重なりを許容)
        rects, _ = cv2.groupRectangles(valid_rects, groupThreshold=1, eps=0.2)

        for x1, y1, x2, y2 in rects[:RAY_COUNT_MAX]:  # 2本までの直線を表示
            cv2.rectangle(
                img,
                (x1 - padding, y1 - padding),
                (x2 + padding, y2 + padding),
                (0, 255, 0),
                2,
            )
    cv2.imshow("Preview", img)


def save_cutimg(img, lines, padding=PADDING):
    """
    取得画像中の直線検出箇所を切り抜き保存
    枠が重なっている場合は枠をまとめる
    引数:
    同上
    """
    dt = datetime.now()
    f_name = os.path.join(
        "image_data/tmp", "{}.jpg".format(dt.strftime("%y%m%d%H%M%S"))
    )
    imgs_cut = []
    if lines is not None:
        valid_rects = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                valid_rects.append((x1, y1, x2, y2))

        # 重なっている枠をまとめる(20%の重なりを許容)
        rects, _ = cv2.groupRectangles(valid_rects, groupThreshold=1, eps=0.2)

        for x1, y1, x2, y2 in rects[:RAY_COUNT_MAX]:  # 2本までの直線を切り取る
            x, y, w, h = cv2.boundingRect(
                np.array([[x1 - padding, y1 - padding], [x2 + padding, y2 + padding]])
            )
            cut_img = img[y : y + h, x : x + w]
            if cut_img.size > 0:  # 画像が空でない場合に保存
                imgs_cut.append(cut_img)

    if not imgs_cut:
        return -1
    if len(imgs_cut) > 1:
        for i in range(len(imgs_cut)):
            cv2.imwrite(f_name[:-4] + "_" + str(i + 1) + f_name[-4:], imgs_cut[i])
    else:
        cv2.imwrite(f_name, imgs_cut[0])
    return len(imgs_cut)


def save_img(img):
    """
    取得画像をそのまま保存
    引数:
    img: カメラ画像
    """
    dt = datetime.now()
    fname = os.path.join("image_data/tmp", "{}.jpg".format(dt.strftime("%y%m%d%H%M%S")))
    cv2.imwrite(fname, img)


def take_photo_peltie():
    """
    ペルチェ用の画像を撮影する
    背景撮影->物体撮影, 保存
    """
    cnt = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("=============== Set background ===============\n", end="", flush=True)
    print("+------------- Key Instructions -------------+")
    print("|              p : take Picture              |")
    print("|              q : Quit                      |")
    print("+--------------------------------------------+")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("Preview", frame)
        cv2.moveWindow("Preview", 0, 0)

        wkey = cv2.waitKey(5) & 0xFF  # キー入力受付 5ms

        if wkey == ord("q"):
            cv2.destroyAllWindows()
            cap.release()
            return
        elif wkey == ord("p"):
            save_img(frame)
            back_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("done!")
            break

    # # 既存の画像ファイルを背景画像として使う場合
    # background_image_path = "image_data/tmp/240628200357.jpg"
    # back_frame = cv2.imread(background_image_path)
    # if back_frame is None:
    #     print(
    #         f"Error: Could not read the background image from {background_image_path}"
    #     )
    #     return

    # back_gray = cv2.cvtColor(back_frame, cv2.COLOR_BGR2GRAY)

    print("============= Take photos! ===================")
    print("+-------------  Key Instructions --------------+")
    print("|              p : take Picture                |")
    print("|              i : Initialize (set background) |")
    print("|              q : Quit                        |")
    print("+----------------------------------------------+")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        stream_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(stream_gray, back_gray)
        filter = cv2.medianBlur(diff, FILTER_SIZE)  # medianフィルターの適用(ノイズ除去)
        mask = cv2.threshold(filter, GRAY_THR, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("mask", mask)
        cv2.moveWindow("mask", 700, 0)

        lines = cv2.HoughLinesP(
            mask,
            RHO_HOUGH,
            THETA_HOUGH,
            threshold=COUNT_HOUGH,
            minLineLength=MIN_LEN_HOUGH,
            maxLineGap=MAX_GAP_HOUGH,
        )

        imshow_rect(frame.copy(), lines, PADDING)  # 5本までの直線を表示

        wkey = cv2.waitKey(2000) & 0xFF  # 2秒待つ
        if wkey == ord("q"):
            cv2.destroyAllWindows()
            cap.release()
            return
        elif wkey == ord("i"):
            cv2.destroyAllWindows()
            cap.release()
            break
        elif wkey == ord("p"):
            if CUT_MODE:
                num = save_cutimg(frame, lines)
                if num > 0:
                    cnt += num
                    print("{} new img added... ({} img in total now)".format(num, cnt))
            else:
                save_img(frame)
                cnt += 1
                print("1 new img added... ({} img in total now)".format(cnt))

    print("Initialized")
    take_photo_peltie()

def take_photo_n2():
    """
    液チ用の画像を撮影する
    背景撮影->物体撮影, 保存
    """
    cnt = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("============= Take photos! ===================")
    print("+-------------  Key Instructions --------------+")
    print("|              p : take Picture                |")
    print("|              i : Initialize (set background) |")
    print("|              q : Quit                        |")
    print("+----------------------------------------------+")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        back_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stream_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(stream_gray, back_gray)
        filter = cv2.medianBlur(diff, FILTER_SIZE)  # medianフィルターの適用(ノイズ除去)
        mask = cv2.threshold(filter, GRAY_THR, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("mask", mask)
        cv2.moveWindow("mask", 700, 0)

        lines = cv2.HoughLinesP(
            mask,
            RHO_HOUGH,
            THETA_HOUGH,
            threshold=COUNT_HOUGH,
            minLineLength=MIN_LEN_HOUGH,
            maxLineGap=MAX_GAP_HOUGH,
        )

        imshow_rect(frame.copy(), lines, PADDING)  # 5本までの直線を表示

        wkey = cv2.waitKey(500) & 0xFF  # 0.5秒待つ
        if wkey == ord("q"):
            cv2.destroyAllWindows()
            cap.release()
            return
        elif wkey == ord("i"):
            cv2.destroyAllWindows()
            cap.release()
            break
        elif wkey == ord("p"):
            if CUT_MODE:
                num = save_cutimg(frame, lines)
                if num > 0:
                    cnt += num
                    print("{} new img added... ({} img in total now)".format(num, cnt))
            else:
                save_img(frame)
                cnt += 1
                print("1 new img added... ({} img in total now)".format(cnt))

    print("Initialized")
    take_photo_n2()

if __name__ == "__main__":
    # take_photo_peltie() # ペルチェ霧箱用
    take_photo_n2() # 液チ霧箱用
