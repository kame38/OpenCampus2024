import os
import numpy as np
import cv2
from datetime import datetime

from param import MIN_LEN, MAX_GAP, GRAY_THR, CUT_MODE
"""
MIN_LEN = 20        # 検出する直線の最小長さ
MAX_GAP = 200       # 直線として認識する最大の間隔
GRAY_THR = 20       # 濃度変化の閾値
CUT_MODE = True     # True:検出物体を切り取って保存
"""

def imshow_rect(img, lines, minlen=0):
    """
    取得画像中の直線検出箇所全てを四角枠で囲む
    引数:
    img: カメラ画像
    lines: 検出された直線
    minlen: 検出の大きさの閾値（これより直線が短い箇所は除く）
    """
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(x2 - x1) < minlen and abs(y2 - y1) < minlen:
                    continue
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Preview', img)

def save_cutimg(img, lines, minlen=0):
    """
    取得画像中の直線検出箇所を全て切り抜き保存
    引数:
    同上
    """
    dt = datetime.now()
    f_name = os.path.join('image_data/tmp', '{}.jpg'.format(dt.strftime('%y%m%d%H%M%S')))
    imgs_cut = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(x2 - x1) < minlen and abs(y2 - y1) < minlen:
                    continue
                x, y, w, h = cv2.boundingRect(np.array([[x1, y1], [x2, y2]]))
                imgs_cut.append(img[y:y+h, x:x+w])

    if not imgs_cut:
        return -1
    if len(imgs_cut) > 1:
        for i in range(len(imgs_cut)):
            cv2.imwrite(f_name[:-4]+'_'+str(i+1)+f_name[-4:], imgs_cut[i])
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
    fname = os.path.join('image_data/tmp', '{}.jpg'.format(dt.strftime('%y%m%d%H%M%S')))
    cv2.imwrite(fname, img)

def take_photo():
    """
    背景撮影->物体撮影, 保存
    """
    cnt = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print('Set background!\n', end='', flush=True)
    print("************* Key Instructions *************\n")
    print("              p : take Picture")
    print("              q : Quit\n")
    print("********************************************")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow('Preview', frame)

        wkey = cv2.waitKey(5) & 0xFF  # キー入力受付
        print(f"Key pressed: {wkey}")  # デバッグ出力

        if wkey == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            return
        elif wkey == ord('p'):
            save_img(frame)
            back_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print('done')
            break

    print('Take photos!\n')
    print("************* Key Instructions *************\n")
    print("              p : take Picture")
    print("              i : Initialize (set background")
    print("              q : Quit\n")
    print("********************************************")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        stream_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(stream_gray, back_gray)
        mask = cv2.threshold(diff, GRAY_THR, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow('mask', mask)

        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=50, minLineLength=MIN_LEN, maxLineGap=MAX_GAP)
        imshow_rect(frame.copy(), lines, MIN_LEN)

        wkey = cv2.waitKey(5) & 0xFF

        if wkey == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            return
        elif wkey == ord('i'):
            break
        elif wkey == ord('p'):
            if CUT_MODE:
                num = save_cutimg(frame, lines, MIN_LEN)
                if num > 0:
                    cnt += num
                    print('{} new img added... ({} img in total now)'.format(num, cnt))
            else:
                save_img(frame)
                cnt += 1
                print('1 new img added... ({} img in total now)'.format(cnt))

    cap.release()
    print('Initialized')
    take_photo()

if __name__ == '__main__':
    take_photo()
