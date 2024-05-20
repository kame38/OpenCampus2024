# coding: utf-8
import os
import numpy as np
import cv2
from datetime import datetime
import picamera
import picamera.array

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
    # 日時を取得しファイル名に使用
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

    # 物体を切り抜いて保存
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
    # picamera起動
    with picamera.PiCamera() as camera:
        camera.resolution = (480, 480)  # 解像度
        camera.rotation = 0  # カメラの回転角(度)
        # ストリーミング開始
        with picamera.array.PiRGBArray(camera) as stream:
            print('Set background ... ', end='', flush=True)
            # 初めに背景を撮影
            while True:
                # ストリーミング画像を取得、表示
                camera.capture(stream, 'bgr', use_video_port=True)
                cv2.imshow('Preview', stream.array)

                print("************* Key Instructions *************\n")
                print("              p : take Picture")
                print("              r : Rotate camera")
                print("              q : Quit\n")
                print("********************************************")

                wkey = cv2.waitKey(5) & 0xFF  # キー入力受付

                stream.seek(0)  # 新しくcaptureするための呪文x2
                stream.truncate()

                if wkey == ord('q'):
                    cv2.destroyAllWindows()
                    return print()
                elif wkey == ord('r'):
                    camera.rotation += 90
                elif wkey == ord('p'):
                    camera.exposure_mode = 'off'  # ホワイトバランス固定
                    save_img(stream.array)
                    # グレースケール化して背景画像に設定
                    back_gray = cv2.cvtColor(stream.array, 
                                             cv2.COLOR_BGR2GRAY)
                    print('done')
                    break

            # 背景を設定し終えたら, カメラを動かさないように対象物撮影
            print('Take photos!')
            while True:
                camera.capture(stream, 'bgr', use_video_port=True)
                # 現在のフレームをグレースケール化
                stream_gray = cv2.cvtColor(stream.array, 
                                           cv2.COLOR_BGR2GRAY)

                # 差分の絶対値を計算し二値化, マスク作成
                diff = cv2.absdiff(stream_gray, back_gray)
                mask = cv2.threshold(diff, GRAY_THR, 255, 
                                     cv2.THRESH_BINARY)[1]
                cv2.imshow('mask', mask)

                # Hough変換を使った直線検出
                lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=50, minLineLength=MIN_LEN, maxLineGap=MAX_GAP)
                
                # 検出された直線を四角で囲み表示
                stream_arr = stream.array.copy()
                imshow_rect(stream_arr, lines, MIN_LEN)

                print("************* Key Instructions *************\n")
                print("              p : take Picture")
                print("              i : Initialize (set background")
                print("              q : Quit\n")
                print("********************************************")

                wkey = cv2.waitKey(5) & 0xFF

                stream.seek(0)
                stream.truncate()

                if wkey == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif wkey == ord('i'):
                    break
                elif wkey == ord('p'):
                    if CUT_MODE:
                        num = save_cutimg(stream.array, lines, MIN_LEN)
                        if num > 0:
                            cnt += num
                            print('{} new img added... ({} img in total now)'.format(num, cnt))
                    else:
                        save_img(stream.array)
                        cnt += 1
                        print('1 new img added... ({} img in total now)'.format(cnt))

    print('Initialized')
    take_photo()


if __name__ == '__main__':
    take_photo()
