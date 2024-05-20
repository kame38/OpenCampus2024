import os
import cv2
import shutil

SRC_DIR = 'image_data/tmp'
DEST_ALPHA = 'image_data/all/alpha'
DEST_COSMIC = 'image_data/all/cosmic'


def label_img(src_dir, dest_alpha, dest_cosmic):
    """
    tmp内の画像をalphaとcosmicに分類
    引数：
    src_dir: 元画像の場所
    dest_alpha: alphaの保存先
    dest_cosmic: cosmicの保存先
    """

    images = [img for img in os.listdir(src_dir) if img.lower().endswith('jpg')]

    print("************* Key Instructions *************\n")
    print("                Alpha    ->   1")
    print("              Cosmic ray ->   2")
    print("                 Skip    ->   0")
    print("                Delete   ->   d\n")
    print("********************************************")
    
    for image in images:
        while True:
            image_path = os.path.join(src_dir, image)
            img = cv2.imread(image_path)
            cv2.imshow('Image', img)

            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key == ord('1'):
                shutil.move(image_path, os.path.join(dest_alpha, image))
                print(f"Moved {image} to {dest_alpha}")
                break
            elif key == ord('2'):
                shutil.move(image_path, os.path.join(dest_cosmic, image))
                print(f"Moved {image} to {dest_cosmic}")
                break
            elif key == ord('0'):
                print(f"Skipped {image}")
                break
            elif key == ord('d'):
                os.remove(image_path)
                print(f"Deleted {image}")
                break
            else:
                print(f"Key {key} is not recognized. Try again.")              

if __name__ == "__main__":
    label_img(SRC_DIR, DEST_ALPHA, DEST_COSMIC)
