import os
import cv2
import shutil

from param import SRC_DIR, DEST_ALPHA, DEST_COSMIC  # param.pyを確認!!


def label_img(src_dir, dest_alpha, dest_cosmic):
    """
    tmp内の画像をalphaとcosmicに分類
    引数：
    src_dir: 元画像の場所
    dest_alpha: alphaの保存先
    dest_cosmic: cosmicの保存先
    """

    images = [img for img in os.listdir(src_dir) if img.lower().endswith("jpg")]

    print("************* Key Instructions *************\n")
    print("                Alpha    ->   a")
    print("              Cosmic ray ->   c")
    print("                 Skip    ->   s")
    print("                Delete   ->   d\n")
    print("********************************************")

    for image in images:
        while True:
            image_path = os.path.join(src_dir, image)
            img = cv2.imread(image_path)
            cv2.imshow("Image", img)

            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key == ord("a"):
                shutil.move(image_path, os.path.join(dest_alpha, image))
                print(f"Moved {image} to {dest_alpha}")
                break
            elif key == ord("c"):
                shutil.move(image_path, os.path.join(dest_cosmic, image))
                print(f"Moved {image} to {dest_cosmic}")
                break
            elif key == ord("s"):
                print(f"Skipped {image}")
                break
            elif key == ord("d"):
                os.remove(image_path)
                print(f"Deleted {image}")
                break
            else:
                print(f"Key {key} is not recognized. Try again...")


if __name__ == "__main__":
    label_img(SRC_DIR, DEST_ALPHA, DEST_COSMIC)
