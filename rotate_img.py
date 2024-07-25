import os
from PIL import Image


def augment(image_path, output_dir):
    """
    画像を回転および反転させて保存する
    """
    image = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 90度回転
    rotated_90 = image.rotate(90)
    rotated_90.save(os.path.join(output_dir, base_name + "_rotated_90.jpg"))

    # 180度回転
    rotated_180 = image.rotate(180)
    rotated_180.save(os.path.join(output_dir, base_name + "_rotated_180.jpg"))

    # 270度回転
    rotated_270 = image.rotate(270)
    rotated_270.save(os.path.join(output_dir, base_name + "_rotated_270.jpg"))

    # 左右反転
    flipped_lr = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_lr.save(os.path.join(output_dir, base_name + "_flipped_lr.jpg"))

    # 上下反転
    flipped_ud = image.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_ud.save(os.path.join(output_dir, base_name + "_flipped_ud.jpg"))


def augment_directory(directory):
    """
    指定されたディレクトリ内のすべての画像を回転および反転させる
    """
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # 拡張された画像を再度読み込まないように、オリジナルの画像だけを処理
            if "rotated" not in filename and "flipped" not in filename:
                image_path = os.path.join(directory, filename)
                augment(image_path, directory)
                print(f"Processed {filename}")


if __name__ == "__main__":
    directory = "./image_data/all/cosmic"
    # directory = "./image_data/all/noise"
    # directory = "./image_data/all/alpha"
    augment_directory(directory)
