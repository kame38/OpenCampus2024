import os
import cv2
import shutil

def label_img(src_dir, dest_alpha, dest_cosmic):

    # Get list of images in the source directory
    images = [img for img in os.listdir(src_dir) if img.lower().endswith('jpg')]

    print("************* Key Instructions *************\n")
    print("                Alpha    ->   1")
    print("              Cosmic ray ->   2")
    print("                 Skip    ->   0")
    print("                Delete   ->   d\n")
    print("********************************************")
    
    for image in images:
        while True:
            # Full path to the image
            image_path = os.path.join(src_dir, image)

            # Read and display the image using OpenCV
            img = cv2.imread(image_path)
            cv2.imshow('Image', img)

            # Wait for key press
            key = cv2.waitKey(0) & 0xFF

            # Close the image window
            cv2.destroyAllWindows()

            # Move the image based on the key press
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
    # Directories
    src_dir = 'image_data/tmp'
    dest_alpha = 'image_data/class/alpha'
    dest_cosmic = 'image_data/class/cosmic'

    # Classify images
    label_img(src_dir, dest_alpha, dest_cosmic)
