import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_image(image_path, smallest_dim): #divisable_by):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Resize the smallest dimension to the specified size while maintaining aspect ratio
    if h < w:
        new_h = smallest_dim
        new_w = int(w * (smallest_dim / h))
    else:
        new_w = smallest_dim
        new_h = int(h * (smallest_dim / w))
    
    # Resize to the new dimensions
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Resize to be divisible by 32
    # image_cut = image[:(new_h // divisable_by) * divisable_by, :new_w // divisable_by * divisable_by, :]
    
    #return image_cut
    return image

def crop_image(img_path, start_idx, plot: bool = False):
    target_length = 512 + 128
    img1 = cv2.imread(img_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    end_idx = start_idx + target_length

    # Cropt height
    img1 = img1[start_idx:end_idx, :, :]

    if plot:
        plt.imshow(img1)
        plt.title('Image title')
        plt.axis('off')
        plt.show()

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

    img_path_no_ext = img_path.split('.')
    cv2.imwrite(f'{img_path_no_ext[0]}_cut.png', img1)
    
    return

if __name__ == "__main__":
    smallest_dim = 512
    # divisable_by = 64
    
    # Resize and save the images
    img_path = 'assets/kahlua.jpg'
    resize_path = 'assets/kahlua_512.png'
    img1 = resize_image(img_path, smallest_dim) # divisable_by)
    cv2.imwrite(resize_path, img1)
    
    # Crop and save the images
    start_idx = 20
    crop_image(resize_path, start_idx)
