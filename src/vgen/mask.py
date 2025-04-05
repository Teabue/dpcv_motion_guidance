import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor

def get_bounding_box(mask):
    """Get Bbox of non-zero region"""
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) == 0:  # Empty mask
        return None
    return (min(y_coords), max(y_coords), min(x_coords), max(x_coords))

def letterbox_resize(mask, target_shape):
    """
    Resizes a binary mask while preserving aspect ratio and pads to fit the target shape.
    This makes the mask translation and scale invariant.
    
    Args:
        mask (np.ndarray): The binary mask to be resized.
        target_shape (tuple): Desired (height, width).

    Returns:
        np.ndarray: Resized and padded mask.
    """
    h, w = mask.shape
    target_h, target_w = target_shape

    # Compute the scaling factor (maintain aspect ratio)
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize while keeping original proportions
    resized_mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Compute padding
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    # Apply padding
    padded_mask = np.pad(resized_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                         mode='constant', constant_values=0)
    
    return padded_mask

def symmetric_padding(mask, target_shape):
    """ Symmetric padding without resizing to make translation invariant iou computation"""
    h, w = mask.shape
    target_h, target_w = target_shape

    # Compute padding
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    # Apply symmetric padding
    padded_mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                         mode='constant', constant_values=0)
    
    return padded_mask



def compute_iou(mask1, mask2, mode='translate', plot=False):
    """Compute iou between two binary masks"""
    bbox1 = get_bounding_box(mask1)
    bbox2 = get_bounding_box(mask2)

    if bbox1 is None or bbox2 is None:
        return 0.0  # No overlap if one mask is empty

    # Crop masks to bbox
    mask1_cropped = mask1[bbox1[0]:bbox1[1]+1, bbox1[2]:bbox1[3]+1]
    mask2_cropped = mask2[bbox2[0]:bbox2[1]+1, bbox2[2]:bbox2[3]+1]

    # Find maximum dimensions
    target_shape = (max(mask1_cropped.shape[0], mask2_cropped.shape[0]),
                    max(mask1_cropped.shape[1], mask2_cropped.shape[1]))
    
    if mode == 'translate':
        mask1_cropped = symmetric_padding(mask1_cropped, target_shape)
        mask2_cropped = symmetric_padding(mask2_cropped, target_shape)
    elif mode == 'translate-scale':
        mask1_cropped = letterbox_resize(mask1_cropped, target_shape)
        mask2_cropped = letterbox_resize(mask2_cropped, target_shape)
    else:
        raise NotImplementedError(f"Mode '{mode}' is not implemented.")
    
    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(mask1_cropped)
        plt.title('Mask 1')
        plt.subplot(1, 2, 2)
        plt.imshow(mask2_cropped)
        plt.title('Mask 2')
        plt.show()
    # TODO: Insert reshaping orignal mask to match target flow shape for other modes like rotate and scale

    # Compute IoU
    intersection = np.logical_and(mask1_cropped, mask2_cropped).sum()
    union = np.logical_or(mask1_cropped, mask2_cropped).sum()
    
    return intersection / union if union > 0 else 0.0
    

def automatic_mask(img: np.ndarray,
                   prev_mask: np.ndarray,
                   target_point: np.ndarray,
                   iou_threshold=0.5, 
                   chkpt_path='./assets/sam_vit_b_01ec64.pth',
                   image_format='RGB',
                   mode='translate'):

    sam = sam_model_registry['vit_b'](checkpoint=chkpt_path)
    sam.to(device='cpu')
    predictor = SamPredictor(sam)
    predictor.set_image(img, image_format=image_format)
    masks, _, _ = predictor.predict(
        point_coords=target_point[None],
        point_labels=np.array([1]),
        multimask_output=True,
    )

    best_iou = 0.0
    best_mask = None

    for mask in masks:
        iou = compute_iou(prev_mask, mask, mode, plot=True)
        if iou > best_iou:
            best_iou = iou
            best_mask = mask

    # TODO: Insert if this fails, try generating every mask 
    return best_mask if best_iou >= iou_threshold else None


if __name__ == '__main__':
    # Example demo that assumes the motion guidance went perfect, and the teapot moved to the target point
    # whoever gitignored pngs it's your fault that I'm too lazy to add
    # the images I used (I just used original teapot and moved teapot results)
    
    import cv2
    import matplotlib.pyplot as plt
    
    prompt_point = np.array([[325,240]])
    img1 = cv2.imread('./assets/orig.png')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB, img1) 

    sam = sam_model_registry['vit_b'](checkpoint='./assets/sam_vit_b_01ec64.pth')
    sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    predictor = SamPredictor(sam)
    predictor.set_image(img1, image_format='RGB')
    masks, _, _ = predictor.predict(
        point_coords=prompt_point,
        point_labels=np.array([1]),
        multimask_output=False,
    )
    mask = masks[0]
    
    mask_center_mass = np.array([np.mean(np.nonzero(mask)[1]), np.mean(np.nonzero(mask)[0])])
    
    masked_img = mask[..., None]*img1
    masked_img = cv2.circle(masked_img, (int(mask_center_mass[0]), int(mask_center_mass[1])), 10, (0,255,0), -1)
    
    plt.imshow(masked_img)
    plt.title('Masked original image and center mass displayed')
    plt.show()
    
    # Target point determined from matching the where the center mass was of the orig img
    # This is a little cheaty, since I have assumed with this that the teapot moved 
    # exactly to the target point.
    target_point = np.array([303, 370])
    
    img2 = cv2.imread('./assets/moved.png')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB, img2)
    mask2 = automatic_mask(img2, mask, target_point, iou_threshold=0.4)
    
    masked_img2 = mask2[..., None]*img2
    masked_img2 = cv2.circle(masked_img2, (int(target_point[0]), int(target_point[1])), 10, (0,255,0), -1)
    plt.imshow(masked_img2)
    plt.title('Masked moved image and target point displayed')
    plt.show()
