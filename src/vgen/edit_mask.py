from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import EuclideanTransform, warp, rotate

def get_edit_mask_v2(mask: np.ndarray, 
                     target_point: np.ndarray, 
                     output_shape: list,
                     mode: str, 
                     save_path: str = None,
                     dilate: bool = True) -> np.ndarray:
    """Generates edit mask by the use of the skimage transform library
    Args:
        mask (np.ndarray): 2D binary array mask for object of interest
        target_point (np.ndarray): Target point for flow in (x, y) meaning (col, row) format.
        mode (str): What type of flow to generate. Defaults to 'translate'.
        save_path (str, optional): Path to save the edit mask. Defaults to None.
    Returns:
        np.ndarray: Masked flow numpy (H, W, 2)
    """
    assert len(mask.shape) == 2, "Mask should be a 2D array"
    assert mode in ['translate', 'rotate'], "Mode should be either 'translate' or 'rotate'"
    
    mask = mask.astype(np.uint8)
    
    if dilate:
        # Dilate the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=5)
    
    # Get the center mass of the mask
    y, x = np.nonzero(mask)
    center_pos = np.array([np.mean(x), np.mean(y)])
    
    # Get the value of interest depending on the mode
    if mode == 'translate':
        translation = target_point - center_pos
        tform = EuclideanTransform(translation=translation)
        
        target_mask = warp(mask, tform.inverse, output_shape=mask.shape, order=1, mode='edge', cval=0)
        
    elif mode == 'rotate':
        # Center the points according to the mass center
        target_pos_c = target_point - center_pos
        target_pos_c[1] = - target_pos_c[1]  # Invert y-coordinate for rotation
        
        # Get the angle of rotation
        rotation_angle = np.arctan2(target_pos_c[1], target_pos_c[0])
        rotation_angle = np.degrees(rotation_angle)
        
        target_mask = rotate(mask, rotation_angle, center=center_pos)
        
    else:
        raise ValueError("Invalid mode. Choose either 'translate' or 'rotate'")
    
    edit_mask = mask.astype(bool) | target_mask.astype(bool)
    
    # Optionally save the edit mask
    if save_path is not None:
        np.save(save_path, edit_mask)
    
    # Resize to the output shape
    edit_mask = cv2.resize(edit_mask.astype(np.float32), (output_shape[2], output_shape[1]), interpolation=cv2.INTER_NEAREST)
    edit_mask_bool = edit_mask.astype(bool)
    edit_mask_bool = np.repeat(edit_mask_bool[None], output_shape[0], axis=0)
    
    return ~edit_mask_bool #(C, H, W)
    

def get_edit_mask(flow: np.ndarray, output_shape: list, save_path: Optional[str] = None) -> np.ndarray:
    """Automatically generates a mask that specifies all locations 
    that any pixel is moving to or from

    Args:
        flow (np.ndarray): The optical flow for the image
        2D array of shape (H, W, 2)
        output_shape (list): The target shape of the output mask
            (C, H, W)
        save_path (Optional[str] = None): Path to save the edit mask. 
            Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    edit_mask = np.zeros(flow.shape[:2], dtype=np.uint8)
    
    # Get the non-zero flow vectors
    rows_nonzero, cols_nonzero, _ = np.nonzero(flow)
    edit_mask[rows_nonzero, cols_nonzero] = 1
    
    # Get the target positions for the non-zero flow vectors
    target_nonzero = np.array([rows_nonzero, cols_nonzero]).T + np.flip(flow[rows_nonzero, cols_nonzero], axis = 1)
    target_nonzero = target_nonzero.astype(int)
    
    # Remove out-of-bounds target positions
    target_nonzero = target_nonzero[
        (target_nonzero[:, 0] >= 0) & (target_nonzero[:, 0] < flow.shape[0]) &
        (target_nonzero[:, 1] >= 0) & (target_nonzero[:, 1] < flow.shape[1])
    ]
    
    edit_mask[target_nonzero[:, 0], target_nonzero[:, 1]] = 1
    
    # Fill holes in the mask
    edit_mask = cv2.morphologyEx(edit_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    # Optionally save the edit mask
    if save_path is not None:
        np.save(save_path, edit_mask)
    
    # Resize to the output shape - NOTE: The output_shape might be switched
    edit_mask_bool = cv2.resize(edit_mask, (output_shape[2], output_shape[1]), interpolation=cv2.INTER_NEAREST)
    edit_mask_bool = edit_mask_bool.astype(bool)
    edit_mask_bool = np.repeat(edit_mask_bool[None], output_shape[0], axis=0)
    
    return ~edit_mask_bool #(C, H, W)
    

if __name__ == "__main__":
    from vgen.flow import get_masked_flow

    # Load mask and frame points
    mask = np.load('mask_c.npy')
    frame_points = np.load('frame_points-20250430T095641Z.npy')
    mode = 'rotate'
    
    for frame in frame_points:
        # Get the flow for the first target point
        flow = get_masked_flow(mask, frame, mode=mode)
        
        # Get the edit mask
        # edit_mask = get_edit_mask(flow, output_shape=(4, 64, 64))
        edit_mask = get_edit_mask_v2(mask, frame, output_shape=(4, 64, 64), mode=mode)
        
        # Display edit mask
        plt.imshow(edit_mask.transpose(1, 2, 0)[...,0], cmap='gray')
        plt.title("Edit Mask")
        plt.axis('off')
        plt.show()