from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    frame_points = np.load('frame_points-20250429T161323Z.npy')
    
    for frame in frame_points:
        # Get the flow for the first target point
        flow = get_masked_flow(mask, frame, mode='translate')
        
        # Get the edit mask
        edit_mask = get_edit_mask(flow, output_shape=(4, 64, 64))
        
        # Display edit mask
        plt.imshow(edit_mask.transpose(1, 2, 0)[...,0], cmap='gray')
        plt.title("Edit Mask")
        plt.axis('off')
        plt.show()