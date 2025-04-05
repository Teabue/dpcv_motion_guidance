from typing import List, Tuple, Union
import numpy as np


def get_center_mass(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(mask)
    center_pos = np.array([np.mean(x), np.mean(y)])
    return center_pos


def get_translation_flow(cur_pos, target_pos, hw=(512, 512)) -> np.ndarray:
    flow = np.zeros((hw[0], hw[1], 2))
    flow[:,:,:] = target_pos - cur_pos
    return flow


def get_masked_flow(mask: np.ndarray, 
         target_point: Union[List[int], Tuple[int, int], np.ndarray], 
         mode = 'translate') -> np.ndarray:
    """Generates flow by moving the center mass of the mask to the target point.

    Args:
        mask (np.ndarray): 2D binary array mask for object of interest
        target_point (List[int] | Tuple[int, int] | np.ndarray): Target point for flow in (x, y) meaning (col, row) format.
        mode (str, optional): What type of flow to generate. Defaults to 'translate'.

    Returns:
        np.ndarray: Masked flow numpy (H, W, 2) array
    """
    assert len(mask.shape) == 2, "Mask should be a 2D array"

    cur_pos = get_center_mass(mask)
    if not isinstance(target_point, np.ndarray):
        target_point = np.array(target_point)
    
    if mode == 'translate':
        flow = get_translation_flow(cur_pos, target_point, hw=mask.shape)
    else:
        raise NotImplementedError(f"Flow: Mode {mode} not implemented")
    
    return flow * mask[..., None]
    

if __name__ == '__main__':
    # Example usage, click around to see the flow change
    import cv2
    from gui.colorwheel import flow_to_image

    def mouse_callback(event, x, y, flags, param):
        global target_point, flow, flow_im
        if event == cv2.EVENT_LBUTTONDOWN:
            target_point = (x, y)
            flow = get_masked_flow(mask, target_point, mode='translate')
            flow_im = flow_to_image(flow, convert_to_bgr=True)
    
    # Circle mask
    mask = np.zeros((512, 512))
    center = (256, 256)
    mask = cv2.circle(mask, center, 60, 1, thickness=-1)

    # Initial flow
    target_point = (300, 300)
    flow = get_masked_flow(mask, target_point, mode='translate')
    flow_im = flow_to_image(flow, convert_to_bgr=True)

    cv2.namedWindow('Flow')
    cv2.setMouseCallback('Flow', mouse_callback)

    while True:
        cv2.imshow('Flow', flow_im)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cv2.destroyAllWindows()

