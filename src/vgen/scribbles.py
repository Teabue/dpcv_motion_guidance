import gradio as gr
import cv2
import numpy as np
import numpy.ma as ma
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import torch.backends
import torch.backends.mps

from datetime import datetime
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import skfmm

import numpy as np

def get_sorted_coordinates(masked_distances: np.ndarray) -> np.ndarray:
    """
    Get a sorted array of coordinates based on the smallest distances
    in a 2D masked array.

    Args:
        masked_array (np.ma.MaskedArray): A 2D masked array of shape (H, W) of distances.

    Returns:
        np.ndarray: An array of shape (H*W, 2) with sorted coordinates.
    """
    # Flatten the masked array and get valid (non-masked) indices
    valid_indices = np.ma.nonzero(masked_distances)
    distances = masked_distances[valid_indices]

    # Combine valid indices into a list of coordinates - NOTE: y, x order (I hate different conventions)
    coordinates = np.array(list(zip(valid_indices[1], valid_indices[0])))

    # Sort the coordinates by their corresponding distances
    sorted_indices = np.argsort(distances)
    sorted_coordinates = coordinates[sorted_indices]

    return sorted_coordinates

def get_distances_from_mask_center(paint_mask: np.ndarray, sam_center: np.ndarray) -> np.ndarray:
    """Computes the geodesic distance from the SAM mask center to each point in the scribble

    Args:
        paint_mask (np.ndarray): The mask from the scribble
        sam_center (np.ndarray): The center of the SAM mask

    Returns:
        np.ndarray: The geodesic distance from the mask center to the rest of the mask
    """
    # Determine where distances should not be computed
    mask_inv = ~paint_mask.astype(bool)
    
    # Determine the origin point for the distance transform
    m = np.ones_like(paint_mask)
    m[sam_center[1], sam_center[0]] = 0
    
    # Compute the distance transform for only the mask
    m_masked = np.ma.masked_array(distance_transform_edt(m), mask_inv)

    # Reconfigure the 0 contour by getting the closest point to the center
    smallest = np.argwhere(m_masked == np.min(m_masked))[0]
    m_masked[smallest[0], smallest[1]] = 0

    # Use fast marching to compute distances
    distance = skfmm.distance(m_masked, dx=1)
    
    return distance

def fit_points_to_skeleton(skeleton: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Fits a set of points to the skeleton (medial axis) of the scribble.
    
    Args:
        skeleton (np.ndarray): The skeleton of the mask as a 2D array. Shape (H, W)
        points (np.ndarray): The points (N, 2) to fit to the skeleton. Shape (N, 2)
        NOTE: The points are assumed to be sorted according to the distance 
        from the center of the mask by the use of the get_sorted_coordinates 
        function and the get_distances_from_mask_center function.
        
    Returns:
        np.ndarray: The fitted points
    """
    # Get the coordinates of the mask
    skele_coords = np.argwhere(skeleton)
    skele_coords = np.flip(skele_coords, axis=1) # Once again, I hate the different conventions

    # Find the coordinates closest to the skeleton
    distances = np.linalg.norm(points[:, None, :] - skele_coords[None, : , :], axis=2) # Shape (N, W)
    closest_coords = np.argmin(distances, axis=1)
    points = skele_coords[closest_coords]
    
    return points

def make_scribble_to_frames(scribble: np.ndarray, sam_center: np.ndarray, frames: int = 3) -> np.ndarray:
    """Preprocesses the scribble mask to extract the geodesic distance from the center.

    Args:
        scribble (np.ndarray): The scribble mask
        sam_center (np.ndarray): The center of the SAM mask

    Returns:
        np.ndarray: The preprocessed scribble mask
    """
    if len(scribble.shape) == 3:
        # Make the mask binary
        scribble = cv2.cvtColor(scribble, cv2.COLOR_RGBA2GRAY)
    
        scribble[scribble > 0] = 1
    
    # If no mask is drawn, return the original image
    if scribble.sum() == 0:
        return None
    
    # Sort the coordinates by distance from the center
    distance = get_distances_from_mask_center(paint_mask=scribble, 
                                              sam_center=sam_center)
    sorted_distances = get_sorted_coordinates(masked_distances=distance)
    
    # Keep the end point fixed and space the coordinates into N points
    spaced_coords = np.empty((frames, 2))
    spaced_coords[-1] = sorted_distances[-1]
    steps = np.linspace(0, len(sorted_distances), frames - 1, dtype=int, endpoint=False)
    spaced_coords[:-1] = sorted_distances[steps]
    
    # Skeletonize the mask
    mask_skeleton = skeletonize(scribble) 
    
    # Fit the points to the skeleton
    points = fit_points_to_skeleton(skeleton=mask_skeleton, 
                                    points=spaced_coords)
    
    return points

# SAM model
SAM_CHECKPOINT_PATH = "./assets/sam_vit_b_01ec64.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "vit_b"

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
predictor = SamPredictor(sam)

with gr.Blocks() as demo:
    # --------------------------------------------------------------------
    # --                     Gradio Components                          --
    # --------------------------------------------------------------------
    
    # Images
    image_input = gr.Image(type="numpy", label="Upload an Image", interactive=True, visible=True)
    image_editing = gr.Paint(label="Drawing image", type="numpy", interactive=False, layers=False)
    image_guidance = gr.Image(label="Estimated motion guidance",interactive=False)

    # States
    state_translation_center = gr.State()
    state_frame_points = gr.State()
    state_number_of_frames = gr.State(value = 3)
    state_motion_option = gr.State(value = "Segment")
    
    # Textboxes
    text_translation_center = gr.Textbox(placeholder="Click on the image to get the center of the motion", interactive=False)
    text_number_of_frames = gr.Textbox(placeholder="Number of frames (click enter to confirm)", interactive=False, visible=False)
    
    # Buttons
    button_make_video = gr.Button("Make Video", visible=True, interactive=False)
    button_motion_option = gr.Radio(["Segment", "Translate", "Rotate"], label="Motion Guidance", info="Choose among the available motion guidance options", value="Segment", visible=False, interactive=False)
    
    # --------------------------------------------------------------------
    # --                     Gradio Events                              --
    # --------------------------------------------------------------------
    # Upload image
    @image_input.upload(
        inputs=image_input,
        outputs=[image_input, 
                 image_editing, 
                 text_number_of_frames, 
                 button_motion_option],
    )
    def on_image_upload(image):
        """Hides the static uploaded image and shows the editable image"""
        predictor.set_image(image)
        return gr.update(visible=False), image, gr.update(visible=True, interactive = True), gr.update(visible=True)
    
    # Click on image
    @image_editing.select(
        inputs=image_input,
        outputs=[image_guidance, 
                 image_editing, 
                 state_translation_center,
                 button_motion_option],
    )
    def on_image_click(image, evt: gr.SelectData):
        """Get clicked coordinates and run SAM model"""

        clicked_points = evt.index
        
        # Run SAM model
        input_point = np.array([clicked_points])
        input_label = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        mask = masks[scores.argmax()]
        plt.imshow(mask)
        plt.axis('off')
        plt.savefig('mask.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        np.save('mask.npy', mask)
        
        # Overlay mask on image
        mask = mask.astype(np.uint8) * 255 
        color_mask = np.zeros_like(image)
        color_mask[:, :, 1] = mask 
        
        # Find center of color mask
        mask_center = np.mean(np.argwhere(mask > 0), axis=0).astype(int)[::-1]
        
        # Overlay the mask and draw a circle at the center
        masked_image = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
        masked_image = cv2.circle(masked_image, tuple(mask_center), 5, (0, 0, 255), -1)
        plt.imshow(masked_image)
        plt.axis('off')
        plt.savefig('masked_image.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        np.save('masked_image.npy', masked_image)
        
        # Update the image, and save the mask center
        return masked_image, masked_image, mask_center, gr.update(interactive=True)

    # Update text box with the center of the translation
    @state_translation_center.change(
        inputs=state_translation_center,
        outputs=text_translation_center,
    )
    def on_state_translation_center_change(state_translation_center):
        """Update text box with the center of the translation"""
        if state_translation_center is not None:
            return f"Center of mask: {state_translation_center[0]}, {state_translation_center[1]}"
        else:
            return "Click on the image to get the center of the translation"
    
    @text_number_of_frames.submit(
        inputs=[image_editing,
                state_translation_center,
                state_motion_option,
                text_number_of_frames],
        outputs=[state_number_of_frames, image_guidance, state_frame_points]
    )
    def on_state_number_of_frames_change(image, mask_center, motion_state, text):
        """Update text box with the number of frames"""
        if text.isdigit() and not text == "0":
            new_frames = int(text)
            new_image, new_frame_points = on_paint(image, mask_center, new_frames, motion_state)
            return new_frames, new_image, new_frame_points
        else:
            return None, image['background'], None
    
    @image_editing.apply(
        inputs=[image_editing, 
                state_translation_center, 
                state_number_of_frames, 
                state_motion_option],
        outputs=[image_guidance, 
                 state_frame_points]
    )
    def on_paint(image, mask_center, frames, motion_state):
        """Apply an approximation of the motion from the mask center
        to the scribble."""
        
        # If no mask center is provided, return the original image
        if mask_center is None:
            return image['background'], None
        
        # Make the scribble a binary mask
        scribble = cv2.cvtColor(image['layers'][0], cv2.COLOR_RGBA2GRAY)
        scribble[scribble > 0] = 1
        
        # Save the image overlayed with the scribble
        overlayed_image = image['background'].copy()
        scribble_idx = np.argwhere(scribble > 0)
        overlayed_image[scribble_idx[:, 0], scribble_idx[:, 1]] = image['layers'][0][scribble_idx[:, 0], scribble_idx[:, 1]]
        plt.imshow(overlayed_image)
        plt.axis('off')
        plt.savefig('scribble_overlayed.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        np.save('scribble_overlayed.npy', overlayed_image)
        
        # If no scribble is drawn, return the original image
        if scribble.sum() == 0:
            return image['background'], None
        plt.imshow(scribble)
        plt.axis('off')
        plt.savefig('scribble.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        np.save('scribble.npy', scribble)
        frame_points = make_scribble_to_frames(scribble = scribble, 
                                               sam_center = mask_center, 
                                               frames = frames)
        
        arrow_color = (0, 255, 0)
        arrow_width = 5
        image = image['background']
        
        if motion_state == "Translate":
            frame_points[0] = mask_center # Make sure the first point is the center of the mask
        elif motion_state == "Rotate":
            # Center all points to the mass center
            cx, cy = mask_center
            centered_points = frame_points - mask_center
            centered_points[:, 1] = -centered_points[:, 1]  # Invert the y-axis to account for the top-left origin
            
            # Compute angles for each point
            angles = np.arctan2(centered_points[:, 1], centered_points[:, 0])
            angles = np.unwrap(angles)

            r = 50 # NOTE: This is a hardcoded radius for the rotation
            frame_points = np.column_stack([cx + r * np.cos(angles),
                                            cy - r * np.sin(angles)  # Subtract to account for inverted y-axis
                                            ]).astype(int)
        else:
            raise NotImplementedError("Only translation and rotation is implemented for now.")
        
        # Draw an arrow from the start point to the clicked point
        for start_point, end_point in zip(frame_points[:-1], frame_points[1:]):
            image = cv2.arrowedLine(img = image,
                                    pt1 = tuple(start_point), 
                                    pt2 = tuple(end_point),
                                    color = arrow_color, 
                                    thickness=arrow_width, 
                                    )
        plt.imshow(image)
        plt.axis('off')
        plt.savefig('estimated_motion_guidance.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        np.save("estimated_motion_guidance.npy", image)
        
        return image, frame_points
    
    # Make the button to save the frame points visible
    @state_frame_points.change(
        inputs=[state_frame_points],
        outputs=[button_make_video],
    )
    def on_new_frame_points(frame_points):
        """Make the button to save the frame points visible"""
        # Only make it possible to make a video if there are frame points
        if frame_points is not None:
            return gr.update(interactive=True)
        else:
            return gr.update(interactive=False)
    
    # Save the frame points to an npy file
    @button_make_video.click(
        inputs=[state_frame_points],
        outputs=[button_make_video],
    )
    def on_make_video(frame_points):
        if frame_points is not None:
            # Save the frame points to a npy file
            np.save(f"frame_points-{datetime.now().strftime('%Y%m%dT%H%M%SZ')}.npy", frame_points)
            return gr.update(interactive=False)
    
    # Make the editing interactable depending on the button state
    @button_motion_option.select(
        inputs=[],
        outputs=[image_editing, state_motion_option],
    )
    def on_button_select(evt: gr.SelectData):
        """Change the button state and update the stored state"""
        
        if evt.value == "Segment":
            return gr.update(interactive=False), evt.value
        elif evt.value == "Translate":
            return gr.update(interactive=True), evt.value
        elif evt.value == "Rotate":
            return gr.update(interactive=True), evt.value
        else:
            raise ValueError("When did this happen?")
        
    
if __name__ == "__main__":  
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind-wildcard', action='store_true')
    args = parser.parse_args()
    demo.launch(server_name='0.0.0.0' if args.bind_wildcard else None)

