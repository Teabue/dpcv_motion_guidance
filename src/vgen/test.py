import cv2
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import skfmm
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, gaussian_filter

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

if __name__ == "__main__":
    image = plt.imread("apple/apple.jpg")
    scribble = np.load("scribble.npy")
    mask = np.load("mask.npy")
    mask_center = np.array([186, 236])
    frames = 8

    distance = get_distances_from_mask_center(scribble, mask_center)
    y_coords, x_coords = np.nonzero(~distance.mask)

    # Compute sum of distances and count per bin
    sum_heatmap, _, _ = np.histogram2d(x_coords, y_coords, bins=(100, 30), weights=distance.compressed())
    count_heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=(100, 30))

    # Avoid division by zero
    average_heatmap = np.divide(sum_heatmap, count_heatmap, out=np.zeros_like(sum_heatmap), where=count_heatmap!=0)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    points = make_scribble_to_frames(scribble, mask_center, frames = frames)
    points[0] = mask_center

    # ---- Plotting ----
    # Plot the image
    plt.imshow(image)

    # Plot the SAM mask
    # Overlay the mask on the image with transparency
    masked_image = np.ma.masked_where(mask == 0, mask)
    plt.imshow(masked_image, cmap="Accent", alpha=0.5)

    # Plot the heatmap
    # Set black (zero values) to be fully transparent
    average_heatmap_masked = np.ma.masked_where(average_heatmap == 0, average_heatmap)
    plt.imshow(average_heatmap_masked.T, extent=extent, origin='lower', cmap='hot', interpolation='nearest', alpha=0.7)
    plt.colorbar(label='Distance')
    plt.gca().invert_yaxis()

    # Plot the scribble mask
    plt.plot(mask_center[0], mask_center[1], 'ro', markersize=5, label ='SAM center')


    # Plot the points fitted points on the image
    x_coords, y_coords = zip(*points)
    plt.scatter(x_coords, y_coords, c="purple", s=10, label = 'Fitted Points')

    sam_color = plt.cm.Accent(0)
    legend_patch = Patch(color=sam_color, label='SAM Mask')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(legend_patch)
    labels.append('SAM Mask')

    plt.legend(handles=handles, labels=labels, loc ='lower right')
    plt.title('Fitted Points on Scribble')
    plt.axis('off')
    plt.savefig("assets/fitted_points.png", dpi=300, bbox_inches='tight')
    plt.show()