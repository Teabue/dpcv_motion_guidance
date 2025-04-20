import cv2
import numpy as np
import matplotlib.pyplot as plt
import skfmm
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

def get_sorted_coordinates(masked_array):
    """
    Get a sorted array of coordinates based on the smallest distances
    in a 2D masked array.

    Args:
        masked_array (np.ma.MaskedArray): A 2D masked array of distances.

    Returns:
        np.ndarray: An array of shape (H*W, 2) with sorted coordinates.
    """
    # Flatten the masked array and get valid (non-masked) indices
    valid_indices = np.ma.nonzero(masked_array)
    distances = masked_array[valid_indices]

    # Combine valid indices into a list of coordinates
    coordinates = np.array(list(zip(valid_indices[1], valid_indices[0])))

    # Sort the coordinates by their corresponding distances
    sorted_indices = np.argsort(distances)
    sorted_coordinates = coordinates[sorted_indices]

    return sorted_coordinates

def get_distances_from_mask_center(paint_mask: np.ndarray, mask_center: np.ndarray) -> np.ndarray:
    """Computes the geodesic distance from the mask center to the rest of the mask.

    Args:
        paint_mask (np.ndarray): The mask from the scribble
        mask_center (np.ndarray): The center of the SAM mask

    Returns:
        np.ndarray: The geodesic distance from the mask center to the rest of the mask
    """
    # Compute distances from the mask center in the skeletonized mask
    mask_inv = ~paint_mask.astype(bool)
    m = np.ones_like(paint_mask)
    m[mask_center[1], mask_center[0]] = 0
    m_masked = np.ma.masked_array(distance_transform_edt(m), mask_inv)

    # Reconfigure the 0 contour by getting the closest point to the center
    smallest = np.argwhere(m_masked == np.min(m_masked))[0]
    m_masked[smallest[0], smallest[1]] = 0

    # Use fast marching to compute distances
    distance = skfmm.distance(m_masked, dx=1)
    
    return distance

def fit_points_to_skeleton(skeleton: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Fits the points to the skeleton of the mask.
    Args:
        skeleton (np.ndarray): The skeleton of the mask
        points (np.ndarray): The points to fit to the skeleton
    Returns:
        np.ndarray: The fitted points
    """
    # Get the coordinates of the mask
    skele_coords = np.argwhere(skeleton)
    skele_coords = np.flip(skele_coords, axis=1)

    # Find the coordinates closest to the skeleton
    distances = np.linalg.norm(points[:, None, :] - skele_coords[None, : , :], axis=2) # Shape (N, M)
    closest_coords = np.argmin(distances, axis=1)
    points = skele_coords[closest_coords]
    
    return points

def preprocess_scribble(mask: np.ndarray, mask_center: np.ndarray) -> np.ndarray:
    """Preprocesses the scribble mask to extract the geodesic distance from the center.

    Args:
        mask (np.ndarray): The scribble mask
        mask_center (np.ndarray): The center of the SAM mask

    Returns:
        np.ndarray: The preprocessed scribble mask
    """
    if len(mask.shape) == 3:
        # Make the mask binary
        mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
    mask[mask > 0] = 1
    
    # If no mask is drawn, return the original image
    if mask.sum() == 0:
        return None
    
    # Compute distances from the mask center in the skeletonized mask
    distance = get_distances_from_mask_center(mask, mask_center)
    sorted_distances = get_sorted_coordinates(distance)
    
    # Space the coordinates into N points
    N = 10
    step = len(sorted_distances) // N
    spaced_coords = sorted_distances[::step]
    spaced_coords = np.array(spaced_coords)
    
    # Skeletonize the mask
    mask_skeleton = skeletonize(mask) 
    
    # Fit the points to the skeleton
    points = fit_points_to_skeleton(mask_skeleton, spaced_coords)
    
    return points

mask = np.load("mask_round.npy")
mask_center = np.array([418, 209])

points = preprocess_scribble(mask, mask_center)

plt.imshow(mask, cmap='gray')
color = plt.cm.rainbow(np.linspace(0, 1, len(points)))
x_coords, y_coords = zip(*points)
plt.scatter(x_coords, y_coords, c=color, s=10)
plt.show()


# def get_sorted_coordinates(masked_array):
#     """
#     Get a sorted array of coordinates based on the smallest distances
#     in a 2D masked array.

#     Args:
#         masked_array (np.ma.MaskedArray): A 2D masked array of distances.

#     Returns:
#         np.ndarray: An array of shape (H*W, 2) with sorted coordinates.
#     """
#     # Flatten the masked array and get valid (non-masked) indices
#     valid_indices = np.ma.nonzero(masked_array)
#     distances = masked_array[valid_indices]

#     # Combine valid indices into a list of coordinates
#     coordinates = np.array(list(zip(valid_indices[1], valid_indices[0])))

#     # Sort the coordinates by their corresponding distances
#     sorted_indices = np.argsort(distances)
#     sorted_coordinates = coordinates[sorted_indices]

#     return sorted_coordinates

# # MY CODE
# DEBUG = False

# mask = np.load("mask_round.npy")
# mask_center = np.array([418, 209])

# # Compute distances from the mask center in the skeletonized mask
# mask_inv = ~mask.astype(bool)
# m = np.ones_like(mask)
# m[mask_center[1], mask_center[0]] = 0
# m_masked = np.ma.masked_array(distance_transform_edt(m), mask_inv)

# # Reconfigure the 0 contour by getting the closest point to the center
# smallest = np.argwhere(m_masked == np.min(m_masked))[0]
# m_masked[smallest[0], smallest[1]] = 0

# # Use fast marching to compute distances
# distance = skfmm.distance(m_masked, dx=1)

# sorted_distances = get_sorted_coordinates(distance)

# if DEBUG:
#     color = plt.cm.rainbow(np.linspace(0, 1, len(sorted_distances)))
#     x_coords, y_coords = zip(*sorted_distances)
#     plt.scatter(x_coords, y_coords, c=color, s=10)

# # Space the coordinates into N points
# N = 10
# step = len(sorted_distances) // N
# spaced_coords = sorted_distances[::step]
# spaced_coords = np.array(spaced_coords)

# # Skeletonize the mask
# mask_skeleton = skeletonize(mask) 
# plt.imshow(mask_skeleton)

# skele_coords = np.argwhere(mask_skeleton)
# skele_coords = np.flip(skele_coords, axis=1)

# # Find the coordinates closest to the skeleton
# distances = np.linalg.norm(spaced_coords[:, None, :] - skele_coords[None, : , :], axis=2) # Shape (N, M)
# closest_coords = np.argmin(distances, axis=1)
# points = skele_coords[closest_coords]

# for point in points:
#     plt.plot(point[0], point[1], 'ro')

# plt.show()