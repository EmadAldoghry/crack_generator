import numpy as np
import cv2
import noise # From syncrack_generator's dependencies
from scipy import ndimage # From syncrack_generator's dependencies
from copy import deepcopy

# --- Helper functions potentially borrowed or adapted from syncrack_generator.image_generation ---
# --- Or from syncrack_generator.noisy_labels (like get_windows, join_windows if used) ---

# For simplicity, we'll assume a function to get Perlin noise similar to syncrack
def pnoise1_array(points, octaves=4, persistence=0.5, lacunarity=2.0, base=0, repeat=1024):
    """Generates 1D Perlin noise for an array of points."""
    return np.array([noise.pnoise1(p, octaves=octaves, persistence=persistence,
                                   lacunarity=lacunarity, repeat=repeat, base=base)
                     for p in points])

def create_crack_mask_from_line(image_shape, start_point, end_point,
                                crack_avg_width=3,
                                waviness_scale=50.0, # How far points deviate perpendicular to the line
                                waviness_octaves=4,
                                waviness_persistence=0.6,
                                waviness_lacunarity=2.0,
                                width_variation_strength=0.5, # 0 to 1, how much width varies
                                width_variation_scale=0.1, # Scale of perlin noise for width
                                rng_seed=None):
    """
    Generates a binary crack mask along a specified line.
    """
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
        perlin_base = rng.integers(0, 10000)
    else:
        rng = np.random.default_rng()
        perlin_base = rng.integers(0, 10000)


    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    start_x, start_y = int(start_point[0]), int(start_point[1])
    end_x, end_y = int(end_point[0]), int(end_point[1])

    # Generate points along the line
    num_points = int(np.hypot(end_x - start_x, end_y - end_y)) * 2 # More points for smoother curves
    if num_points < 2: num_points = 2
    line_x = np.linspace(start_x, end_x, num_points)
    line_y = np.linspace(start_y, end_y, num_points)

    # Calculate perpendicular deviations using Perlin noise
    # Normalize line progress for Perlin noise input
    progress = np.linspace(0, 1, num_points)
    
    # Ensure 'repeat' is large enough for pnoise1, e.g., num_points or a fixed large value
    perlin_repeat = max(num_points, 1024)

    deviations = pnoise1_array(progress * 10, # Multiply progress to get more variations
                               octaves=waviness_octaves,
                               persistence=waviness_persistence,
                               lacunarity=waviness_lacunarity,
                               base=perlin_base,
                               repeat=perlin_repeat)

    deviations = (deviations - np.min(deviations)) / (np.max(deviations) - np.min(deviations) + 1e-6) # Normalize 0-1
    deviations = (deviations - 0.5) * 2 * waviness_scale # Scale and center

    # Vector perpendicular to the line
    dx = end_x - start_x
    dy = end_y - start_y
    line_length = np.hypot(dx, dy)
    if line_length == 0: line_length = 1 # Avoid division by zero

    norm_dx = dx / line_length
    norm_dy = dy / line_length
    
    perp_dx = -norm_dy
    perp_dy = norm_dx

    crack_points_x = line_x + perp_dx * deviations
    crack_points_y = line_y + perp_dy * deviations
    
    # Ensure points are within bounds
    crack_points_x = np.clip(crack_points_x, 0, width - 1).astype(int)
    crack_points_y = np.clip(crack_points_y, 0, height - 1).astype(int)

    # Draw the crack path with varying thickness
    for i in range(len(crack_points_x) - 1):
        # Width variation using Perlin noise along the crack length
        current_width_noise_val = noise.pnoise1(progress[i] / width_variation_scale,
                                                octaves=2, persistence=0.5, lacunarity=2.0,
                                                base=perlin_base + 1, repeat=perlin_repeat)
        # Normalize noise from approx -0.7 to 0.7 to 0-1
        normalized_width_noise = (current_width_noise_val / 0.7 + 1) / 2
        
        current_thickness = int(crack_avg_width * (1 - width_variation_strength + width_variation_strength * normalized_width_noise * 2))
        current_thickness = max(1, current_thickness)

        pt1 = (crack_points_x[i], crack_points_y[i])
        pt2 = (crack_points_x[i+1], crack_points_y[i+1])
        cv2.line(mask, pt1, pt2, 255, thickness=current_thickness, lineType=cv2.LINE_AA)

    # Optional: Further morphological operations like in original create_crack_shape
    # e.g., small random erosions/dilations on segments if desired for more detail.
    # For simplicity, we'll skip the windowed approach for now.

    return mask


def apply_crack_to_image(base_image, crack_binary_mask,
                         intensity_avg=0.65, # Darker cracks (0.0) vs lighter (1.0)
                         intensity_std=0.05, # Variation in darkness
                         fade_kernel_size_min=3, # Min blur kernel for edges
                         fade_kernel_size_max=7, # Max blur kernel for edges
                         rng_seed=None):
    """
    Applies the crack to the base image, modifying pixel intensities.
    This adapts logic from syncrack_generator.image_generation.add_crack.
    """
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()

    output_image = base_image.astype(np.float32) / 255.0 # Work with float 0-1

    # Create a float mask for intensity modification
    # Values closer to 0 mean darker crack, 1 means no change
    intensity_mask = np.ones(base_image.shape[:2], dtype=np.float32)

    crack_pixels_y, crack_pixels_x = np.where(crack_binary_mask == 255)

    for y, x in zip(crack_pixels_y, crack_pixels_x):
        # Random intensity for each crack pixel based on global average
        pixel_intensity_multiplier = rng.normal(intensity_avg, intensity_std)
        intensity_mask[y, x] = np.clip(pixel_intensity_multiplier, 0.01, 0.99) # Ensure it darkens

    # Blurring/fading the edges of the crack for realism
    # Create a slightly blurred version for soft transitions
    kernel_size = rng.integers(fade_kernel_size_min, fade_kernel_size_max + 1)
    if kernel_size % 2 == 0: kernel_size +=1 # Ensure odd kernel
    
    # Apply blur only to the crack region to create faded edges
    # This is a simplified fading compared to the complex windowed one in syncrack
    temp_intensity_for_blur = np.ones_like(intensity_mask)
    temp_intensity_for_blur[crack_pixels_y, crack_pixels_x] = intensity_mask[crack_pixels_y, crack_pixels_x]

    blurred_intensity_mask = cv2.GaussianBlur(temp_intensity_for_blur, (kernel_size, kernel_size), 0)
    
    # Combine: use original intensity for core crack, blurred for edges
    # This ensures core crack isn't too faded by a large blur
    final_intensity_mask = np.ones_like(intensity_mask)
    # Where original binary mask is, use the potentially sharper intensity_mask values
    final_intensity_mask[crack_pixels_y, crack_pixels_x] = intensity_mask[crack_pixels_y, crack_pixels_x] 
    # Blend with blurred mask outside the core binary mask to smooth transitions
    # A more sophisticated blending might be needed for perfect results
    # For now, let's use the blurred mask globally and then reinforce the core
    final_intensity_mask = blurred_intensity_mask
    # Reinforce core crack if blur was too strong
    final_intensity_mask[crack_pixels_y, crack_pixels_x] = np.minimum(
        final_intensity_mask[crack_pixels_y, crack_pixels_x],
        intensity_mask[crack_pixels_y, crack_pixels_x] # Take the darker (smaller) value
    )


    # Apply the intensity mask to all channels
    if len(output_image.shape) == 3:
        for c in range(output_image.shape[2]):
            output_image[..., c] *= final_intensity_mask
    else: # Grayscale
        output_image *= final_intensity_mask

    return (np.clip(output_image, 0, 1) * 255).astype(np.uint8)