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

    num_points = int(np.hypot(end_x - start_x, end_y - end_y)) * 2 
    if num_points < 2: num_points = 2
    line_x = np.linspace(start_x, end_x, num_points)
    line_y = np.linspace(start_y, end_y, num_points)

    progress = np.linspace(0, 1, num_points)
    perlin_repeat = max(num_points, 1024)

    deviations = pnoise1_array(progress * 10, 
                               octaves=waviness_octaves,
                               persistence=waviness_persistence,
                               lacunarity=waviness_lacunarity,
                               base=perlin_base,
                               repeat=perlin_repeat)

    deviations = (deviations - np.min(deviations)) / (np.max(deviations) - np.min(deviations) + 1e-6) 
    deviations = (deviations - 0.5) * 2 * waviness_scale 

    dx = end_x - start_x
    dy = end_y - start_y
    line_length = np.hypot(dx, dy)
    if line_length == 0: line_length = 1 

    norm_dx = dx / line_length
    norm_dy = dy / line_length
    
    perp_dx = -norm_dy
    perp_dy = norm_dx

    crack_points_x = line_x + perp_dx * deviations
    crack_points_y = line_y + perp_dy * deviations
    
    crack_points_x = np.clip(crack_points_x, 0, width - 1).astype(int)
    crack_points_y = np.clip(crack_points_y, 0, height - 1).astype(int)

    for i in range(len(crack_points_x) - 1):
        current_width_noise_val = noise.pnoise1(progress[i] / width_variation_scale,
                                                octaves=2, persistence=0.5, lacunarity=2.0,
                                                base=perlin_base + 1, repeat=perlin_repeat)
        normalized_width_noise = (current_width_noise_val / 0.7 + 1) / 2
        
        float_thickness = crack_avg_width * (1 - width_variation_strength + width_variation_strength * normalized_width_noise * 2)
        
        current_thickness = int(round(float_thickness)) 
        current_thickness = max(1, current_thickness) 

        pt1 = (crack_points_x[i], crack_points_y[i])
        pt2 = (crack_points_x[i+1], crack_points_y[i+1])
        # MODIFICATION: Try LINE_8 for a potentially crisper 1-pixel line in the mask
        cv2.line(mask, pt1, pt2, 255, thickness=current_thickness, lineType=cv2.LINE_8) # Changed from LINE_AA

    return mask


def apply_crack_to_image(base_image, crack_binary_mask,
                         intensity_avg=0.65, 
                         intensity_std=0.05, 
                         fade_kernel_size_min=1, # Default as per your provided code
                         fade_kernel_size_max=1, # Default as per your provided code
                         rng_seed=None,
                         target_crack_avg_width=1.0):
    """
    Applies the crack to the base image, modifying pixel intensities.
    For thin cracks (target_crack_avg_width < THIN_CRACK_THRESHOLD),
    the user-defined intensity_avg is used directly, and blur is kept minimal.
    """
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()

    # FIRST: Check if the binary mask has any activated pixels
    if np.sum(crack_binary_mask) == 0:
        # print("Warning: crack_binary_mask is empty. No crack will be applied.")
        return base_image.copy() # Return a copy of the original image if mask is empty

    output_image = base_image.astype(np.float32) / 255.0 
    intensity_mask_values = np.ones(base_image.shape[:2], dtype=np.float32) 

    crack_pixels_y, crack_pixels_x = np.where(crack_binary_mask == 255)

    current_intensity_avg = intensity_avg # Start with the user-provided intensity_avg
    current_fade_kernel_min = fade_kernel_size_min
    current_fade_kernel_max = fade_kernel_size_max

    # You might need to tune this THIN_CRACK_THRESHOLD
    # If cracks are visible at width 3, maybe this threshold should be around 2.5 or 3.0
    THIN_CRACK_THRESHOLD = 3.0 # Adjusted based on "visible > 3" feedback

    if target_crack_avg_width < THIN_CRACK_THRESHOLD:
        # For thin cracks, USE THE PROVIDED intensity_avg DIRECTLY.
        # The "fainter" logic was making them too invisible when intensity_avg is already low.
        # current_intensity_avg is already set to intensity_avg, so no change needed here for that.
        
        # Ensure blur is minimal for these thin cracks
        # Force a very small kernel. Given defaults are 1,1 this will result in 1 or 3.
        current_fade_kernel_min = min(3, fade_kernel_size_min) 
        current_fade_kernel_max = max(current_fade_kernel_min, 3) # Effectively kernel can be 1 or 3

        # If you want to absolutely force a 1x1 or 3x3 kernel for thin cracks:
        # current_fade_kernel_min = 1 # or 3
        # current_fade_kernel_max = 1 # or 3
        
        # print(f"Thin crack (w={target_crack_avg_width:.2f}): using intensity_avg={current_intensity_avg:.2f}, kernel_min={current_fade_kernel_min}, kernel_max={current_fade_kernel_max}")
    # else:
        # print(f"Thick crack (w={target_crack_avg_width:.2f}): using intensity_avg={current_intensity_avg:.2f}, kernel_min={current_fade_kernel_min}, kernel_max={current_fade_kernel_max}")


    for y, x in zip(crack_pixels_y, crack_pixels_x):
        pixel_intensity_multiplier = rng.normal(current_intensity_avg, intensity_std)
        intensity_mask_values[y, x] = np.clip(pixel_intensity_multiplier, 0.01, 0.99)


    kernel_size_val = rng.integers(current_fade_kernel_min, current_fade_kernel_max + 1)
    if kernel_size_val == 0: # Should not happen with current logic if min >= 1
        kernel_size_val = 1 
    if kernel_size_val % 2 == 0: kernel_size_val +=1 
    
    temp_intensity_for_blur = np.ones_like(intensity_mask_values)
    temp_intensity_for_blur[crack_pixels_y, crack_pixels_x] = intensity_mask_values[crack_pixels_y, crack_pixels_x]

    # If kernel_size_val is 1, GaussianBlur with sigma 0 effectively does nothing or minimal change.
    # If you want NO blur for kernel_size_val == 1, you could skip this step.
    if kernel_size_val > 1:
        blurred_intensity_mask = cv2.GaussianBlur(temp_intensity_for_blur, (kernel_size_val, kernel_size_val), 0)
    else:
        blurred_intensity_mask = temp_intensity_for_blur.copy() # No blur if kernel is 1
    
    final_intensity_mask = blurred_intensity_mask
    # Reinforce: ensure core crack pixels are at least as dark as intended before blur
    final_intensity_mask[crack_pixels_y, crack_pixels_x] = np.minimum(
        final_intensity_mask[crack_pixels_y, crack_pixels_x],
        intensity_mask_values[crack_pixels_y, crack_pixels_x] 
    )

    if len(output_image.shape) == 3:
        for c in range(output_image.shape[2]):
            output_image[..., c] *= final_intensity_mask
    else: 
        output_image *= final_intensity_mask

    return (np.clip(output_image, 0, 1) * 255).astype(np.uint8)