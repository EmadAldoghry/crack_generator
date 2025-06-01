import cv2
from PIL import Image # For JP2 and other formats
import numpy as np
import os
import matplotlib.pyplot as plt # Import for displaying the final image

# Assuming line_drawer_gui and crack_generation_engine are in the same directory
# or the package is installed. For direct script running, relative imports:
from line_drawer_gui import LineDrawer
from crack_generation_engine import create_crack_mask_from_line, apply_crack_to_image

def run_crack_painter_with_hardcoded_inputs():
    # --- Hardcode your inputs here ---
    input_image_path = "crack_generator/data/dop10rgbi_32_293_5629_1_nw_2023.jp2"  # <--- CHANGE THIS TO YOUR JP2 IMAGE PATH
    output_image_path = "crack_generator/data/cracked_img.png" # <--- CHANGE THIS IF NEEDED

    # Crack appearance parameters
    crack_avg_width = 3
    crack_waviness_scale = 5.0 # How much the crack deviates (larger is more wavy)
    crack_intensity_avg = 0.05 # Darkness of the crack (0.0 very dark, 1.0 invisible)
    random_seed = None # Use None for random, or a number for reproducibility
    # --- End of hardcoded inputs ---

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        # Pillow can handle JP2 if OpenJPEG is installed
        # pip install Pillow openjpeg
        pil_image = Image.open(input_image_path)
        print(f"Successfully opened JP2 image: {input_image_path} with mode {pil_image.mode}")

        # Convert to NumPy array, ensure it's RGB or Grayscale
        if pil_image.mode == 'RGBA':
            image_np = np.array(pil_image.convert('RGB'))
            print("Converted RGBA to RGB")
        elif pil_image.mode == 'P': # Palette
            image_np = np.array(pil_image.convert('RGB'))
            print("Converted Palette to RGB")
        elif pil_image.mode == 'L': # Grayscale
            image_np = np.array(pil_image) # Keep as grayscale
            print("Loaded as Grayscale")
        elif pil_image.mode == 'RGB':
            image_np = np.array(pil_image)
            print("Loaded as RGB")
        elif pil_image.mode.startswith('I') or pil_image.mode == 'F': # Integer or Float grayscale
             # Convert to uint8 for display and common processing
            if pil_image.mode == 'I;16' or pil_image.mode == 'I;16B' or pil_image.mode == 'I;16L': # 16-bit int
                temp_array = np.array(pil_image, dtype=np.int32) # Read as int32 to avoid overflow
                image_np = (temp_array / 256).astype(np.uint8) # Scale to 8-bit
                print(f"Converted 16-bit int grayscale {pil_image.mode} to 8-bit uint8")
            elif pil_image.mode == 'F': # 32-bit float
                temp_array = np.array(pil_image, dtype=np.float32)
                if np.max(temp_array) <= 1.0 and np.min(temp_array) >=0.0 : # Assuming 0-1 range
                    image_np = (temp_array * 255).astype(np.uint8)
                else: # Try to normalize if not in 0-1 range
                    temp_array = (temp_array - np.min(temp_array)) / (np.max(temp_array) - np.min(temp_array) + 1e-6)
                    image_np = (temp_array * 255).astype(np.uint8)
                print(f"Converted float grayscale {pil_image.mode} to 8-bit uint8")
            else:
                # For other integer modes, try a direct conversion and hope for the best
                image_np = np.array(pil_image.convert('L'))
                print(f"Converted integer mode {pil_image.mode} to 8-bit L (Grayscale)")

        else: # For other modes, attempt conversion to RGB or L
            try:
                image_np = np.array(pil_image.convert('RGB'))
                print(f"Attempted conversion of mode {pil_image.mode} to RGB")
            except Exception:
                image_np = np.array(pil_image.convert('L'))
                print(f"Attempted conversion of mode {pil_image.mode} to L (Grayscale)")

        # Ensure it's BGR if it was color for OpenCV, or keep grayscale
        # Matplotlib expects RGB, PIL loads as RGB.
        if len(image_np.shape) == 3 and image_np.shape[2] == 4: # e.g. RGBA from some conversions
            image_np = image_np[..., :3] # Take first 3 channels (RGB)
            print("Took first 3 channels from a 4-channel image.")


    except FileNotFoundError:
        print(f"Error: JP2 image not found at {input_image_path}")
        return
    except Exception as e:
        print(f"Error loading JP2 image: {e}")
        print("Make sure you have the OpenJPEG library installed if Pillow needs it for JP2.")
        print("You can try installing it with: pip install Pillow openjpeg")
        return

    print("Image loaded. Displaying for line drawing...")

    # Use a copy for drawing GUI
    gui_image_display = image_np.copy()

    drawer = LineDrawer(gui_image_display)
    lines_to_crack = drawer.get_lines()

    if not lines_to_crack:
        print("No lines drawn. Exiting.")
        return

    print(f"\nProcessing {len(lines_to_crack)} lines to generate cracks...")

    # Make a fresh copy of the original image to apply cracks to
    image_with_cracks = image_np.copy()

    if random_seed is not None:
        np.random.seed(random_seed) # For global numpy randomness

    for i, (start_pt, end_pt) in enumerate(lines_to_crack):
        print(f"Generating crack {i+1}/{len(lines_to_crack)} from {start_pt} to {end_pt}")

        current_crack_seed = random_seed + i if random_seed is not None else None

        crack_mask = create_crack_mask_from_line(
            image_shape=image_with_cracks.shape,
            start_point=start_pt,
            end_point=end_pt,
            crack_avg_width=crack_avg_width,
            waviness_scale=crack_waviness_scale,
            rng_seed=current_crack_seed
        )

        image_with_cracks = apply_crack_to_image(
            base_image=image_with_cracks,
            crack_binary_mask=crack_mask,
            intensity_avg=crack_intensity_avg,
            rng_seed=current_crack_seed
        )

    try:
        # Convert back to PIL Image for saving
        if len(image_with_cracks.shape) == 3 and image_with_cracks.shape[2] == 3:
            output_pil_image = Image.fromarray(image_with_cracks, mode='RGB')
        else: # Grayscale
            output_pil_image = Image.fromarray(image_with_cracks, mode='L')

        output_pil_image.save(output_image_path)
        print(f"Output image saved to {output_image_path}")

        # Optionally display final image
        plt.figure(figsize=(10,10))
        plt.imshow(image_with_cracks, cmap='gray' if len(image_with_cracks.shape)==2 else None)
        plt.title(f"Image with Generated Cracks (Seed: {random_seed})")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error saving or displaying output image: {e}")

if __name__ == "__main__":
    # This will run the function with hardcoded values when the script is executed
    run_crack_painter_with_hardcoded_inputs()