import cv2
import os

# Set paths for input and output folders
input_folder = '/Volumes/Expansion/20240928/20240926_a2780_post_drug__2024-09-26T22_59_12-Measurement 1/Sobel_combined_images/ch1/combined'
output_folder = '/Volumes/Expansion/20240928/20240926_a2780_post_drug__2024-09-26T22_59_12-Measurement 1/Sobel_combined_images/Enhanced'

# Check if output folder exists, create if not
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each file in the input folder
for filename in os.listdir(input_folder):
    if not filename.startswith('.') and filename.endswith('.tif') or filename.endswith('.tiff'):  # Check if file is a .tiff image
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error: Could not open or find the image {filename}.")
            continue

        # # Create CLAHE object
        # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2,2))
        # Apply CLAHE for adaptive histogram equalization
        # enhanced_img = clahe.apply(img)

        # Increase exposure
        alpha = 2 # Control contrast (>1 to increase contrast)
        beta = 1   # Control brightness (>0 to increase brightness)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Save enhanced image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path,adjusted)
        # cv2.imshow("Adjusted Image", adjusted)
        # cv2.waitKey(0)
        print(f"Processed and saved: {filename}")

print("Processing complete!")