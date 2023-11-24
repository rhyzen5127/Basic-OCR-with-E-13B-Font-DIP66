import os
from PIL import Image, ImageOps

def augment_image(image_path, output_dir, base_name, augment_count=1300):
    """
    Apply stretch augmentations to an image and save them to the output directory.
    """
    with Image.open(image_path) as img:
        # Ensure image is in black and white
        img = img.convert('L')

        # Original image
        img.save(os.path.join(output_dir, f"{base_name}.png"))

        # Apply different augmentations
        for i in range(1, augment_count + 1):
            # Example augmentations: Stretching
            # Adjust the scale factor slightly for each augmentation
            scale_x, scale_y = 1 + 0.01 * i, 1 + 0.01 * i
            new_size = (int(28 * scale_x), int(28 * scale_y))
            augmented_img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Crop or pad to get back to 28x28 if necessary
            if augmented_img.size != (28, 28):
                augmented_img = ImageOps.fit(augmented_img, (28, 28), Image.Resampling.LANCZOS)

            augmented_img.save(os.path.join(output_dir, f"{base_name}-{i}.png"))

def process_folder(root_folder):
    """
    Process each subfolder in the root folder to apply augmentations.
    """
    for folder_name in sorted(os.listdir(root_folder)):
        subfolder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(subfolder_path):
            print(f"Processing folder: {subfolder_path}")
            for image_file in os.listdir(subfolder_path):
                if image_file.endswith('.png'):
                    image_path = os.path.join(subfolder_path, image_file)
                    base_name = os.path.splitext(image_file)[0]
                    augment_image(image_path, subfolder_path, base_name)

if __name__ == "__main__":
    data_root = "C:/Users/RhyzenWorkspace/OneDrive/ocr/data/E-13B"
    process_folder(data_root)