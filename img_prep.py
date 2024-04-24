import os
import cv2
import torch
import numpy as np

from lightglue import DISK

# Constants
IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'raw', 'tif', 'ppm']

# Initialize Torch and Device
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extractor = DISK(max_num_keypoints=2048).eval().to(device)

def prepare_images(cache_dir, src_dir, max_num_keypoints):
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Get the extractor here
    extractor = get_extractor(max_num_keypoints)

    # Get a list of image files in the directory
    image_files = sorted([f for f in os.listdir(src_dir) if is_image_file(f)])

    with open(os.path.join(cache_dir, "visual_list.txt"), "wt") as text_file:
        extracted_descriptors = []
        extracted_points = []
        extracted_colors = []
        extracted_size_data = []

        for image_name in image_files:
            if is_valid_image(image_name):
                text_file.write(image_name + '\n')
                img = cv2.imread(os.path.join(src_dir, image_name))
                
                if img.shape[1] < img.shape[0]:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_ = img.transpose((2, 0, 1))
                img_ = torch.tensor(img_ / 255., dtype=torch.float).to(device)
                
                feats = extractor.extract(img_)
                descriptors = feats['descriptors'].cpu().detach().numpy()
                keypoints = feats['keypoints'].cpu().detach().numpy()
                size = feats['image_size'].cpu().detach().numpy()
                colors = get_colors_from_points(img, keypoints)
                
                # print(f"{image_files} : {colors}")

                extracted_size_data.append(size[0])
                extracted_points.append(keypoints[0])
                extracted_descriptors.append(descriptors[0])
                extracted_colors.append(colors)

        # Save the extracted data
        save_data(cache_dir, extracted_descriptors, extracted_points, extracted_colors, extracted_size_data)
        # print(f"extracted_descriptors: {extracted_descriptors}")
        # print(extracted_points)
        # print(f"extracted_colors: {extracted_colors}")
        # print(extracted_size_data)


def get_extractor(max_num_keypoints):
    return DISK(max_num_keypoints=max_num_keypoints).eval().to(device)

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)

def is_valid_image(filename):
    return filename.split('.')[-1].lower() in IMG_EXTENSIONS

def get_colors_from_points(img, points):
    colors = [img[int(y), int(x), :] for x, y in points[0]]
    return colors

def save_data(cache_dir, extracted_descriptors, extracted_points, extracted_colors, extracted_size_data):
    np.save(os.path.join(cache_dir, 'extracted_descriptors.npy'), np.array(extracted_descriptors, dtype=object))
    np.save(os.path.join(cache_dir, 'extracted_points.npy'), np.array(extracted_points, dtype=object))
    np.save(os.path.join(cache_dir, 'color_data.npy'), np.array(extracted_colors, dtype=object))
    np.save(os.path.join(cache_dir, 'extracted_size_data.npy'), np.array(extracted_size_data, dtype=object))


if __name__ == "__main__":
    cache_dir = "./dataset03/cache"  # Specify the manual cache directory here
    src_dir = "./dataset03/"
    prepare_images(cache_dir, src_dir, 2048)