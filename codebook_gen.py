import os
import numpy as np
from scipy.cluster.vq import kmeans
import joblib

# Configuration
# cluster_count = 200
# iterations = 1

def generate_codebook(cache_dir, cluster_count, iterations):

    descriptors_all = np.load(os.path.join(cache_dir, "extracted_descriptors.npy"), allow_pickle=True)
    print(type(descriptors_all))
    print("Loaded descriptors:", descriptors_all.shape)

    concatenated_descriptors = np.concatenate([desc.astype("float") for desc in descriptors_all])

    codebook, variance = kmeans(concatenated_descriptors, cluster_count, iterations)

    # Save the codebook
    joblib.dump((cluster_count, codebook), os.path.join(cache_dir, "codebook.plk"), compress=3)

    return type(descriptors_all), f"Loaded descriptors: {descriptors_all.shape}"

if __name__ == "__main__":
    cache_dir = "./dataset03/cache"  # Specify the manual cache directory here
    generate_codebook(cache_dir, 200, 1)
