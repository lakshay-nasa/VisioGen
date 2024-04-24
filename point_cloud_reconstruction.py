import torch
import numpy as np
from lightglue import LightGlue
from lightglue.utils import rbd
import joblib
from scipy.cluster.vq import vq
from numpy.linalg import norm
import os

def generate_point_cloud(cache_dir, similarity_threshold):
    # Load data
    extracted_descriptors = np.load(os.path.join(cache_dir, "extracted_descriptors.npy"), allow_pickle=True)
    extracted_points = np.load(os.path.join(cache_dir, "extracted_points.npy"), allow_pickle=True)
    img_size_data = np.load(os.path.join(cache_dir, "extracted_size_data.npy"), allow_pickle=True)
    num_clusters, codebook_value = joblib.load(os.path.join(cache_dir, "codebook.plk"))


    # Enable/disable GPU
    torch.set_grad_enabled(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matcher = LightGlue(features='disk').eval().to(device)

    # Step 1: Compute TF-IDF
    print("Computing TF-IDF")
    visual_word_histograms = []

    for descriptors in extracted_descriptors:
        img_visual_words, _ = vq(descriptors.astype("float"), codebook_value)
        visual_word_histograms.append(img_visual_words)

    frequency_vectors = np.array([[np.sum(img_visual_words == word) for word in range(num_clusters)] for img_visual_words in visual_word_histograms])
    N = frequency_vectors.shape[0]

    document_frequency = np.sum(frequency_vectors > 0, axis=0)
    inverse_document_frequency = np.log(N / (document_frequency + 1))  # Add 1 to avoid division by zero
    tfidf = frequency_vectors * inverse_document_frequency

    # Step 2: Compute Similarity Matrix
    print("Computing Similarity Matrix")
    cosine_similarity_matrix = np.dot(tfidf, tfidf.T) / (norm(tfidf, axis=1)[:, np.newaxis] * norm(tfidf, axis=1))

    # Step 3: Build Connection Graph
    print("Building Connection Graph")
    # similarity_threshold = 0.75
    connections = {}
    for i in range(N):
        connections[i] = [j for j, sim in enumerate(cosine_similarity_matrix[i]) if i != j and sim > float(similarity_threshold)]

    # Step 4: Find the Starting Node
    print("Finding the Starting Node")
    max_connections = 0
    starting_node = 0
    for i, c in enumerate(connections.values()):
        if len(c) > max_connections:
            max_connections = len(c)
            starting_node = i

    # Step 5: Generate 3D Points
    print("Generating 3D Points")
    point3d_index = 0
    all_matches = []
    all_3d_points = [None] * extracted_points.shape[0]
    queue = [(starting_node, starting_node)]
    visited = [False] * N
    visited[starting_node] = True
    i = 0


    while True:
        for id in connections[queue[i][1]]:
            if not visited[id]:
                reference_id = queue[i][1]

                for id_ in connections[id]:
                    if id_ == queue[i][1]:
                        break
                    if visited[id_]:
                        reference_id = id_
                        break

                feats0_data = {
                "keypoints": torch.tensor(np.array([extracted_points[reference_id]], dtype=float), dtype=torch.float).to(device), 
                "descriptors": torch.tensor(np.array([extracted_descriptors[reference_id]], dtype=float), dtype=torch.float).to(device), 
                'image_size': torch.tensor(np.array([img_size_data[reference_id]], dtype=float), dtype=torch.float).to(device)
            }
                feats1_data = {
                "keypoints": torch.tensor(np.array([extracted_points[id]], dtype=float), dtype=torch.float).to(device), 
                "descriptors": torch.tensor(np.array([extracted_descriptors[id]], dtype=float), dtype=torch.float).to(device), 
                'image_size': torch.tensor(np.array([img_size_data[id]], dtype=float), dtype=torch.float).to(device)
            }

                matches01 = matcher({'image0': feats0_data, 'image1': feats1_data})

                feats0_data, feats1_data, matches01 = [rbd(x) for x in [feats0_data, feats1_data, matches01]]  # remove batch dimension
                kpts0, kpts1, matches = feats0_data['keypoints'], feats1_data['keypoints'], matches01['matches']
                idx0, idx1 = matches[..., 0].detach().cpu().numpy(), matches[..., 1].detach().cpu().numpy()

                interlaced_points = 0

                for p1, p2 in zip(idx0, idx1):
                    if not all_3d_points[reference_id]:
                        all_3d_points[reference_id] = [-1] * extracted_points[reference_id].shape[0]
                    if not all_3d_points[id]:
                        all_3d_points[id] = [-1] * extracted_points[id].shape[0]
                    if all_3d_points[reference_id][p1] == -1 and all_3d_points[id][p2] == -1:
                        continue
                    elif all_3d_points[reference_id][p1] != -1:
                        interlaced_points += 1
                    elif all_3d_points[id][p1] != -1:
                        interlaced_points += 1

                if len(idx0) >= 500 and (queue[i][1] == starting_node or (
                        queue[i][1] != starting_node and interlaced_points / len(idx0) >= 0.3)):
                    point3d_indexes = []
                    for p1, p2 in zip(idx0, idx1):
                        if all_3d_points[reference_id][p1] == -1 and all_3d_points[id][p2] == -1:
                            all_3d_points[reference_id][p1] = point3d_index
                            all_3d_points[id][p2] = point3d_index
                            point3d_index += 1
                        elif all_3d_points[reference_id][p1] != -1:
                            all_3d_points[id][p2] = all_3d_points[reference_id][p1]
                        elif all_3d_points[id][p1] != -1:
                            all_3d_points[reference_id][p2] = all_3d_points[id][p1]

                        point3d_indexes.append(all_3d_points[reference_id][p1])

                    all_matches.append([idx0, idx1, np.array(point3d_indexes)])
                    queue.append((reference_id, id))
                    visited[id] = True
                else:
                    continue

        i += 1
        if i >= len(queue):
            break

    print(queue, len(queue))
    np.save(os.path.join(cache_dir, 'visual_pairs.npy'), queue[1:])
    np.save(os.path.join(cache_dir, 'point_pairings.npy'), np.array(all_matches, dtype=object))

    return type(extracted_descriptors), f"Loaded descriptors: {extracted_descriptors.shape}"


if __name__ == "__main__":
    cache_dir = "./dataset03/cache"  # Specify the manual cache directory here
    generate_point_cloud(cache_dir, 0.75)