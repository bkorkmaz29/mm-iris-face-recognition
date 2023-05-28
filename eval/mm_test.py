import os
import numpy as np
from glob import glob
from random import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import PIL.Image
import cv2
dir = "data/mm"
N_IMAGES = 7
thresholds = np.linspace(start=0, stop=1, num=100)

if __name__ == '__main__':

    files = glob(os.path.join(dir, "*"))

    for i in range(len(files)):
        files[i] = files[i][8:]
    identities = set(files)
    n_identities = len(identities)

 # Construct a dictionary of files
    files_dict = {}
    face_files_orig = []
    face_images = []
    iris_images = []
    iris_files_orig = []
# Iterate over each identity
    for identity in identities:
        # Retrieve face files for the current identity
        face_files = glob(os.path.join(
            dir, "*/subject" + identity + ".*"))
        shuffle(face_files)

        # Retrieve iris files for the current identity
        iris_files = glob(os.path.join(
            dir, identity + "/" + "0**/*/*_*_*.bmp"))

        shuffle(iris_files)

        # Select a subset of face and iris files based on N_IMAGES
        selected_face_files = face_files[:N_IMAGES]
        selected_iris_files = iris_files[:N_IMAGES]

        # Store the selected face and iris files in the dictionary
        files_dict[identity] = {
            'face': selected_face_files, 'iris': selected_iris_files}

        # Concatenate the selected face and iris files to the overall lists

        face_images += selected_face_files
        iris_images += selected_iris_files

    n_image_files = len(face_images)
    print(n_image_files)
    # Ground truth
    ground_truth = np.zeros([n_image_files, n_image_files], dtype=int)
    for i in range(ground_truth.shape[0]):
        for j in range(ground_truth.shape[1]):
            if i//N_IMAGES == j//N_IMAGES:
                ground_truth[i, j] = 1

    # Evaluate parameters
    best_results = []
    face_features, iris_features = [], []
    for face_image in face_images:
        im = PIL.Image.open(face_image)
        im2 = im.convert('RGB')
        image = np.array(im2)
        print(face_image)
        face_features.append(get_face_encodings(image))

    for iris_image in iris_images:
        print(iris_image)
        iris_features.append(iris_extract(iris_image))

    args = []
    dists = []
    for i in range(n_image_files):
        for j in range(n_image_files):
            if i >= j:
                continue
            arg = (face_features[i], iris_features[i][0], iris_features[i][1],
                   face_features[j],  iris_features[j][0], iris_features[j][1])

            args.append(arg)
            # print(args)

    for arg in args:
        face1, temp1, mask1, face2, temp2, mask2 = arg

        dists.append(weighted_product(
            face1, temp1, mask1, face2, temp2, mask2, 0.515))

    print(dists)

    k = 0
    dist_mat = np.zeros([n_image_files, n_image_files])
    for i in range(n_image_files):
        for j in range(n_image_files):
            if i < j:
                dist_mat[i, j] = dists[k]
                k += 1
            elif i > j:
                dist_mat[i, j] = dist_mat[j, i]

    #dist_mat = np.load("mm_dist.npy")
    #np.save("prod_dist", dist_mat)

    accuracies, precisions, recalls, fscores, frrs, fars, f05s, f2s = [
    ], [], [], [], [], [], [], []

    for threshold in thresholds:
        decision_map = (dist_mat <= threshold).astype(int)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            ground_truth, decision_map, average='macro')
        accuracy = (decision_map == ground_truth).sum() / ground_truth.size
        f05 = (1 + 0.5**2) * (precision * recall) / \
            (0.5**2 * precision + recall)
        f2 = (1 + 2**2) * (precision * recall) / (2**2 * precision + recall)
        frr = 1 - recall
        far = 1 - precision
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(f1_score)
        frrs.append(frr)
        fars.append(far)
        f05s.append(f05)
        f2s.append(f2)

    best_acc = max(accuracies)
    best_f05 = max(f05s)
    best_fscore = max(fscores)
    best_f2 = max(f2s)
    best_pre = max(precisions)
    best_threshold = thresholds[fscores.index(best_fscore)]
    best_threshold_pre = thresholds[precisions.index(best_pre)]
    best_threshold_f05 = thresholds[f05s.index(best_f05)]
    best_threshold_f2 = thresholds[f2s.index(best_f2)]
    best_threshold_acc = thresholds[accuracies.index(best_acc)]

    print("Maximum fscore:", best_fscore, "threshold:", best_threshold)
    print("Maximum f05score:", best_f05, "threshold:", best_threshold_f05)
    print("Maximum f2score:", best_f2, "threshold:", best_threshold_f2)
    print("Maximum acc:", best_acc, "threshold:", best_threshold_acc)
    print("Maximum pre:", best_pre, "threshold:", best_threshold_pre)

    plt.plot(thresholds, fscores)
    plt.ylabel('F1 Score')
    plt.xlabel('Threshold')
    plt.title('F1 - Threshold')
    plt.legend()
    plt.show()

    plt.plot(thresholds, accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('Threshold')
    plt.title('Accuracy - Threshold')
    plt.show()

    plt.plot(thresholds, fars, label='FAR')
    plt.plot(thresholds, frrs, label='FRR')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('FAR and FRR vs. Threshold')
    plt.show()
