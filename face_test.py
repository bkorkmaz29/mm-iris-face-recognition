import os
from functions.face.face_extraction import get_face_encodings
from functions.face.face_matching import face_distance
import numpy as np
from glob import glob
from random import shuffle
from functions.iris.iris_matching import iris_distance
from functions.iris.iris_extraction import iris_extract
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import PIL.Image
import cv2
yale_dir = "yale"
N_IMAGES = 11

thresholds = np.linspace(start=0.1, stop=1.0, num=100)

if __name__ == '__main__':
    files = glob(os.path.join(yale_dir, "*"))
    for i in range(len(files)):
        files[i] = files[i][:14][5:]
    identities = set(files)
    n_identities = len(identities)

 # Construct a dictionary of files
    files_dict = {}
    image_files = []
    for identity in identities:
        files = glob(os.path.join(yale_dir, identity + ".*"))
        shuffle(files)
        files_dict[identity] = files[:N_IMAGES]
        image_files += files[:N_IMAGES]
    print(files_dict)
    n_image_files = len(image_files)
    print("Number of images:", n_image_files)

    # Ground truth
    ground_truth = np.zeros([n_image_files, n_image_files], dtype=int)
    for i in range(ground_truth.shape[0]):
        for j in range(ground_truth.shape[1]):
            if i//N_IMAGES == j//N_IMAGES:
                ground_truth[i, j] = 1

    # Evaluate parameters
    best_results = []
    features = []
    for image_file in image_files:
        im = PIL.Image.open(image_file)
        im2 = im.convert('RGB')
        image = np.array(im2)
        print(image_file)
        features.append(get_face_encodings(image))

    args = []
    dists = []
    for i in range(n_image_files):
        for j in range(n_image_files):
            if i >= j:
                continue
            arg = (features[i], features[j])

            args.append(arg)

    for arg in args:
        feature1, feature2 = arg
        dists.append(face_distance(feature1, np.array(feature2)))

    k = 0
    dist_mat = np.zeros([n_image_files, n_image_files])
    for i, feature1 in enumerate(features):
        for j, feature2 in enumerate(features):
            if i < j:
                dist_mat[i, j] = dists[k]
                k += 1
            elif i > j:
                dist_mat[i, j] = dist_mat[j, i]

    accuracies, precisions, recalls, fscores, frrs, fars = [], [], [], [], [], []
    '''
    for threshold in thresholds:
        decision_map = (dist_mat <= threshold).astype(int)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            ground_truth, decision_map, average='macro')
        accuracy = (decision_map == ground_truth).sum() / ground_truth.size
        frr = 1 - recall
        far = 1 - precision
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(f1_score)
        frrs.append(frr)
        fars.append(far)

    best_acc = max(accuracies)
    best_fscore = max(fscores)
    min_recall = min(recalls)

    best_threshold = thresholds[fscores.index(best_fscore)]
    best_threshold_acc = thresholds[accuracies.index(best_acc)]

    print("Maximum fscore:", best_fscore, "threshold:", best_threshold)
    print("Maximum acc:", best_acc, "threshold:", best_threshold_acc)

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
    plt.legend()
    plt.show()

    plt.plot(thresholds, fars, label='FAR')
    plt.plot(thresholds, frrs, label='FRR')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('FAR and FRR vs. Threshold')
    plt.legend()
    plt.show()
    '''
    decision_map = (dist_mat <= 0.482).astype(int)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        ground_truth, decision_map, average='macro')
    accuracy = (decision_map == ground_truth).sum() / ground_truth.size
    frr = 1 - recall
    far = 1 - precision

    print("accuracy", accuracy)
    print("f1_score", f1_score)
    print("FAR", 1 - precision)
    print("FRR", 1 - recall)

    '''
    
    for i in range(n_image_files):
        for j in range(n_image_files):
            if i < j:
                print(i)
                dists.append(face_distance(i, np.array(j)))
                dist_mat[i, j] = dists[k]
                k += 1
            elif i > j:
                dist_mat[i, j] = dist_mat[j, i]

    precisions, recalls, f1_scores, far_values, frr_values = [], [], [], [], []

    for threshold in thresholds:
        decision_map = (dist_mat <= threshold).astype(int)
        conf_mat = multilabel_confusion_matrix(ground_truth, decision_map)

        # calculate precision, recall, and F1-score
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            ground_truth, decision_map, average='macro')
        frr = 1 - recall
        far = 1 - precision
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        frr_values.append(frr)
        far_values.append(far)
    print(far_values)
    best_pre = np.max(precisions)
    best_fscore = np.max(f1_scores)
    best_frr = np.max(frr_values)
    best_far = np.max(far_values)
    min_recall = np.min(recalls)
    min_pre = np.min(precisions)
    best_threshold = thresholds[f1_scores.index(best_fscore)]
    best_threshold_far = thresholds[np.argmin(far_values)]
    best_threshold_frr = thresholds[np.argmin(frr_values)]
    print(best_threshold)
    print(best_threshold_far)
    print(best_threshold_frr)
    plt.scatter(best_threshold, best_fscore, color='red', label='Overlap')
    plt.annotate('Overlap', (best_threshold, best_fscore),
                 xytext=(5, 5), textcoords='offset points')

    plt.plot(thresholds, far_values, label='FAR')
    plt.plot(thresholds, frr_values, label='FRR')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('FAR and FRR vs. Threshold')
    plt.legend()
    plt.show()


    '''
