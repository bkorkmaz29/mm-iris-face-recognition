import matplotlib.pyplot as plt
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from random import shuffle
from itertools import repeat
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functions.iris.iris_matching import iris_distance
from functions.iris.iris_extraction import iris_extract
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
CASIA1_DIR = "data/CASIA1"
N_IMAGES = 7

eyelashes_thresholds = np.linspace(start=10, stop=250, num=25)
thresholds = np.linspace(start=0.0, stop=1.0, num=100)

if __name__ == '__main__':

    identities = glob(os.path.join(CASIA1_DIR, "**"))
    identities = sorted([os.path.basename(identity)
                        for identity in identities])
    n_identities = len(identities)

    print("Number of identities:", n_identities)

    # Construct a dictionary of files
    files_dict = {}
    image_files = []
    for identity in identities:
        files = glob(os.path.join(CASIA1_DIR, identity, "*/*_*_*.bmp"))
        shuffle(files)
        files_dict[identity] = files[:N_IMAGES]
        image_files += files[:N_IMAGES]

    n_image_files = len(image_files)
    print("Number of image files:", n_image_files)

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
        print(image_file)
        features.append(iris_extract(image_file))
    # Calculate the distances
    args = []
    dists = []
    for i in range(n_image_files):
        for j in range(n_image_files):
            if i >= j:
                continue
            arg = (features[i][0], features[i][1],
                   features[j][0], features[j][1])
            args.append(arg)

    for arg in args:
        mask, temp, temp2, mask2, = arg
        dists.append(iris_distance(mask, temp, temp2, mask2))

    # Construct a distance matrix
    k = 0
    dist_mat = np.zeros([n_image_files, n_image_files])
    for i in range(n_image_files):
        for j in range(n_image_files):
            if i < j:
                dist_mat[i, j] = dists[k]
                k += 1
            elif i > j:
                dist_mat[i, j] = dist_mat[j, i]
    np.save("db", dist_mat)
    decision_map = (dist_mat <= 0.414).astype(int)
    accuracy = (decision_map == ground_truth).sum() / ground_truth.size
    precision = (ground_truth*decision_map).sum() / decision_map.sum()
    recall = (ground_truth*decision_map).sum() / ground_truth.sum()
    fscore = 2*precision*recall / (precision+recall)
    frr = 1 - recall
    far = 1 - precision
    print("accuracy", accuracy)
    print("f1_score", fscore)
    print("FAR", 1 - precision)
    print("FRR", 1 - recall)

    decision_map = (dist_mat <= 0.401).astype(int)
    accuracy = (decision_map == ground_truth).sum() / ground_truth.size
    precision = (ground_truth*decision_map).sum() / decision_map.sum()
    recall = (ground_truth*decision_map).sum() / ground_truth.sum()
    fscore = 2*precision*recall / (precision+recall)
    frr = 1 - recall
    far = 1 - precision
    print("accuracy", accuracy)
    print("f1_score", fscore)
    print("FAR", 1 - precision)
    print("FRR", 1 - recall)

    decision_map = (dist_mat <= 0.394).astype(int)
    accuracy = (decision_map == ground_truth).sum() / ground_truth.size
    precision = (ground_truth*decision_map).sum() / decision_map.sum()
    recall = (ground_truth*decision_map).sum() / ground_truth.sum()
    fscore = 2*precision*recall / (precision+recall)
    frr = 1 - recall
    far = 1 - precision
    print("accuracy", accuracy)
    print("f1_score", fscore)
    print("FAR", 1 - precision)
    print("FRR", 1 - recall)

    '''

    accuracies, precisions, recalls, fscores = [], [], [], []
    # For finding the best threshold
    far_values = []  # FAR values
    frr_values = []  # FRR values
    for threshold in thresholds:
        decision_map = (dist_mat <= threshold).astype(int)
        accuracy = (decision_map == ground_truth).sum() / ground_truth.size
        precision = (ground_truth*decision_map).sum() / decision_map.sum()
        recall = (ground_truth*decision_map).sum() / ground_truth.sum()
        fscore = 2*precision*recall / (precision+recall)
        frr = 1 - recall
        far = 1 - precision
        far_values.append(far)
        frr_values.append(frr)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)

    best_frr = np.max(frr_values)
    best_far = np.max(far_values)

    best_acc = max(accuracies)
    best_rec = max(recalls)
    best_fscore = max(fscores)
    min_recall = min(recalls)
    max_pre = max(precisions)

    best_threshold = thresholds[fscores.index(best_fscore)]
    best_threshold_rec = thresholds[recalls.index(best_rec)]
    best_threshold_pre = thresholds[precisions.index(max_pre)]
    best_threshold_acc = thresholds[accuracies.index(best_acc)]
    best_threshold_far = thresholds[np.argmin(far_values)]
    best_threshold_frr = thresholds[np.argmin(frr_values)]

    print("Maximum rec:", max(recalls), best_threshold_rec)
    print("Maximum pre:", max(precisions),  best_threshold_pre)
    print("Minimum far:", min(far_values), best_threshold_far)
    print("Minimum frr:", min(frr_values),  best_threshold_frr)

    print("Maximum accuracy:", best_acc, "threshold:", best_threshold_acc)
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Threshold')
    plt.legend()
    plt.show()

    print("Maximum fscore:", best_fscore, "threshold:", best_threshold)
    plt.plot(thresholds, fscores, label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold')
    plt.legend()
    plt.show()

    plt.plot(thresholds, far_values, label='FAR')
    plt.plot(thresholds, frr_values, label='FRR')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('FAR and FRR vs. Threshold')
    plt.legend()
    plt.show()
    decision_map = (dist_mat <= 0.4).astype(int)
    conf_mat = multilabel_confusion_matrix(ground_truth, decision_map)

    # calculate the false positive, false negative, true positive, and true negative rates
    fp = conf_mat[:, 1, 0] / (conf_mat[:, 1, 0] + conf_mat[:, 1, 1])
    fn = conf_mat[:, 0, 1] / (conf_mat[:, 0, 0] + conf_mat[:, 0, 1])
    tp = conf_mat[:, 1, 1] / (conf_mat[:, 1, 1] + conf_mat[:, 0, 1])
    tn = conf_mat[:, 0, 0] / (conf_mat[:, 0, 0] + conf_mat[:, 1, 0])

    # calculate the false acceptance rate (FAR) and false rejection rate (FRR)
    far = fp.mean()
    frr = fn.mean()

    # calculate precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        ground_truth, decision_map, average='macro')

    print("FAR:", far)
    print("FRR:", frr)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1_score)
 '''
