import matplotlib.pyplot as plt
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from random import shuffle
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

CASIA1_DIR = "data/CASIA12"
N_IMAGES = 7

#eyelashes_thresholds = np.linspace(start=10, stop=250, num=25)
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
    '''
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
    print("Maximum accuracy:", best_acc, "threshold:", best_threshold_acc)
    '''
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
