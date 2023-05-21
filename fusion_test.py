import os
from functions.face.face_extraction import get_face_encodings
from functions.face.face_matching import face_distance
import numpy as np
from glob import glob
from random import shuffle
from functions.iris.iris_matching import iris_distance
from functions.iris.iris_extraction import iris_extract
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import PIL.Image
import cv2
yale_dir = "yale"
ft_dir = "data/ft"
N_IMAGES = 2

thresholds = np.linspace(start=0.0, stop=1.0, num=100)

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
        filesF = glob(os.path.join(yale_dir, identity + ".centerlight"))
        filesI = glob(os.path.join(ft_dir, identity, ".bmp"))
        files = {face_files: filesF, iris_files: filesI}
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
    '''      
    for i in range(n_image_files):
        for j in range(n_image_files):
            if i < j:
                print(i)
                dists.append(face_distance(i, j))
                dist_mat[i, j] = dists[k]
                k += 1
            elif i > j:
                dist_mat[i, j] = dist_mat[j, i]
    
    accuracies, precisions, recalls, fscores, frrs, fars = [], [], [], [], [], []

    for threshold in thresholds:
        decision_map = (dist_mat <= threshold).astype(int)
        accuracy = (decision_map == ground_truth).sum() / ground_truth.size
        precision = (ground_truth*decision_map).sum() / decision_map.sum()
        far = 1 - precision
        recall = (ground_truth*decision_map).sum() / ground_truth.sum()
        frr = 1 - recall
        fscore = 2*precision*recall / (precision+recall)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)
        frrs.append(fscore)
        fars.append(fscore)

        
   
    best_acc = max(accuracies)
    best_fscore = max(fscores)
    min_recall = min(recalls)
    min_pre = min(precisions)
    best_threshold = thresholds[fscores.index(best_fscore)]
    best_threshold_acc = thresholds[accuracies.index(best_acc)]

    print("Maximum acc:", max(accuracies), best_threshold_acc)
    print("Maximum pre:", max(precisions))
    print("Maximum fscore:", best_fscore, "threshold:", best_threshold)
    import matplotlib.pyplot as plt

    plt.plot(accuracies, thresholds)
    plt.xlabel('accuracy')
    plt.ylabel('threshold')
    plt.title('Accuracy - Threshold')
    plt.show()

    plt.plot(fscores, thresholds)
    plt.xlabel('F1')
    plt.ylabel('threshold')
    plt.title('F1 - Threshold')
    plt.show()

    plt.plot(fars, thresholds)
    plt.xlabel('FAR')
    plt.ylabel('threshold')
    plt.title('FAR - Threshold')
    plt.show()

    plt.plot(frrs, thresholds)
    plt.xlabel('FRR')
    plt.ylabel('threshold')
    plt.title('FRR - Threshold')
    plt.show()
        '''
    decision_map = (dist_mat <= 0.5).astype(int)
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
