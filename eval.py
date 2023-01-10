 eyelashes_thresholds = np.linspace(start=10, stop=250, num=25)
    thresholds = np.linspace(start=0.0, stop=1.0, num=100)
    
        #------------------------------------------------------------------------------
    #	Main execution
    #------------------------------------------------------------------------------
    # Get identities of MMU2 dataset
    identities = glob.glob(os.path.join(data_dir, "**"))
    identities = sorted([os.path.basename(identity) for identity in identities])
    n_identities = len(identities)
    print("Number of identities:", n_identities)


    # Construct a dictionary of files
    files_dict = {}
    image_files = []
    for identity in identities:
        files = glob.glob(os.path.join(data_dir, identity, "2/*.bmp"))
        random.shuffle(files)
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
    pools = Pool(processes=cpu_count())
    best_results = []
    for eye_threshold in tqdm(eyelashes_thresholds, total=len(eyelashes_thresholds)):
        # Extract features
        args = zip(image_files, repeat(eye_threshold), repeat(False))
        features = list(pools.map(pool_extract_feature, args))

        # Calculate the distances
        args = []
        for i in range(n_image_files):
            for j in range(n_image_files):
                if i>=j:
                    continue
                arg = (features[i][0], features[i][1], features[j][0], features[j][1])
                args.append(arg)
        distances = pools.map(pool_calHammingDist, args)

        # Construct a distance matrix
        k = 0
        dist_mat = np.zeros([n_image_files, n_image_files])
        for i in range(n_image_files):
            for j in range(n_image_files):
                if i<j:
                    dist_mat[i, j] = distances[k]
                    k += 1
                elif i>j:
                    dist_mat[i, j] = dist_mat[j, i]

        # Metrics
        accuracies, precisions, recalls, fscores = [], [], [], []
        for threshold in thresholds:
            decision_map = (dist_mat<=threshold).astype(int)
            accuracy = (decision_map==ground_truth).sum() / ground_truth.size
            precision = (ground_truth*decision_map).sum() / decision_map.sum()
            recall = (ground_truth*decision_map).sum() / ground_truth.sum()
            fscore = 2*precision*recall / (precision+recall)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            fscores.append(fscore)

        # Save the best result
        best_fscore = max(fscores)
        best_threshold = thresholds[fscores.index(best_fscore)]
        best_results.append((eye_threshold, best_threshold, best_fscore))

    # Show the final best result
    eye_thresholds = [item[0] for item in best_results]
    thresholds = [item[1] for item in best_results]
    fscores = [item[2] for item in best_results]

    print("Maximum fscore:", max(fscores))
    print("Best eye_threshold:", eye_thresholds[fscores.index(max(fscores))])
    print("Best threshold:", thresholds[fscores.index(max(fscores))])
        