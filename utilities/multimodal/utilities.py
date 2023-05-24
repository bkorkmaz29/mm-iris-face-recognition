import numpy as np
import json


def get_weighted_scores(scores, weight):
    weighted_scores = []
    for score in scores:
        weighted_scores.append(score * weight)
    return weighted_scores


def get_normalized_scores(scores):
    norm_scores = []
    denominator = (np.max(scores) - np.min(scores))
    for score in scores:
        if denominator != 0:
            norm_score = (score - np.min(scores)) / denominator
        else:
            norm_score = 0

        norm_scores.append(norm_score)

    return norm_scores


def get_normalized_score(score):
    denominator = (np.max(score) - np.min(score))

    if denominator != 0:
        norm_score = (score - np.min(score)) / denominator
    else:
        norm_score = 0

    return norm_score


def json_write(dir, info):
    with open(dir, "w") as json_file:
        json.dump(info, json_file)


def is_valid_path(filepath):
    if filepath and Path(filepath).exists():
        return True

    return False
