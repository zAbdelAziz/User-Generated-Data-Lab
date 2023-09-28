import pandas as pd
import numpy as np
import random as rnd
from typing import Callable

import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity


def inter_matr_implicit(users: pd.DataFrame,
                        items: pd.DataFrame,
                        interactions: pd.DataFrame,
                        dataset_name: str,
                        threshold=1) -> np.ndarray:
    '''
    users - pandas Dataframe, use it as loaded from the dataset;
    items - pandas Dataframe, use it as loaded from the dataset;
    interactions - pandas Dataframe, use it as loaded from the dataset;
    dataset_name - string out of ["lfm-ismir", "lfm-tiny", "ml-1m"], name of the dataset, used in case there are differences in the column names of the data frames;
    threshold - int > 0, criteria of a valid interaction

    returns - 2D np.array, rows - users, columns - items;
    '''

    res = None

    # TODO: YOUR IMPLEMENTATION

    interactions = interactions.copy()

    # getting number of users and items from the respective files to be on the safe side
    n_users = len(users.index)
    n_items = len(items.index)

    # preparing the output matrix
    res = np.zeros([n_users, n_items], dtype=np.int8)

    # for every interaction assign 1 to the respective element of the matrix
    if dataset_name == 'lfm-ismir':
        inter_column_name = 'listening_events'
    elif dataset_name == 'ml-1m':
        inter_column_name = 'rating'
    elif dataset_name == 'lfm-tiny':
        inter_column_name = 'listening_events'
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name} ")

    row = interactions["user_id"].to_numpy()
    col = interactions["item_id"].to_numpy()

    data = interactions[inter_column_name].to_numpy()
    data[data < threshold] = 0
    data[data >= threshold] = 1

    res[row, col] = data

    return res


def recTopKPop(inter_matr: np.array,
               user: int,
               top_k: int) -> np.array:
    '''
    inter_matr - np.array from the task 1;
    user - user_id, integer;
    top_k - expected length of the resulting list;

    returns - list/array of top K popular items that the user has never seen
              (sorted in the order of descending popularity);
    '''

    top_pop = None

    # TODO: YOUR IMPLEMENTATION

    # global item-popularity distribution:
    item_pop = inter_matr.sum(axis=0)

    # finding items seen by the user, basicaly indices of non-zero elements ...
    # ... in the interaction array corresponding to the user:
    items_seen = np.nonzero(inter_matr[user])

    # force values seen by the user to become 'unpopular'
    item_pop[items_seen] = 0

    top_pop = np.full((top_k,), -1)

    # get indices of top_K (new) popular items
    t_pop = (-item_pop).argsort()[:top_k]
    top_pop[:len(t_pop)] = t_pop

    return top_pop


def svd_decompose(inter_matr: np.ndarray, f=50) -> (np.ndarray, np.ndarray):
    """
    inter_matr - np.ndarray - interaction matrix to construct svd from.
    f - int - expected size of embeddings

    returns - U_final, V_final - (as above) user-/item-embeddings of given length f
    """

    U_final = None
    V_final = None

    # TODO: YOUR IMPLEMENATION.

    U, s, Vh = np.linalg.svd(inter_matr, full_matrices=False)
    U_final = U[:, :f] @ np.diag(s[:f] ** 0.5)  # users x features
    V_final = (np.diag(s[:f] ** 0.5) @ Vh[:f, :]).T  # items x features

    return U_final, V_final


def svd_recommend_to_list(user_id: int, seen_item_ids: list, U: np.ndarray, V: np.ndarray, topK: int) -> np.ndarray:
    """
    Recommend with svd to selected users

    user_id - int - id of target user.
    seen_item_ids - list[int] ids of items already seen by the users (to exclude from recommendation)
    U and V - user- and item-embeddings
    topK - number of recommendations per user to be returned

    returns - np.ndarray - list of ids of recommended items in the order of descending score
                           use -1 as a place holder item index, when it is impossible to recommend topK items
    """
    recs = None

    scores = U @ V.T
    u_scores = scores[user_id]
    u_scores[seen_item_ids] = -np.inf
    m = min(topK, scores.shape[1])
    recs = (-u_scores).argsort()[:m]

    return np.array(recs)


def jaccard_score(a: np.array, b: np.array) -> float:
    """
    a, b: - vectors of the same length corresponding to the two items

    returns: float - jaccard similarity score for a and b
    """
    score = None

    # TODO: YOUR IMPLEMENTATION

    # form union over a and b by summing both vectors
    c = a + b
    # value in union > 1 means user was present in both item vectors -> intersection
    intersection = np.zeros_like(c)
    intersection[c > 1] = 1
    # set union vector values > 1 to 1 for counting by summing up
    union = np.zeros_like(c)
    union[c >= 1] = 1

    score = np.sum(intersection) / np.sum(union)

    return float(score)


def calculate_sim_scores(similarity_measure: Callable[[int, int], float], inter: np.array,
                         target_vec: np.array) -> np.array:
    """
    inter: np.array - interaction matrix - calculate similarity between each item and the target item (see below)
    target_vec: int - target item vector

    use my_jaccard_similarity function from above

    returns: np.array - similarities between every item from <inter> and <target_vec> in the respective order
    """

    item_similarities = None

    # TODO: YOUR IMPLEMENTATION

    item_similarities = np.zeros((inter.shape[1],))

    # calculate jaccard similarity of every item.
    for item in range(inter.shape[1]):
        inter_items = inter[:, item]
        item_similarities[item] = similarity_measure(inter_items, target_vec)

    return np.array(item_similarities)


def get_user_item_score(sim_scores_calculator: Callable[[Callable, np.array, np.array], np.array],
                        inter: np.array,
                        target_user: int,
                        target_item: int,
                        n: int = 2) -> float:
    """
    inter: np.array - interaction matrix.
    target_user: target user id
    target_item: int - target item id
    n: int - n closest neighbors to consider for the score prediction

    returns: float - mean of similarity scores
    """

    item_similarities_mean = None

    # TODO: YOUR IMPLEMENTATION.

    inter_pred = inter.copy()

    # Get all items which were consumed by the user.
    item_consumed_by_user = inter_pred[target_user, :] == 1
    item_consumed_by_user[target_item] = False

    # get column of the target_item.
    inter_target_item = inter_pred[:, target_item]

    # create a mask to remove the user from the interaction matrix.
    not_user = np.full((inter_pred.shape[0],), True)
    not_user[target_user] = False

    # remove items not interacted with user
    inter_pred = inter_pred[:, item_consumed_by_user]

    # remove user
    inter_pred = inter_pred[not_user]
    inter_target_item = inter_target_item[not_user]

    # get closest items to target_item, which is at the last indices.
    scores = sim_scores_calculator(inter_pred, inter_target_item)

    # get items with the highes scores.
    scores_ids = np.argsort((- scores))
    scores = scores[scores_ids]

    scores = scores[:n]

    if len(scores) > 0:
        # calculate mean of normed scores.
        item_similarities_mean = scores.mean()
    else:
        item_similarities_mean = 0.0

    return item_similarities_mean


def sim_score_calc(inter, target_vec): return calculate_sim_scores(jaccard_score, inter, target_vec)


def user_item_scorer(inter, target_user, target_item, n): return get_user_item_score(sim_score_calc, inter,
                                                                                     target_user, target_item, n)


def _recTopK_base(user_item_scorer: Callable[[Callable, np.array, int, int], float],
                  inter_matr: np.array,
                  user: int,
                  top_k: int,
                  n: int) -> (np.array, np.array):
    '''
    inter_matr - np.array from the task 1
    user - user_id, integer
    top_k - expected length of the resulting list
    n - number of neighbors to consider

    returns - array of recommendations (sorted in the order of descending scores) & array of corresponding scores
    '''

    top_rec = None
    scores = None

    # TODO: YOUR IMPLEMENTATION.

    scores = np.zeros((inter_matr.shape[1],))

    for item in range(inter_matr.shape[1]):
        if inter_matr[user, item] == 0:
            score = user_item_scorer(inter_matr, user, item, n)
            scores[item] = score

    top_rec = (- scores).argsort()[:top_k]
    scores = scores[top_rec]

    return np.array(top_rec), np.array(scores)


def recTopK(inter_matr: np.array,
            user: int,
            top_k: int,
            n: int) -> (np.array, np.array):
    return _recTopK_base(user_item_scorer, inter_matr, user, top_k, n)[0]
