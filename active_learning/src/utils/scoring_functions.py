"""
Function definitions for the different acquisition/scoring functions in active learning
"""
import typing as tp

import torch
from torch import nn

SupportedScoringFunc = tp.Literal[
    "entropy", "mutual_info", "least_confidence", "margin_sampling", "variation_ratios"
]


@torch.no_grad()
def score_batch(
    preds: torch.Tensor, scoring_functions: tp.List[SupportedScoringFunc]
) -> torch.Tensor:
    """
    Parameters
    ----------
    preds : torch.Tensor
        Size is (nb_models, batch_size, nb_classes)
    scoring_functions : tp.List[SupportedScoringFunc]
        List of scoring functions to use on the batch

    Returns
    -------
    torch.Tensor
        Size is (batch_size, nb_scoring_functions). The order of the columns follows
        the order of the given list of scoring functions.
    """

    scores_all = []
    for scoring_func in scoring_functions:
        if scoring_func == "entropy":
            scores = entropy(preds)
        elif scoring_func == "mutual_info":
            scores = mutual_information(preds)
        elif scoring_func == "mutual_info":
            scores = mutual_information(preds)
        elif scoring_func == "least_confidence":
            scores = least_confidence(preds)
        elif scoring_func == "margin_sampling":
            scores = margin_sampling(preds)
        elif scoring_func == "variation_ratios":
            scores = variation_ratios(preds)
        else:
            raise ValueError(f"{scoring_func} is not a supported scoring function")

        scores_all.append(scores)

    scores_all = torch.stack(scores_all, dim=1)

    return scores_all


@torch.no_grad()
def entropy(preds: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    preds : torch.Tensor
        Size is (nb_models, batch_size, nb_classes) or (batch_size, nb_classes)

    Returns
    -------
    torch.Tensor
        Size is (batch_size)
    """

    # 1: Average predicted probablities across models in the ensemble
    if preds.dim() == 3:
        preds_avg = preds.mean(dim=0)
    elif preds.dim() == 2:
        preds_avg = preds
    else:
        raise ValueError(
            f"Given prediction tensor has {preds.dim()} dimentions, but expects 2 or 3"
        )

    # 2: Compute entropy. If a prediction is exactly zero, the log will give -inf and
    # the entropy of the corresponding vector of predictions is nan. We avoid that by
    # replacing -inf with zero
    neg_log_preds = -torch.log2(preds_avg)
    neg_log_preds[neg_log_preds.isinf()] = 0
    entropies = torch.diag(torch.matmul(preds_avg, neg_log_preds.transpose(0, 1)))

    return entropies


@torch.no_grad()
def mutual_information(preds: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    preds : torch.Tensor
        Size is (nb_models, batch_size, nb_classes), with nb_models > 1

    Returns
    -------
    torch.Tensor
        Size is (batch_size)
    """
    if preds.dim() != 3 and not preds.shape[0] > 1:
        raise ValueError(f"Mutual information can only be computed with an ensemble")

    nb_models = preds.shape[0]

    entropy_of_avg_prediction = entropy(preds)
    entropies_of_single_predictions = torch.stack(
        [entropy(preds[e : e + 1]) for e in range(nb_models)]
    )
    avg_entropy = entropies_of_single_predictions.mean(dim=0)
    mutual_infos = entropy_of_avg_prediction - avg_entropy

    return mutual_infos


@torch.no_grad()
def least_confidence(preds: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    preds : torch.Tensor
        Size is (nb_models, batch_size, nb_classes) or (batch_size, nb_classes)

    Returns
    -------
    torch.Tensor
        Size is (batch_size)
    """

    # 1: Average predicted probablities across models in the ensemble
    if preds.dim() == 3:
        preds_avg = preds.mean(dim=0)
    elif preds.dim() == 2:
        preds_avg = preds
    else:
        raise ValueError(
            f"Given prediction tensor has {preds.dim()} dimentions, but expects 2 or 3"
        )

    # 2: Compute scores
    scores = 1 - torch.max(preds_avg, dim=1)[0]

    return scores


@torch.no_grad()
def margin_sampling(preds: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    preds : torch.Tensor
        Size is (nb_models, batch_size, nb_classes) or (batch_size, nb_classes)

    Returns
    -------
    torch.Tensor
        Size is (batch_size)
    """

    # 1: Average predicted probablities across models in the ensemble
    if preds.dim() == 3:
        preds_avg = preds.mean(dim=0)
    elif preds.dim() == 2:
        preds_avg = preds
    else:
        raise ValueError(
            f"Given prediction tensor has {preds.dim()} dimentions, but expects 2 or 3"
        )

    # 2: Compute scores
    top2 = torch.topk(preds_avg, k=2, dim=1)[0]
    scores = 1 - (top2[:, 0] - top2[:, 1])

    return scores


@torch.no_grad()
def variation_ratios(preds: torch.Tensor):
    """Compute the fraction of models that do not agree with the majority vote. This
    score can only take a finite number of values, that depends on the number of models
    in the ensemble. Thus it is not expected to be a good scoring functions with small
    ensembles.

    Parameters
    ----------
    preds : torch.Tensor
        Size is (nb_models, batch_size, nb_classes), with nb_models > 1

    Returns
    -------
    torch.Tensor
        Size is (batch_size)
    """

    if preds.dim() != 3 and not preds.shape[0] > 1:
        raise ValueError(f"Variation ratios can only be computed with an ensemble")

    nb_models = preds.shape[0]

    votes = preds.argmax(dim=2)
    majority_vote = torch.mode(votes, dim=0)[0]
    nb_in_agreement = torch.stack(
        [votes[e] == majority_vote for e in range(nb_models)]
    ).sum(dim=0)
    var_ratios = 1 - nb_in_agreement / nb_models

    return var_ratios
