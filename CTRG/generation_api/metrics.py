from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.meteor import Meteor
from .pycocoevalcap.rouge import Rouge
from .CaptionMetrics.pycocoevalcap.cider import Cider


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        #(Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDER")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            # print(method)
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score

        # print(score)
    return eval_res

def compute_scores_2(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDER")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            # print(method)
            score, scores = scorer.compute_score(gts, res)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score

        # print(score)
    # print(eval_res)
    # # eval_res["METEOR"] = Meteor(gts, res)
    print(eval_res)
    # print(gts, res)
    # exit(0)
    return eval_res

if __name__ == "__main__":
    pred = ["this is a test code."]
    target = ["you are so good to be true code."]

    compute_scores({1: target}, {1: pred})