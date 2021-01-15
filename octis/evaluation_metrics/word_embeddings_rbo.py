from __future__ import division

import math
from bisect import bisect_left
from collections import namedtuple
from collections import OrderedDict

RBO = namedtuple("RBO", "min res ext")
RBO.__doc__ += ": Result of full RBO analysis"
RBO.min.__doc__ = "Lower bound estimate"
RBO.res.__doc__ = "Residual corresponding to min; min + res is an upper bound estimate"
RBO.ext.__doc__ = "Extrapolated point estimate"


def _round(obj):
    if isinstance(obj, RBO):
        return RBO(_round(obj.min), _round(obj.res), _round(obj.ext))
    else:
        return round(obj, 3)


def set_at_depth(lst, depth):
    ans = set()
    for v in lst[:depth]:
        if isinstance(v, set):
            ans.update(v)
        else:
            ans.add(v)
    return ans


def embeddings_overlap(list1, list2, depth, index2word, word2vec, norm=True):
    # set1, set2 = set_at_depth(list1, depth), set_at_depth(list2, depth)
    # return len(set1.intersection(set2)), len(set1), len(set2)

    set1, set2 = set_at_depth(list1, depth), set_at_depth(list2, depth)
    word_list1 = [index2word[index] for index in list1]
    word_list2 = [index2word[index] for index in list2]

    similarities = {}
    for w1 in word_list1[:depth]:
        for w2 in word_list2[:depth]:
            cos_sim = word2vec.similarity(w1, w2)
            if cos_sim > 1:
                cos_sim = 1
            elif cos_sim < -1:
                cos_sim = -1

            if norm:
                similarities[(w1, w2)] = 1 - (math.acos(cos_sim) / math.pi)
            else:
                similarities[(w1, w2)] = cos_sim  # + 1)/2
            # similarities[(w1,w2)] = (word2vec.similarity(w1, w2) + 1)/2

    similarities = OrderedDict(sorted(similarities.items(), key=lambda x: -x[1]))

    e_ov = 0
    key_list = list(similarities.keys())
    for k in key_list:
        if k in similarities.keys():
            # print(k, similarities[k])
            e_ov = e_ov + similarities[k]
            similarities = {save_k: v for save_k, v in similarities.items()
                            if save_k[0] != k[0] and save_k[1] != k[1]}
    # e_ov = 1
    # print("****")
    return e_ov, len(set1), len(set2)


def overlap(list1, list2, depth, index2word, word2vec, norm):
    # return agreement(list1, list2, depth) * min(depth, len(list1), len(list2))
    # NOTE: comment the preceding and uncomment the following line if you want
    # to stick to the algorithm as defined by the paper
    ov = embeddings_overlap(list1, list2, depth, index2word, word2vec, norm)[0]
    # print("overlap", ov)
    return ov


def agreement(list1, list2, depth, index2word, word2vec, norm):
    """Proportion of shared values between two sorted lists at given depth."""
    len_intersection, len_set1, len_set2 = embeddings_overlap(list1, list2, depth, index2word, word2vec, norm)
    return 2 * len_intersection / (len_set1 + len_set2)


def cumulative_agreement(list1, list2, depth, index2word, word2vec, norm):
    return (agreement(list1, list2, d, index2word, word2vec, norm) for d in range(1, depth + 1))


'''
def average_overlap(list1, list2, index2word, word2vec, depth=None):
    """Calculate average overlap between ``list1`` and ``list2``.
    """
    depth = min(len(list1), len(list2)) if depth is None else depth
    return sum(cumulative_agreement(list1, list2, depth, index2word=index2word, word2vec=word2vec, norm)) / depth


def rbo_at_k(list1, list2, p, index2word, word2vec, depth=None):
    # ``p**d`` here instead of ``p**(d - 1)`` because enumerate starts at
    # 0
    depth = min(len(list1), len(list2)) if depth is None else depth
    d_a = enumerate(cumulative_agreement(list1, list2, depth, index2word=index2word, word2vec=word2vec))
    return (1 - p) * sum(p ** d * a for (d, a) in d_a)
'''


def rbo_min(list1, list2, p, index2word, word2vec, depth=None, norm=True):
    """Tight lower bound on RBO.
    See equation (11) in paper.
    """
    depth = min(len(list1), len(list2)) if depth is None else depth
    x_k = overlap(list1, list2, depth, index2word, word2vec, norm)
    log_term = x_k * math.log(1 - p)
    sum_term = sum(
        p ** d / d * (overlap(list1, list2, d, index2word, word2vec=word2vec, norm=norm) - x_k) for d in
        range(1, depth + 1)
    )
    return (1 - p) / p * (sum_term - log_term)


def rbo_res(list1, list2, p, index2word, word2vec, norm):
    """Upper bound on residual overlap beyond evaluated depth.
    See equation (30) in paper.
    NOTE: The doctests weren't verified against manual computations but seem
    plausible. In particular, for identical lists, ``rbo_min()`` and
    ``rbo_res()`` should add up to 1, which is the case.
    """
    S, L = sorted((list1, list2), key=len)
    s, l = len(S), len(L)
    x_l = overlap(list1, list2, l, index2word, word2vec, norm)
    # since overlap(...) can be fractional in the general case of ties and f
    # must be an integer --> math.ceil()
    f = int(math.ceil(l + s - x_l))
    # upper bound of range() is non-inclusive, therefore + 1 is needed
    term1 = s * sum(p ** d / d for d in range(s + 1, f + 1))
    term2 = l * sum(p ** d / d for d in range(l + 1, f + 1))
    term3 = x_l * (math.log(1 / (1 - p)) - sum(p ** d / d for d in range(1, f + 1)))
    return p ** s + p ** l - p ** f - (1 - p) / p * (term1 + term2 + term3)


def rbo_ext(list1, list2, p, index2word, word2vec, norm):
    """RBO point estimate based on extrapolating observed overlap.
    See equation (32) in paper.
    NOTE: The doctests weren't verified against manual computations but seem
    plausible.
    >>> _round(rbo_ext("abcdefg", "abcdefg", .9))
    1.0
    >>> _round(rbo_ext("abcdefg", "bacdefg", .9))
    0.9
    """
    S, L = sorted((list1, list2), key=len)
    s, l = len(S), len(L)
    x_l = overlap(list1, list2, l, index2word, word2vec, norm)
    x_s = overlap(list1, list2, s, index2word, word2vec, norm)
    # the paper says overlap(..., d) / d, but it should be replaced by
    # agreement(..., d) defined as per equation (28) so that ties are handled
    # properly (otherwise values > 1 will be returned)
    # sum1 = sum(p**d * overlap(list1, list2, d)[0] / d for d in range(1, l + 1))
    sum1 = sum(p ** d * agreement(list1, list2, d, index2word=index2word, word2vec=word2vec, norm=norm)
               for d in range(1, l + 1))
    sum2 = sum(p ** d * x_s * (d - s) / s / d for d in range(s + 1, l + 1))
    term1 = (1 - p) / p * (sum1 + sum2)
    term2 = p ** l * ((x_l - x_s) / l + x_s / s)
    return term1 + term2


def word_embeddings_rbo(list1, list2, p, index2word, word2vec, norm):
    """Complete RBO analysis (lower bound, residual, point estimate).
    ``list`` arguments should be already correctly sorted iterables and each
    item should either be an atomic value or a set of values tied for that
    rank. ``p`` is the probability of looking for overlap at rank k + 1 after
    having examined rank k.
    >>> lst1 = [{"c", "a"}, "b", "d"]
    >>> lst2 = ["a", {"c", "b"}, "d"]
    >>> _round(rbo(lst1, lst2, p=.9))
    RBO(min=0.489, res=0.477, ext=0.967)
    """
    if not 0 <= p <= 1:
        raise ValueError("The ``p`` parameter must be between 0 and 1.")
    args = (list1, list2, p, index2word, word2vec, norm)

    return RBO(rbo_min(*args), rbo_res(*args), rbo_ext(*args))


def sort_dict(dct, *, ascending=False):
    """Sort keys in ``dct`` according to their corresponding values.
    Sorts in descending order by default, because the values are
    typically scores, i.e. the higher the better. Specify
    ``ascending=True`` if the values are ranks, or some sort of score
    where lower values are better.
    Ties are handled by creating sets of tied keys at the given position
    in the sorted list.
    >>> dct = dict(a=1, b=2, c=1, d=3)
    >>> list(sort_dict(dct)) == ['d', 'b', {'a', 'c'}]
    True
    >>> list(sort_dict(dct, ascending=True)) == [{'a', 'c'}, 'b', 'd']
    True
    """
    scores = []
    items = []
    # items should be unique, scores don't have to
    for item, score in dct.items():
        if not ascending:
            score *= -1
        i = bisect_left(scores, score)
        if i == len(scores):
            scores.append(score)
            items.append(item)
        elif scores[i] == score:
            existing_item = items[i]
            if isinstance(existing_item, set):
                existing_item.add(item)
            else:
                items[i] = {existing_item, item}
        else:
            scores.insert(i, score)
            items.insert(i, item)
    return items


def rbo_dict(dict1, dict2, p, index2word, word2vec, norm, *, sort_ascending=False):
    """Wrapper around ``rbo()`` for dict input.
    Each dict maps items to be sorted to the score according to which
    they should be sorted. The RBO analysis is then performed on the
    resulting sorted lists.
    The sort is descending by default, because scores are typically the
    higher the better, but this can be overridden by specifying
    ``sort_ascending=True``.
    """
    list1, list2 = (
        sort_dict(dict1, ascending=sort_ascending),
        sort_dict(dict2, ascending=sort_ascending),
    )
    return word_embeddings_rbo(list1, list2, p, index2word, word2vec, norm)
