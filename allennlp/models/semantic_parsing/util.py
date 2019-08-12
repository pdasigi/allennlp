from typing import List


def get_child_span_mask(span_indices: List[List[int]],
                        include_self: bool = False) -> List[List[int]]:
    # Takes a list of spans of size k and returns a matrix (as a list of list) of size k * k that has 1s only
    # on cells (i, j) if span i fully contains span j, and there is no span longer than j and shorter than i
    # that also contains j.
    # If ``include_self`` is set, the mask for each node will also include the index of the node itself. That is,
    # elements at (i, i) are set to 1 for all i that correspond to non-padding spans.
    num_spans = len(span_indices)
    child_span_masks = []
    for begin_i, end_i in span_indices:
        if begin_i == -1:
            child_span_masks.append([0] * num_spans)
            continue
        child_span_masks.append([])
        for begin_j, end_j in span_indices:
            if begin_j == -1:
                child_span_masks[-1].append(0)
            elif begin_i <= begin_j and end_i >= end_j and (begin_i, end_i) != (begin_j, end_j):
                child_span_masks[-1].append(1)
            else:
                child_span_masks[-1].append(0)
    for i in range(num_spans):
        for j in range(num_spans):
            if child_span_masks[i][j] == 0:
                continue
            for k in range(num_spans):
                if child_span_masks[i][k] == 1 and child_span_masks[k][j] == 1 and k != i:
                    # There exists a span k that is a subspan of i such that j is also a subspan of k. So j is
                    # not a direct subspan of i.
                    child_span_masks[i][j] = 0
    if include_self:
        for i, span in enumerate(span_indices):
            if span != [-1, -1]:
                child_span_masks[i][i] = 1
    return child_span_masks
