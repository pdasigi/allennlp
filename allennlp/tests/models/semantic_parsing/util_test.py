# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.semantic_parsing import util


class TestSemparseUtil(AllenNlpTestCase):
    def test_get_child_span_mask(self):
        span_indices = [[1, 1],
                        [2, 3],
                        [2, 11],
                        [5, 6],
                        [7, 9],
                        [7, 11],
                        [8, 9],
                        [10, 11],
                        [-1, -1]]  # This is padding.
        child_span_mask = util.get_child_span_mask(span_indices)
        # There are 1s only for three spans:
        # [2, 11] is the direct parent of [2, 3], [5, 6], and [7, 11]
        # [7, 9] is the direct parent of [8, 9]
        # [7, 11] is the direct parent of [7, 9] and [10, 11]
        assert child_span_mask == [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    def test_get_child_span_mask_with_partial_overlaps(self):
        # The previous test checks the case where the spans are outputs of a constituent parser, and hence they
        # either completely overlap other spans or are fully disjoint. This one tests the case where the spans may
        # be partially overlapping. Spans are recognized as direct parents only if they completely overlap other
        # spans.
        span_indices = [[1, 3],
                        [2, 4],
                        [3, 5],
                        [2, 6],
                        [7, 10],
                        [7, 11],
                        [8, 11]]
        child_span_mask = util.get_child_span_mask(span_indices)
        # [2, 6] is the direct parent of [2, 4] and [3, 5]
        # [7, 11] is the direct parent of [7, 10] and [8, 11]
        assert child_span_mask == [[0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0]]
