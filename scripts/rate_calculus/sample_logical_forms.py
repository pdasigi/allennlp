#! /usr/bin/env python
import argparse
import os
import sys
import random
import pickle
import hashlib
from typing import List

# pylint: disable=wrong-import-position,invalid-name

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))

from allennlp.data.tokenizers import Token
from allennlp.semparse import ActionSpaceWalker
from allennlp.semparse.contexts import QuestionKnowledgeGraph
from allennlp.semparse.worlds import RateCalculusWorld


def print_sample(tokens: List[str],
                 max_length: int,
                 sample_size: int):
    token_hash = hashlib.md5(str(tokens).encode('utf-8')).hexdigest()
    serialized_walker_path = f"walker_{token_hash}_pl={max_length}.pkl"
    if os.path.isfile(serialized_walker_path):
        print("Reading serialized walker", file=sys.stderr)
        with open(serialized_walker_path, "rb") as serialized_file:
            walker = pickle.load(serialized_file)
    else:
        graph = QuestionKnowledgeGraph.read([Token(x) for x in tokens])
        world = RateCalculusWorld(graph)
        walker = ActionSpaceWalker(world, max_length)
        print("Serializing walker", file=sys.stderr)
        with open(serialized_walker_path, "wb") as serialized_file:
            pickle.dump(walker, serialized_file)
    # These are sorted by length
    large_sample = walker.get_all_logical_forms(10000)
    random.shuffle(large_sample)
    sample_to_print = large_sample[:sample_size]
    print("\n".join(sample_to_print))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, default="The Kings won two less than three times as"
                        "many games as they lost. They played 82 games. How many wins and losses"
                        "did the team have?")
    parser.add_argument("--max-length", type=int, default=10, dest="max_length")
    parser.add_argument("--sample-size", type=int, default=20, dest="sample_size")
    args = parser.parse_args()
    question_tokens = args.sentence.split()
    print_sample(question_tokens, args.max_length, args.sample_size)
