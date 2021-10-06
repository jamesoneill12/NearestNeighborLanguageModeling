import numpy as np
from util.neighbor import get_nearest_vocab_neighbor
import torch


class NNRS:
    """
    Replaces past word with a sampled nearest neighbor
    """
    def __init__(self):
        pass

    def get_neighbor_dict(self):
        # this not all embeddings, just that from the vocab
        neighbor_lookup = get_nearest_vocab_neighbor(self.dictionary.word2idx, self.dictionary.wv)
        return neighbor_lookup

    def sample_neighbor(self, word_id):
        ids, id_probs = self.neighbourhood_dict[word_id]
        # AH ! THIS IS THE <EOS> and <BOS> TAG SO IGNORE
        if float('nan') in id_probs:
            return word_id
        chosen_neighbor_id = np.random.choice(ids, p=id_probs)
        return chosen_neighbor_id

    def sample_target_sequence(self, word_ids, pred_ids, sample_prob):
        """
        Sample neighbors of input sequence at random position
        :param word_ids:
        :param pred_ids:
        :param sample_prob:
        :return: replaced_ids
        """
        word_ids = word_ids.clone().data.numpy()
        bools = np.random.choice([0, 1], size=(len(word_ids),), p=[1.0 - sample_prob, sample_prob])
        inds = np.where(bools).astype(int)
        word_ids[inds] = pred_ids[inds]
        replaced_ids = torch.cuda.LongTensor(word_ids)
        return replaced_ids

