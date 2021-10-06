import numpy as np


class LogUniformSampler(object):
    def __init__(self, ntokens):

        self.N = ntokens
        self.prob = [0] * self.N

        self.generate_distribution()

    def generate_distribution(self):
        for i in range(self.N):
            self.prob[i] = (np.log(i+2) - np.log(i+1)) / np.log(self.N + 1)

    def probability(self, idx):
        return self.prob[idx]

    def expected_count(self, num_tries, samples):
        freq = list()
        for sample_idx in samples:
            freq.append(-(np.exp(num_tries * np.log(1-self.prob[sample_idx]))-1))
        return freq

    def accidental_match(self, labels, samples):
        sample_dict = dict()

        for idx in range(len(samples)):
            sample_dict[samples[idx]] = idx

        result = list()
        for idx in range(len(labels)):
            if labels[idx] in sample_dict:
                result.append((idx, sample_dict[labels[idx]]))

        return result

    def sample(self, size, labels):
        log_N = np.log(self.N)

        x = np.random.uniform(low=0.0, high=1.0, size=size)
        value = np.floor(np.exp(x * log_N)).astype(int) - 1
        samples = value.tolist()

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)

        return samples, true_freq, sample_freq

    def sample_unique(self, size, labels):
        # Slow. Not Recommended.
        log_N = np.log(self.N)
        samples = list()

        while (len(samples) < size):
            x = np.random.uniform(low=0.0, high=1.0, size=1)[0]
            value = np.floor(np.exp(x * log_N)).astype(int) - 1
            if value in samples:
                continue
            else:
                samples.append(value)

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)

        return samples, true_freq, sample_freq