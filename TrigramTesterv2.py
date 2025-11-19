import math
import argparse
import nltk
import codecs
from collections import defaultdict

class TrigramTester(object):

    def __init__(self):
        """ Initializes variables and data structures for trigram entropy computation. """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        #Maybe use this tonto
        self.bigram_prob = defaultdict(lambda: defaultdict(int))

        # The trigram log-probabilities.
        self.trigram_prob = defaultdict(lambda: defaultdict(dict))

        # Number of unique words in the training corpus.
        self.unique_words = 0

        # Total number of words in the training corpus.
        self.total_words = 0

        # Average log-probability (entropy estimation) of the test corpus.
        self.logProb = 0

        # Identifiers of the previous two words processed in the test corpus.
        self.last_index = (-1, -1)

        # Prob mass unknown words
        self.lambda4 = 0.00001

        # Probability mass for unigram probs.
        self.lambda3 = 0.001 - self.lambda4

        # Probability mass for bigram probabilities.
        self.lambda2 = 0.05 - self.lambda3

        # Probability mass for trigram probabilities.
        self.lambda1 = 0.9

        # Number of words processed in the test corpus.
        self.test_words_processed = 0




    def read_model(self, filename):
        """ Reads the contents of the trigram language model file. """
        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))

                # Läs unigram-delen
                for _ in range(self.unique_words):
                    row = next(f).strip().split()
                    index = int(row[0])
                    self.word[index] = row[1]
                    self.index[row[1]] = index
                    self.unigram_count[row[1]] = int(row[2])

                # Initialize a flag to track the transition from bigram to trigram
                bigram_section = False

                # Läs bigram och trigram-delarna
                for row in f:
                    parts = row.strip().split()

                    # Check for the transition from bigram to trigram section
                    if len(parts) == 1 and parts[0] == "-1":
                        bigram_section = True
                        continue

                    if bigram_section:
                        # Process bigram probabilities
                        i1, i2 = map(int, parts[:2])
                        prob = float(parts[3])
                        self.trigram_prob[i1][i2][-1] = prob  # -1 used to denote bigram probability
                    else:
                        # Process trigram probabilities
                        i1, i2, i3 = map(int, parts)
                        prob = float(parts[4])
                        self.trigram_prob[i1][i2][i3] = prob

                return True

        except IOError:
            print("Couldn't find trigram probabilities file {}".format(filename))
            return False




    def compute_entropy_cumulatively(self, word):
        """ Computes entropy cumulatively for a given word using trigram model. """
        if word not in self.unigram_count:
            unigramprob = 0
            bigramprob = 0  # Include a variable for bigram probability
            trigramprob = 0
            self.index[word] = len(self.index)
            self.word[self.index[word]] = word
        else:
            unigramprob = self.unigram_count[word] / self.total_words
            i1, i2 = self.last_index
            i3 = self.index.get(word, -1)  # Använd -1 om ordet inte finns i index

            trigramprob = 0
            if i1 in self.trigram_prob and i2 in self.trigram_prob[i1] and i3 in self.trigram_prob[i1][i2]:
                trigramprob = math.exp(self.trigram_prob[i1][i2][i3])

            # Calculate bigram probability if available
            bigramprob = 0
            if i2 in self.trigram_prob and i3 in self.trigram_prob[i2]:
                bigramprob = math.exp(self.trigram_prob[i2][i3])

        # Include bigram probability in the entropy calculation
        self.logProb += math.log(self.lambda1 * trigramprob + self.lambda2 * bigramprob + self.lambda3 * unigramprob + self.lambda4) * (-1 / len(self.tokens))
        self.last_index = (self.last_index[1], self.index[word])  
        self.test_words_processed += 1




    def process_test_file(self, test_filename):
        """ Reads and processes the test file one word at a time. """
        try:
            with codecs.open(test_filename, 'r', 'utf-8') as f:
                self.tokens = nltk.word_tokenize(f.read().lower()) 
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)
            return True
        except IOError:
            print('Error reading testfile')
            return False

def main():
    parser = argparse.ArgumentParser(description='TrigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--test_corpus', '-t', type=str, required=True, help='test corpus')

    arguments = parser.parse_args()
    trigram_tester = TrigramTester()
    if trigram_tester.read_model(arguments.file):
        if trigram_tester.process_test_file(arguments.test_corpus):
            print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(trigram_tester.test_words_processed, trigram_tester.logProb))

if __name__ == "__main__":
    main()