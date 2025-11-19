#  -*- coding: utf-8 -*-

import math
import argparse
import nltk
import codecs
from collections import defaultdict


"""

This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.

Created 2017 by Johan Boye and Patrik Jonell.

"""



class BigramTester(object):

    def __init__(self):
        
        """ This class reads a language model file and a test file, and computes 
        the entropy of the latter. 
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0



    def read_model(self, filename):
        # LIKADAN SOM GENERATOR
        """
        Reads the contents of the language model file into the appropriate data structures.
        
        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))

                # YOUR CODE HERE
                model_rows = list(f.readlines())
                model_rows.pop() #tar bort sista raden, -1

                for i in range(self.unique_words):
                    row = model_rows[i].strip().split()
                    self.word[i] = row[1]
                    self.index[row[1]] = i
                    self.unigram_count[row[1]] = int(row[2])

                for x in model_rows[i+1:]:
                    bigram_row = x.strip().split()
                    self.bigram_prob[int(bigram_row[0])][int(bigram_row[1])] = float(bigram_row[2])                

                return True

        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False





    def compute_entropy_cumulatively(self, word):

        # YOUR CODE HERE

        if word not in self.unigram_count:
            unigramprob = 0
            bigramprob = 0
            self.index[word] = len(self.index)  # ge ny plats
            self.word[self.index[word]] = word  # för nytt ord

        else:
            unigramprob = self.unigram_count[word]/self.total_words
            try:
                bigramprob = math.exp(self.bigram_prob[self.last_index][self.index[word]])
            except KeyError:
                # om KeyError så är inte index[word] en nyckel dvs dessa ord bildar ej ett bigram
                bigramprob = 0

        self.logProb += math.log(self.lambda1 * bigramprob + self.lambda2 * unigramprob + self.lambda3)*(-1/len(self.tokens))
        self.last_index = self.index[word]
        self.test_words_processed+=1

    
    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

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
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--test_corpus', '-t', type=str, required=True, help='test corpus')
    parser.add_argument('--check', action='store_true', help='check if your entropy calculation is correct')

    arguments = parser.parse_args()
    bigram_tester = BigramTester()
    bigram_tester.read_model(arguments.file)
    bigram_tester.process_test_file(arguments.test_corpus)

   
    print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb))



if __name__ == "__main__":
    main()