#  -*- coding: utf-8 -*-

from __future__ import unicode_literals
import faulthandler
faulthandler.enable()
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs
import time
from datetime import timedelta

"""

This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.

Created 2017 by Johan Boye and Patrik Jonell.

"""


class TrigramTrainer(object):
    """ This class constructs a tri-, bi-, and unigram language model from a corpus.  """

    def __init__(self):
        """ <p>Constructor. Processes the file <code>f</code> and builds a language model from it.</p>
        :param f: The training file. """

        # The mapping from words to identifiers.
        self.index = {}
        # Var varje ord förekommer första gången i texten

        # The mapping from identifiers to words.
        self.word = {}
        # Ordet som finns på indexplats __ (har en integer som nyckel)

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)
       
        """ The tri and bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway). """

        self.bigram_count = defaultdict(lambda: defaultdict(int))
        # Nästat dictionary. En dictionary där varje unikt ord är en nyckel till varsitt eget dictionary

        self.trigram_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))



            #LOGIKEN HÄRIFRÅN KVARSTÅR
        # De nästade dictionariesen, med det efterföljande ordet som nyckel, innehåller värdet på antalet gånger bigrammet som består av de 2 nycklarna förekommer
        # dvs
        # self.bigram_count.keys() = {ALLA TOKENS I TEXTEN SOM INLEDER ETT BIGRAM} 
        # slef.bigram_count.keys() = {'i', 'live', 'in', 'boston', '.', 'like', 'ants', 'honey', 'therefore', 'too'}     för small.txt

        # self.bigram_count[self.word[0]].keys() = {ALLA FÖLJETOKENS I BIGRAM SOM INLEDS MED ORDET self.word.[0]}
        # self.bigram_count[self.word[0]].keys() = {live, like}    för small.txt där self.word[0] = "i"

        # self.bigram_count[self.word[0]][key] = ANTALET FÖREKOMSTER AV BIGRAMMET SOM INLEDS MED self.word[0] och följs av ordet [key]
        # self.bigram_count[self.word[0]]["like"] = 2 för small.txt vilket då är antalet förekomster av "i like"

        # The identifier of the previous word processed.
        self.ultimate_index = -1

        # Two words previous, behövs för trigram
        self.penultimate_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

    

    # OFÖRÄNDRAD
    def process_files(self, f):
        """ Processes the file @code{f}. """
        with codecs.open(f, 'r', 'utf-8') as text_file:
            text = reader = str(text_file.read()).lower()
        try :
            self.tokens = nltk.word_tokenize(text) 
        except LookupError :
            nltk.download('punkt')
            self.tokens = nltk.word_tokenize(text)
        for token in self.tokens:
            self.process_token(token)





    def process_token(self, token):
        """ Processes one word in the training corpus, and adjusts unigram, bigram and now also trigram counts.
        :param token: The current word to be processed. """
        
        """ Annan idé jmf med originalet, ingen try-except"""

        # Först och främst
        if token in self.index:
            self.unigram_count[token] += 1
        else:
            # För aldrig tidigare lästa ord
            self.index[token] = len(self.index)
            self.word[(len(self.index) - 1)] = token
            self.unigram_count[token] = 1
            self.unique_words += 1
        
        #Uppdaterar bigram. Kommer gå in i denna när det är möjligt att göra bigram, dvs minst en loop har körts (minst 1 ord läst)
        if self.ultimate_index != -1:
            ultimate_word = self.word[self.ultimate_index]

            # ordföljd = A B dvs vi läser nuvarande ord som B och föregående ord som A
            # Alltså försöker vi skapa B|A
            if ultimate_word in self.bigram_count:
                # Om vi redan har bigram som inleds med A

                if token in self.bigram_count[ultimate_word]:
                    # Om vi har instanser av B|A
                    self.bigram_count[ultimate_word][token] += 1

                else:
                    # Om vi inte har B|A men andra bigram som börjar på A finns
                    self.bigram_count[ultimate_word][token] = 1

            else:
                # Om vi inte har bigram som inleds med A
                self.bigram_count[ultimate_word][token] = 1


        # Uppdaterar Trigram. Kommer gå in i denna när det är möjligt att göra trigram, dvs minst 2 ord har lästs (inte nödvändigtvis unika)
        # Följer samma logik, blir några extra if-satser
        if self.penultimate_index != -1:
            penultimate_word = self.word[self.penultimate_index]
            ultimate_word = self.word[self.ultimate_index]

            #C|BA
            if penultimate_word in self.trigram_count:  # om A finns i trigram count 

                if ultimate_word in self.trigram_count[penultimate_word]:   # Om B|A finns i trigram count

                    if token in self.trigram_count[penultimate_word][ultimate_word]:    # Om C|BA finns i trigram count
                        self.trigram_count[penultimate_word][ultimate_word][token] += 1

                    else:   # Om C|BA inte finns i trigram count
                        self.trigram_count[penultimate_word][ultimate_word][token] = 1

                else: # Om B|A ej finns i trigram count
                    self.trigram_count[penultimate_word][ultimate_word][token] = 1

            else:   # Om A inte finns i trigram count
                self.trigram_count[penultimate_word][ultimate_word][token] = 1


        self.penultimate_index = self.ultimate_index
        self.ultimate_index = self.index[token]
        self.total_words += 1


    def stats(self):
        """
        Creates a list of rows to print of the language model.
        """
        rows_to_print = []
        bigram_rows = []
        trigram_rows = []

        first_row = str(self.unique_words) + " " + str(self.total_words)
        rows_to_print.append(first_row)

        """ Denna lösning håller ner tidskomplexiteten för utskrift av stats jämfört med vår original-lösning i labb 2"""

        # unigram probabilities of a word
        for i in range(len(self.word)):
            word = self.word[i]
            word_frequency = self.unigram_count[word]
            rows_to_print.append(str(i) + " " + word + " " + str(word_frequency))

            # Bigram probabilities of the same word
            for second_word in self.bigram_count[word]:
                bigram_occurrences = self.bigram_count[word][second_word]
                probability = str("%.15f" % math.log(bigram_occurrences/word_frequency))
                bigram_rows.append(str(self.index[word]) + " " + str(self.index[second_word]) + " " + probability)

                # Trigram probabilities of the same word
                for third_word in self.trigram_count[word][second_word]:
                    trigram_occurrences = self.trigram_count[word][second_word][third_word]
                    probability = str("%.15f" % math.log(trigram_occurrences/bigram_occurrences))
                    trigram_rows.append(str(self.index[word]) + " " + str(self.index[second_word]) + " " + str(self.index[third_word]) + " " + probability)

        for row in bigram_rows:
            rows_to_print.append(row)
        rows_to_print.append("-1")

        for row in trigram_rows:
            rows_to_print.append(row)

        # Rows to print är vår språkmodell

        return rows_to_print


def main():
    """ Parse command line arguments """
    
    #start_time = time.time()

    parser = argparse.ArgumentParser(description='TrigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')

    arguments = parser.parse_args()     
    trigram_trainer = TrigramTrainer()
    trigram_trainer.process_files(arguments.file)

    stats = trigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
            for row in stats: f.write(row + '\n')
    else:
        for row in stats: print(row)

    #end_time = time.time()  # Record the end time
    #elapsed_time = timedelta(seconds=end_time - start_time)  # Calculate elapsed time
    #print(f"Time taken to finish: {elapsed_time}")


if __name__ == "__main__":
    main()