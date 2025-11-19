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





class BigramTrainer(object):
    """ This class constructs a bigram language model from a corpus.  """

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
       
        """ The bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway). """

        self.bigram_count = defaultdict(lambda: defaultdict(int))
        # Nästat dictionary. En dictionary där varje unikt ord är en nyckel till varsitt eget dictionary


        # De nästade dictionariesen, med det efterföljande ordet som nyckel, innehåller värdet på antalet gånger bigrammet som består av de 2 nycklarna förekommer
        # dvs
        # self.bigram_count.keys() = {ALLA TOKENS I TEXTEN SOM INLEDER ETT BIGRAM} 
        # slef.bigram_count.keys() = {'i', 'live', 'in', 'boston', '.', 'like', 'ants', 'honey', 'therefore', 'too'}     för small.txt

        # self.bigram_count[self.word[0]].keys() = {ALLA FÖLJETOKENS I BIGRAM SOM INLEDS MED ORDET self.word.[0]}
        # self.bigram_count[self.word[0]].keys() = {live, like}    för small.txt där self.word[0] = "i"

        # self.bigram_count[self.word[0]][key] = ANTALET FÖREKOMSTER AV BIGRAMMET SOM INLEDS MED self.word[0] och följs av ordet [key]
        # self.bigram_count[self.word[0]]["like"] = 2 för small.txt vilket då är antalet förekomster av "i like"

        # The identifier of the previous word processed.
        self.last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        #self.laplace_smoothing = False  # INTE MED I ORIGINAL

    


        # IDENTISK
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
        """ Processes one word in the training corpus, and adjusts the unigram and bigram counts.
        :param token: The current word to be processed. """

        # YOUR CODE HERE

        self.unigram_count[token]+=1

        # Bara första ordet i corpus iom att last_index initieras till -1
        if self.last_index == -1:
            self.index[token] = 0   #index för första token blir såklart 0
            self.word[0] = token    #första token är första ordet
            self.last_index = 0     
            self.unique_words+=1


        else:
            self.bigram_count[self.word[self.last_index]][token]+=1
            #    bigram_count[    föregående ordet      ][detta ord] inkrementeras med 1

            try:
                self.last_index=self.index[token]
                #last index sätts till index för ordet vi nyss analyserade som ord2

            except KeyError:
                """ 
                Om self.index[token] inte funkar har self.index{} inte token som nyckel
                dvs vår token har inte förekommit innan och inte mappats till varken word{} eller index{}
                Så då gör vi det!
                """

                self.index[token] = self.unique_words   
                #om unika ord nr 3 förekommer på plats 5 får det indexvärde = 2 (self.index[token] = self.unique_words)
                self.word[self.index[token]] = token
                #och mappas till word också

                self.last_index=self.index[token]
                self.unique_words+=1

        self.total_words+=1







    def stats(self):
        """ Creates a list of rows to print of the language model."""
        rows_to_print = []
        # YOUR CODE HERE
        rows_to_print.append(str(self.unique_words) + " " + str(self.total_words))
        for i in range(self.unique_words):
            rows_to_print.append(str(i) + " " + str(self.word[i]) + " " + str(self.unigram_count[self.word[i]]))
            
        for i in range(self.unique_words):  
        # För varje unikt ord
            keys_list = list(self.bigram_count[self.word[i]].keys())    
            # hämta nycklarna i unika ordets dictionary (unikt ord = ord1, nycklarna = alla ord2)
            # för bigrammen (self.word[i], keys) = (ord1, ord2)

            for key in keys_list:
                bigram_log_probability = math.log(self.bigram_count[self.word[i]][key]/self.unigram_count[self.word[i]])

                # log(förekomster(bigram)/förekomster(ord1))
                # self.bigram_count[self.word[i]][key] kommer åt antalet förekomster
                rows_to_print.append(str(i) + " " + str(self.index[key]) + " " + f"{bigram_log_probability:.15f}")

        rows_to_print.append("-1")

        return rows_to_print



def main():
    """ Parse command line arguments """

    start_time = time.time()

    parser = argparse.ArgumentParser(description='BigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')

    arguments = parser.parse_args()     
    bigram_trainer = BigramTrainer()
    bigram_trainer.process_files(arguments.file)

    stats = bigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
            for row in stats: f.write(row + '\n')
    else:
        for row in stats: print(row)


    end_time = time.time()  # Record the end time
    elapsed_time = timedelta(seconds=end_time - start_time)  # Calculate elapsed time
    print(f"Time taken to finish: {elapsed_time}")


if __name__ == "__main__":
    main()