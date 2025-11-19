This is a word predictor implemented using Naive Bayes and Markov assumptions. 

The program runs similarly to autocorrect on phones (including the 3 options that android offers).
It predicts the word that is currently being typed, including error checking via Levenshtein distance.
It also predicts the next word using uni-, bi- and trigrams. However, trigrams are not implemented as the training took too long/was too memory expensive

Run it by running predictorfinal-py -f [FILENAME].
Files included that can run are iphone_bigram_model.txt and android_bigram_model.txt

See report pdf for more information.
