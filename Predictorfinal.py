import tkinter as tk
import math
import argparse
import codecs
from collections import defaultdict
import Levenshtein
from operator import itemgetter



class Predictor(object):
    """ Generate a word based off contents of a model and a starting string consisting of a number of non-whitespace chars"""
    """ Utökad från Generator.py"""
    
    def __init__(self):
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        self.trigram_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0



    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        """ Oförändrad från labb 2"""

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))

                model_rows = list(f.readlines())

                # Unika ord
                for i in range(self.unique_words):
                    # "3 boston 1"
                    row = model_rows[i].strip().split()
                    self.word[i] = row[1]
                    self.index[row[1]] = i
                    self.unigram_count[row[1]] = row[2]


                #Bigramsannolikheter
                counter = 0
                for x in model_rows[i+1:]:
                    counter = counter + 1

                    bigram_row = x.strip().split()
                    if bigram_row[0] == "-1":   # En rad som bara är "-1" är det vi lägger till för att signalera slut på bigram
                        model_rows.pop()
                        break

                    self.bigram_prob[int(bigram_row[0])][int(bigram_row[1])] = float(bigram_row[2])
                    #bigram_row[0] är indexet för startord, bigram_row[1] är indexet för efterföljande och bigram_row[2] är sannolikhet

                #Trigramsannolikheter
                trigram_start_row = counter + i + 1
                for y in model_rows[trigram_start_row:]:
                    trigram_row = y.strip().split()
                    self.trigram_prob[int(trigram_row[0])][int(trigram_row[1])][int(trigram_row[2])] = float(trigram_row[3])
                
                return True

        except IOError:
            print("Couldn't find n-gram probabilities file {}".format(filename))
            return False




    """ De följande tre metoderna används när användaren avslutat det föregående ordet med _ och inte ännu börjat skriva på nästa"""

    def generate_next_word_trigram(self, penultimate_word, ultimate_word, words_to_return):
        """ 
        Metod för att generera nästa ord baserat på trigramsannolikheter. 
        Parametern words_to_return är alltid 3. 
        Om vi inte hittar 3 ord att returnera (pga att de föregående orden ksk inte förekommer som inledning på ett trigram),
        så kommer vi använda de ord vi hittar och be "generate_next_word_bigram" returnera de alternativ som återstår att returnera.
        Dvs att vi nyttjar Back-off i de fallen.
        """

        if penultimate_word in self.word.values() and ultimate_word in self.word.values():
            trigram_keys = list(self.trigram_prob[self.index[penultimate_word]][self.index[ultimate_word]].keys())
            sorted_trigrams = sorted(trigram_keys, key=lambda x: self.trigram_prob[self.index[penultimate_word]][self.index[ultimate_word]][x], reverse=True)
            top_trigrams = sorted_trigrams[:words_to_return]

            if len(top_trigrams) != words_to_return:
                words_left = words_to_return - len(top_trigrams)
                remaining_words = self.generate_next_word_bigram(ultimate_word, words_left, top_trigrams)
                remaining_words_index = []
                for i in remaining_words:
                    remaining_words_index.append(self.index[i])
                top_trigrams.extend(remaining_words_index)

            return [self.word[word_index] for word_index in top_trigrams]
        else:
            return self.generate_next_word_bigram(ultimate_word, words_to_return)
        

    def generate_next_word_bigram(self, ultimate_word, words_to_return, exclude_words=None):
        """
        Metod för att generera nästa ord från bigramsannolikheter.
        Om användaren hittils bara angett ett ord så kommer denna metod kallas på med words_to_return = 3.
        Annars kallas den på från trigram-metoden ovan med parametervärde {1-3} och returnerar så många ord.
        Dessutom görs en kontroll som hindrar oss från att skicka dubletter om vi redan hittat nån/några ord via trigram_metoden (parametern exclude_words).
        Kallar på metoden för att generera ord via unigram om bigram ej hittas, t.ex. vid ett föregående ord som inte finns i vårt dictionary.
        """

        if ultimate_word in self.word.values():
            bigram_keys = list(self.bigram_prob[self.index[ultimate_word]].keys())
            sorted_bigrams = sorted(bigram_keys, key=lambda x: self.bigram_prob[self.index[ultimate_word]][x], reverse=True)
            selected_words = []

            if exclude_words:
                for word_index in sorted_bigrams:
                    if word_index not in exclude_words:
                        selected_words.append(word_index)
                    if len(selected_words) == words_to_return:
                        break
            else:
                selected_words = sorted_bigrams[:words_to_return]

            if len(selected_words) != words_to_return:
                words_left = words_to_return - len(selected_words)
                remaining_words = self.generate_next_word_unigram(words_left, selected_words)
                remaining_words_index = []
                for i in remaining_words:
                    remaining_words_index.append(self.index[i])
                selected_words.extend(remaining_words_index)

            return [self.word.get(word_index, "UNKNOWN") for word_index in selected_words if isinstance(word_index, int)]
        else:
            return self.generate_next_word_unigram(words_to_return)


    def generate_next_word_unigram(self, words_to_return, exclude_words=None):
        """
        Genererar "words_to_return" antal ord via unigram-sannolikheter. 
        Innehåller kontroll för dubletter.
        """

        sorted_words = sorted(self.unigram_count.items(), key=lambda x: int(x[1]), reverse=True)
        selected_words = []
        if exclude_words:
            for word, count in sorted_words:
                if word not in exclude_words:
                    selected_words.append(word)
                if len(selected_words) == words_to_return:
                    break
        else:
            selected_words = [word for word, count in sorted_words[:words_to_return]]

        return selected_words
        




    """ De följande tre metoderna nyttjas när användaren börjat skriva på ord, dvs att sista inmatade karaktären inte är _"""
    """ De är liknande de ovan men innehåller en beräkning av levenshtein-avstånd som viktas mot tri/bi/unigram-sannolikheten 
        för att kunna välja mer träffsäkert vilket ord som ska föreslås"""
    

    def complete_word_unigram(self, current_word, wordstoreturn=None):
        """
        Tar in de bokstäver som skrivits hittills och beräknar Levenshtein-avstånd till orden i vårt dictionary
        Baserat på det och ordets sannolikhet att förekomma ges varje ord ett score där högre score betyder mer lämpligt förslag
        Metoden returnerar de ord som har högst score.
        """

        suggestions = []

        for word, count in self.unigram_count.items():
            levenshtein_distance = Levenshtein.distance(current_word, word)
            probability = float(count) / self.total_words

            # Vikt för enkel justering mellan vilket som borde prioriteras mer, sannolikhet att förekomma eller levenshteinavstånd
            weight = 0.1

            score = (1 - weight) * probability - levenshtein_distance * weight
            suggestions.append((word, score))

        sorted_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
        top_suggestions = sorted_suggestions[:3]
        suggested_words = [word for word, _ in top_suggestions]

        if wordstoreturn:
            suggested_words = suggested_words[:-wordstoreturn]

        return suggested_words


    
    def complete_word_bigram(self, ultimate_word, current_word, wordstoreturn=None):
        """
        Utför samma sak som metoden ovan, fast sannolikheterna är nu bigramsannolikheter.
        Kallar på complete_word_unigram om tillräckligt många bigram inte hittas.
        """

        
        if self.index[ultimate_word] in self.bigram_prob:
            index_second_words = list(self.bigram_prob[self.index[ultimate_word]].keys())
            suggestions = []

            if len(suggestions) < 3:
                number_of_extra_suggestions = 3-len(suggestions)
                extra_suggestions = self.complete_word_unigram(current_word, number_of_extra_suggestions)
                suggestions.extend(extra_suggestions)

            for index_second_word in index_second_words:
                probability = math.exp(self.bigram_prob[self.index[ultimate_word]][index_second_word])
                next_word = self.word[index_second_word]
                levenshtein_distance = Levenshtein.distance(current_word, next_word)
                weight = 0.05
                score = (1-weight) * probability - levenshtein_distance * weight
                suggestions.append((next_word, score))


            if wordstoreturn:
                sorted_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
                suggested_words = [word for word, _ in sorted_suggestions[:wordstoreturn]]
                
                if len(suggested_words) < wordstoreturn:
                    len_uni_words = wordstoreturn - len(suggested_words)
                    uni_words = self.complete_word_unigram(current_word, len_uni_words)
                    uni_words = [word for word in uni_words if word not in suggested_words]

                    if len(uni_words) > len_uni_words:
                        while len(uni_words) > len_uni_words:
                            uni_words.pop() 

                    suggested_words.extend(uni_words)
                return suggested_words

            sorted_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
            top_suggestions = sorted_suggestions[:3]
            suggested_words = [word for word, _ in top_suggestions]

            return suggested_words

        else:
            return self.complete_word_unigram(current_word)

    


    
    def complete_word_trigram(self, penultimate_word, ultimate_word, current_word):
        """
        Metod som kompletterar det tredje ordet i trigram. Använder Back-off vid behov. Samma metodik som ovan
        """


        try:
            outer_keys = list(self.trigram_prob.keys())
            intermediate_keys = list(self.trigram_prob[self.index[penultimate_word]].keys())

            if self.index[penultimate_word] in outer_keys and self.index[ultimate_word] in intermediate_keys:
                index_third_words = list(self.trigram_prob[self.index[penultimate_word]][self.index[ultimate_word]].keys())
                
                suggestions = []

                for index_third_word in index_third_words:
                    probability = math.exp(self.trigram_prob[self.index[penultimate_word]][self.index[ultimate_word]][index_third_word])
                    next_word = self.word[index_third_word]
                    levenshtein_distance = Levenshtein.distance(current_word, next_word)
                    weight = 0.01
                    score = (1-weight) * probability - levenshtein_distance * weight
                    suggestions.append((next_word, score))

                if len(suggestions) < 3:
                    number_of_extra_suggestions = 3-len(suggestions)
                    extra_suggestions = self.complete_word_bigram(ultimate_word, current_word, number_of_extra_suggestions)
                    suggestions.extend(extra_suggestions)
                    return suggestions

                top_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)[:3]
                suggested_words = [word for word, _ in top_suggestions]

                return suggested_words
            
        except KeyError:
            return self.complete_word_bigram(ultimate_word, current_word)
    
            



def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    arguments = parser.parse_args()

    # Predictor
    predictor = Predictor()
    predictor.read_model(arguments.file)

    # GUI
    guiwindow = GuiWindow(predictor)
    guiwindow.root.mainloop()  




class GuiWindow():

    def __init__(self, predictor):

        # Lista för knapparna
        self.option_buttons = []

        # Kopplar prediktorobjektet till GUI
        self.predictor = predictor

        # Main window
        self.root = tk.Tk()
        self.root.title("Word Predictor")

        # Input box label
        input_label = tk.Label(self.root, text="Input text here")
        input_label.pack()

        # Entry field
        self.entrybox = tk.Entry(self.root)
        self.entrybox.pack(padx=10, pady=5, fill=tk.X)
        self.entrybox.bind("<FocusIn>", self.update_predictions)   # Vid start
        self.entrybox.bind("<KeyRelease>", self.update_predictions)  # Binder key release event till att inleda uppdatering av ordförslagen


        # Backspace-hantering
        self.entrybox.bind("<Key>", self.handle_key_press)
        self.entrybox.bind("<BackSpace>", self.handle_backspace)


        self.root.mainloop()


    def handle_key_press(self, event):
        self.update_predictions()

    def handle_backspace(self, event):
        # Backspace tömmer hela input-lådan
        self.entrybox.delete(0, tk.END)



    def on_option_click(self, option_text):
        """
        Metod för att hantera tryck på knapparna.
        Olika beroende på om vi förutspår ett nytt ord från ingenting eller om vi kompletterar ett ord användaren redan börjat skriva på. 
        """
        current_text = self.entrybox.get()
        
        # Om vi kompletterar nuvarande ord tar vi bara bort det som skrivits och lägger knappinnehållet
        if current_text and not current_text.endswith(' '):
            last_space_index = current_text.rfind(' ')
            updated_text = current_text[:last_space_index + 1] + option_text + " "
            self.entrybox.delete(0, tk.END)
            self.entrybox.insert(tk.END, updated_text)
            self.update_predictions()

        # Om vi förutspått nuvarande ord lägger vi bara till det (behöver ej ta bort något)
        else:
            self.entrybox.insert(tk.END, option_text + " ")
            self.update_predictions()

    

    def update_predictions(self, *args):
        """ 
        Metod för att avgöra på vilket sätt (med vilka metoder) vi ska uppdatera ordförslagen
        """

        input_text = self.entrybox.get()

        # Om lådan är tom eller ordet är avslutat ska vi gissa nästa ord baserat enbart på ordföljden
        if not input_text or input_text.endswith(' '):
            self.update_predictions_entire_word()

        # Annars ska vi göra det + inkorporera det användaren skrivit i vårt försök att komplettera nuvarande ord 
        else:
           self.update_predictions_current_word()
           

    
    def update_predictions_entire_word(self):
        """
        För förslag av nästa ord, från ingenting.
        Kallar på olika metoder beroende på antalet tidigare ord:
        Trigram-metod om det finns 2+ ord sen tidigare.
        Bigram om det finns 1 ord sen tidigare.
        Unigram om det inte finns ett ord i inputbox.
        """

        input_text = self.entrybox.get().strip()
        words = input_text.split()
        predicted_words = []

        if not words:
            predicted_words = self.predictor.generate_next_word_unigram(3)
        elif len(words) == 1:
            predicted_words = self.predictor.generate_next_word_bigram(words[0], 3)
        else:
            predicted_words = self.predictor.generate_next_word_trigram(words[-2], words[-1], 3)

        self.update_button_content(predicted_words)

    
    
    def update_predictions_current_word(self):
        """
        Metod för uppdatering av föreslagna ord givet att användaren börjat skriva på ordet vi försöker föreslå.
        Kallar på olika metoder beroende på antalet tidigare ord, dock något olikt ovan.
        """

        input_text = self.entrybox.get().strip()
        words = input_text.split()

        # För 0 tidigare ord (words inkluderar det ofullständiga ordet som användaren skriver på)
        if len(words) == 1:
            predicted_words = self.predictor.complete_word_unigram(words[0])

        # För bigram så väljer vi att eventuellt ha med en option från unigram-kompletteringen. Detta då korpus (trots relativt stort) ofta ger 
        # förslag på andra-ord i bigram som inte är särskilt rimliga alternativt.
        # Innan det gör vi dock en dublett-kontroll
        elif len(words) == 2:
            predicted_words = self.predictor.complete_word_bigram(words[0], words[-1], 3)
            unigram_option = self.predictor.complete_word_unigram(words[-1], 1)
            if unigram_option and unigram_option[0] not in predicted_words:
                predicted_words[2] = unigram_option[0]

        # För trigram väljer vi också att ha samma upplägg. Trigram är ännu mer begränsade i förekommande tredje-ord, särskilt om 
        # De 2 föregående orden inte förekommer ofta. Dubletter kontrolleras även här
        else:
            predicted_words = self.predictor.complete_word_trigram(words[-3], words[-2], words[-1])
            unigram_option = self.predictor.complete_word_unigram(words[-1], 1)
            if unigram_option and unigram_option[0] not in predicted_words:
                predicted_words[2] = unigram_option[0]


        self.update_button_content(predicted_words)



    def update_button_content(self, predicted_words):
        """Uppdaterar knappinnehåll"""

        for button in self.option_buttons:
            button.destroy()
        self.option_buttons.clear()

        for i, prediction_text in enumerate(predicted_words):
            button = tk.Button(self.root, text=prediction_text, command=lambda text=prediction_text: self.on_option_click(text))
            button.pack(padx=10, pady=5, fill=tk.X)
            self.option_buttons.append(button)





if __name__ == "__main__":
    main()