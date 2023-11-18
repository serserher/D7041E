from cv2 import add
import numpy as np
from zmq import EVENT_CLOSE_FAILED
import text_functions as tf
import nltk

#@author: The first version of this code is the courtesy of Vadim Selyanik

threshold = 15000 # Frequency threshold in the corpus ??
dimension = 10000 # Dimensionality for high-dimensional vectors
lemmatizer = nltk.WordNetLemmatizer()  # create an instance of lemmatizer
ones_number = 2 # number of nonzero elements in randomly generated high-dimensional vectors
window_size = 2 #number of neighboring words to consider both back and forth. In other words number of words before/after current word
#I find it interesting that I have increased the size of the window to 4, and that reduced the accuracy from 70% to 61.25, I had reasons in favor of both it reducing and
#also it decreasing. I thought it could decrease as with size = 5 that is exactly the structure that we will find in our test data from TOEFL, but also I thought the 
#algorithm would catch more context with a bigger window leading to more accuracy
zero_vector = np.zeros(dimension)
test_name = "new_toefl.txt" # file with TOEFL dataset
data_file_name = "lemmatized.text" # file with the text corpus

amount_dictionary = {}

# Count how many times each word appears in the corpus
text_file = open(data_file_name, "r")
for line in text_file:
    if line != "\n":
        words = line.split()
        for word in words:
            if amount_dictionary.get(word) is None:
                amount_dictionary[word] = 1
            else:
                amount_dictionary[word] += 1
text_file.close()

dictionary = {} #vocabulary and corresponing random high-dimensional vectors
word_space = {} #embedings




#Create a dictionary with the assigned random high-dimensional vectors
text_file = open(data_file_name, "r")
for line in text_file: #read line in the file
    words = line.split() # extract words from the line
    for word in words:  # for each word
        if dictionary.get(word) is None: # If the word was not yed added to the vocabulary
            if amount_dictionary[word] < threshold:
                dictionary[word] = tf.get_random_word_vector(dimension, ones_number) # assign a  
            else:
                dictionary[word] = np.zeros(dimension) # frequent words are assigned with empty vectors. In a way they will not contribute to the word embedding

text_file.close()


#Note that in order to save time we only create embeddings for the words needed in the TOEFL task

    #Find all unique words amongst TOEFL tasks and initialize their embeddings to zeros    
number_of_tests = 0
text_file = open(test_name, "r") #open TOEFL tasks
for line in text_file:
        words = line.split()
        words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, 'v'), 'n'), 'a') for word in
                 words] # lemmatize words in the current test
        word_space[words[0]] = np.zeros(dimension)
        word_space[words[1]] = np.zeros(dimension)
        word_space[words[2]] = np.zeros(dimension)
        word_space[words[3]] = np.zeros(dimension)
        word_space[words[4]] = np.zeros(dimension)
        number_of_tests += 1
text_file.close()


    # Processing the text to build the embeddings 
text_file = open(data_file_name, "r")
lines = ["\n"]*2*window_size
print(len(lines))
"""while i < 4:
        line = "\n"
        while line == "\n":
            line = text_file.readline()
        lines[i] = line.split()
        i += 1
"""
i = 0
while i < 2*window_size:
    line = "\n"
    while line == "\n":
        line = text_file.readline()
    lines[i] = line.split()
    i += 1
print(lines)

line = text_file.readline()
while line != "":
        if line != "\n":
            lines.append(line.split())
            line_index = window_size
            words = lines[line_index]
            length = len(words)
            i = 0
            while i < length:
                if not (word_space.get(words[i]) is None):
                    lines_behind_left = 1
                    aux = 1
                    for k in range(1,window_size+1): #Handling left context
                        words_left = i-k
                        if words_left >= 0:
                            word_space[words[i]] = np.add(word_space[words[i]], np.roll(dictionary[words[i-k]],-k))
                        else:
                            line_index_aux = window_size - lines_behind_left #This way we calculate which one is the index for the line we're going back to

                            if aux >= len(lines[line_index_aux]): #we will ONLY enter this loop if we run out of words in the line we're looking for words
                                
                                aux = 1 #I created this variable to index in the lines to the left in case that there aren't enough words inside the main line
                                
                                lines_behind_left += 1 #The idea of this variable is to be able to go further back than just one line in case than necessary

                                line_index_aux = window_size - lines_behind_left #obviously we would have to recalculate this index

                            if lines[line_index_aux]: #here we check that the line on that calculated index exists
                                word_space[words[i]] = np.add(word_space[words[i]], np.roll(dictionary[lines[line_index_aux][-aux]],-aux))
                                aux += 1

                    lines_advanced_right = 1 #we invert the process that we did before creating a variable to indicate how many lines we've advanced
                    aux = 0
                    for k in range(1,window_size+1): #Handling right context
                        words_reached = i+k
                        if length > words_reached:
                            word_space[words[i]] = np.add(word_space[words[i]], np.roll(dictionary[words[i+k]],k))
                            
                        else:
                            line_index_aux = window_size + lines_advanced_right #This way we calculate which one is the index for the line we're going back to

                            if aux > len(lines[line_index_aux]): #we will ONLY enter this loop if we run out of words in the line we're looking for words
                                
                                aux = 0 #I created this variable to index in the lines to the left in case that there aren't enough words inside the main line
                                
                                lines_advanced_right += 1 #The idea of this variable is to be able to go further back than just one line in case than necessary

                                line_index_aux = window_size + lines_advanced_right #obviously we would have to recalculate this index

                            if lines[line_index_aux] and len(lines[line_index_aux]) > aux: #here we check that the line on that calculated index exists
                                word_space[words[i]] = np.add(word_space[words[i]], np.roll(dictionary[lines[line_index_aux][aux]],aux))
                                aux += 1
                i += 1
            lines.pop(0)
        line = text_file.readline()



#Testing of the embeddings on TOEFL
a = 0.0 # accuracy of the encodings    
i = 0
text_file = open(test_name, 'r')
right_answers = 0.0 # variable for correct answers
number_skipped_tests = 0.0 # some tests could be skipped if there are no corresponding words in the vocabulary extracted from the training corpus
while i < number_of_tests:
        line = text_file.readline() #read line in the file
        words = line.split()  # extract words from the line
        words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, 'v'), 'n'), 'a') for word in
                  words]  # lemmatize words in the current test
        try:
            
            if not(amount_dictionary.get(words[0]) is None): # check if there word in the corpus for the query word
                k = 1
                while k < 5:
                    # if amount_dictionary.get(words[k]) is None:
                    #     word_space[words[k]] = np.random.randn(dimension)
                    if np.array_equal(word_space[words[k]], zero_vector): # if no representation was learnt assign a random vector
                        word_space[words[k]] = np.random.randn(dimension)
                    k += 1
                right_answers += tf.get_answer_mod([word_space[words[0]],word_space[words[1]],word_space[words[2]],
                            word_space[words[3]],word_space[words[4]]]) #check if word is predicted right
        except KeyError: # if there is no representation for the query vector than skip
            number_skipped_tests += 1
            print("skipped test: " + str(i) + "; Line: " + str(words))
        except IndexError:
            print(i)
            print(line)
            print(words)
            break
        i += 1
text_file.close()
a += 100 * right_answers / number_of_tests
print(str(dimension) + " Percentage of correct answers: " + str(100 * right_answers / number_of_tests) + "%")



