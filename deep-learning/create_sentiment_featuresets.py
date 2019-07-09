import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer


# Lemmatizer is used to convert several words written in differents ways to the same word, e. g. ran, run -> run.
lemmatizer = WordNetLemmatizer()
hm_lines = 100000

def create_lexicon(pos,neg):
    # Initializing lexicon (bag-of-words) as empty list
	lexicon = []
    # Reading positive files
	with open(pos,'r') as f:
        # Reading the content of the positive file
		contents = f.readlines()
        # Looping through the number of lines that we want
		for l in contents[:hm_lines]:
            # Tokenizing the lines read
			all_words = word_tokenize(l)
            # Appeding the set of words to the lexicon
			lexicon += list(all_words)

	with open(neg,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l)
			lexicon += list(all_words)
    
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	# Creating variable to count the frequency of each word of the lexicon in the dataset, e.g. {'the': 55304, 'and': 12332}
	w_counts = Counter(lexicon)
    # Creating new optimized lexicon
	optimized_lexicon = []
    # Looping through the dictionary of frequency of word
	for w in w_counts:
		#print(w_counts[w])
        # Only storing the words whose frequency it is in between 1000 and 50
		if 1000 > w_counts[w] > 50:
			optimized_lexicon.append(w)
	print(len(optimized_lexicon))
    # Return the optimized lexicon
	return optimized_lexicon


# Function to model the data, cast raw data into a feature set
def sample_handling(sample,lexicon,classification):

	# The featureset will be like: 
	# [
	# 	[0, 0, 0, 2],
	# 	[1, 1, 1, 0],
	# 	[0, 0, 2, 3],
	# 	[1, 0, 0, 0],
	# ]
	featureset = []

	with open(sample,'r') as f:
		# Read content of file
		contents = f.readlines()
		# For each line inside of the range of 0 to number of lines read
		for l in contents[:hm_lines]:
			# Tokenizing the words currently inside of the batch of lines read
			current_words = word_tokenize(l.lower())
			# Lemmatizing list of words
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			# Creating a instance (list of features) of the length of the lexicon with 0 values
			features = np.zeros(len(lexicon))
		
			for word in current_words:
				# If the word exist in the lexicon, this word is a feature
				if word.lower() in lexicon:
					# Find the index of the word in the lexicon
					index_value = lexicon.index(word.lower())
					# Increase frequency of that feature
					features[index_value] += 1
			# Convert numpy array to list
			features = list(features)
			# Appending list into the list that represents the featureset.
			featureset.append([features,classification])

	return featureset


def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
	# Creating lexicon with words of positive and negative datasets
	lexicon = create_lexicon(pos,neg)
	featureset = []
	# Converting raw data into features, and storing inside of featureset
	featureset += sample_handling('pos.txt',lexicon,[1,0])
	featureset += sample_handling('neg.txt',lexicon,[0,1])
	random.shuffle(featureset)
	# Converting featureset into a numpy arrat
	featureset = np.array(featureset)

	# Defining the testing size
	testing_size = int(test_size*len(featureset))
	# Slicing the featureset into train and test
	# The notation featureset[:, 0] means that we want all the elements of position 0, e.g. featureset is an matrix of features and labels, so featureset is [ [features], [labels] ] =====>
	# 		[ [ [1, 1, 0], [1, 0] ],
	# 		  [ [0, 0, 1], [0, 1] ],
	# 		]
	# 
	train_x = list(featureset[:,0][:-testing_size])
	train_y = list(featureset[:,1][:-testing_size])
	test_x = list(featureset[:,0][-testing_size:])
	test_y = list(featureset[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y


if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
	# if you want to pickle this data:
	with open('sentiment_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)