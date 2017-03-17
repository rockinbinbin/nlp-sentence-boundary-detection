# Assignment 1
# Robin Mehta
#robimeht

from __future__ import division, print_function
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import sys

def parse_data_set(path, just_words=False):
	temp = []

	with open(path,'r') as data:
		for line in data.read().split('\n'):
			if (len(line) > 2) and (line.split()[2] != 'TOK'): # handles empty lines properly
				if just_words:
					temp.append(line.split()[1])
				else:
					temp.append(line.split())

	return temp

def make_new_features(data_set):
	feature_vectors = []
	print('Generating feature vectors...')
	left_words =            []
	right_words =           []
	left_word_is_short =    []
	left_capitalizations =  []
	right_capitalizations = []
	left_word_is_long =     [] # made up feature 1
	left_word_is_number =   [] # made up feature 2
	right_word_is_space = 	[] # made up feature 3
	targets =               []
	train_vocab = parse_data_set('SBD.train', just_words=True)
	test_vocab = parse_data_set('SBD.test', just_words=True)
	vocabulary_list = train_vocab + test_vocab
	vocabulary = []

	[vocabulary.append(i) for i in vocabulary_list if not vocabulary.count(i)]

	for index, point in enumerate(data_set):
		identifier = point[0]
		token = point[1].strip()
		label = point[2]
		targets.append(label)

		try:

			word = token.split('.')
			
			left_word = word[0]
			right_word = word[1]

			left_words.append(word[0])
			right_words.append(right_word    if right_word           else    '')
			left_word_is_short.append(1   if len(left_word) < 3   else 0)
			left_capitalizations.append(1  if left_word.isupper()  else 0)
			right_capitalizations.append(1 if right_word.isupper() else 0)
			left_word_is_long.append(1 if len(left_word) > 4 else 0)
			left_word_is_number.append(1 if left_word.isdigit() else 0)
			right_word_is_space.append(1 if right_word == "" else 0)


			#print(left_words[index], left_word_is_short[index])
		except Exception:
			print(Exception)

		one_hot_word_vector = [0] * len(vocabulary)

		pos = vocabulary.index(token)

		one_hot_word_vector[pos] = 1

		feature_vector = one_hot_word_vector
		feature_vector.append(left_word_is_short[index])
		feature_vector.append(left_capitalizations[index])
		feature_vector.append(right_capitalizations[index])
		feature_vector.append(left_word_is_long[index])
		feature_vector.append(left_word_is_number[index])
		feature_vector.append(right_word_is_space[index])
	
		#print(one_hot_word_vector, left_word_is_short[index], left_capitalizations[index], right_capitalizations[index])
		#print(feature_vector_expanded)
		#feature_vector = [item for sublist in feature_vector_expanded for item in sublist]
		feature_vectors.append(feature_vector)
 

	return (feature_vectors, targets)


def main():
	training_set     = parse_data_set(sys.argv[1])
	test_set         = parse_data_set(sys.argv[2])
	training_vectors, targets = make_new_features(training_set)
	test_vectors, test_targets     = make_new_features(test_set)

	classifier = tree.DecisionTreeClassifier()

	#print(training_vectors)

	#print('Training shape:', training_vectors.shape)
	#print('Target shape:', targets.shape)
	print('Training our decision tree...')
	classifier.fit(training_vectors, targets) # 0 = EOS, 1 = NEOS ?
	print('Training complete!')

	total_seen = 0
	total_correct = 0

	for i, test_example in enumerate(test_vectors):
		correct = test_targets[i]
		#print(test_example, correct)
		pred = classifier.predict(np.array(test_example).reshape(1,-1))

		#print(i, 'Predicted:', pred[0], 'Actual:', correct)

		if str(pred[0]) == str(correct):
			total_correct += 1
			total_seen += 1
		else:
			total_seen += 1 

		accuracy = (total_correct/total_seen)*100
		#print('Accuracy: ', ((total_correct/total_seen)*100))
	print('System Accuracy:', accuracy)

if __name__ == '__main__':
	main()