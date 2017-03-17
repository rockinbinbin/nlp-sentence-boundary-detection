#Robin Mehta
#Collocations
from __future__ import division
import sys
import re
import operator
from scipy.stats import chisquare
import math

def ingest_corpus(path):

	raw_corpus = open(path, 'r').read().split(' ')

	raw_corpus = map(lambda t: t.strip(), raw_corpus)
	
	ngrams = {
		'unigrams': [],
		'bigrams': []
	}

	for token in raw_corpus:
		should_keep = not re.match('\W+', token)
		if should_keep and token is not '':
			ngrams['unigrams'].append(token)


	for index, token in enumerate(ngrams['unigrams']):
		window_start = index
		window_end = index+2

		ngrams['bigrams'].append(ngrams['unigrams'][window_start:window_end])

	ngrams['bigrams'].pop()

	return ngrams

def count_ngrams(ngrams):
	unigram_hashtable = {}
	bigram_hashtable  = {}

	for ngram in ngrams['unigrams']:
		if ngram not in unigram_hashtable:
			unigram_hashtable[ngram] = 1
		else:
			unigram_hashtable[ngram] += 1
	
	for ngram in ngrams['bigrams']:
		
		joined = ngram[0] + ' ' + ngram[1]

		if joined not in bigram_hashtable:
			bigram_hashtable[joined] = 1
		else:
			bigram_hashtable[joined] += 1

	#sorted_unigram_counts = sorted(unigram_hashtable.items(), key=operator.itemgetter(1))
	#sorted_bigram_counts = sorted(bigram_hashtable.items(), key=operator.itemgetter(1))

	return {'unigram_counts':unigram_hashtable, 'bigram_counts':bigram_hashtable } 

def compute_chi_square(freq_table):
	total_bigrams = sum(freq_table['bigram_counts'].values())
	#print(bigram_count, len(freq_table['bigram_counts'].keys()))
	count = 1
	scores = {}

	for key in freq_table['bigram_counts'].keys():
		bigram_count = freq_table['bigram_counts'][key]
		word_A = key.split()[0]
		word_B = key.split()[1] 
		word_A_count = freq_table['unigram_counts'][word_A]
		word_B_count = freq_table['unigram_counts'][word_B]

		#chi-square columns
		A = bigram_count
		B = word_A_count  - bigram_count
		C = word_B_count  - bigram_count
		D = total_bigrams - bigram_count

		expected = (((A + B) / total_bigrams) * ((A + C) / total_bigrams)) * total_bigrams

		chi_square = ((A - expected)**2) / expected

		scores[key] = chi_square

		#print('Bigram ', count, 'Score: ', chi_square)

		count += 1

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=False)[:20]

	for item in sorted_scores:
		print('Bigram: %s Score: %s' % (item[0], item[1]))

def compute_pmi(freq_table):
	total_bigrams = sum(freq_table['bigram_counts'].values())
	count = 1
	scores = {}

	for key in freq_table['bigram_counts'].keys():
		bigram_count = freq_table['bigram_counts'][key]
		word_A = key.split()[0]
		word_B = key.split()[1] 
		word_A_count = freq_table['unigram_counts'][word_A]
		word_B_count = freq_table['unigram_counts'][word_B]

		top = bigram_count
		bottom = word_A_count * word_B_count
		scores[key] = math.log(top/bottom)
		count += 1

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=False)[:20]

	for item in sorted_scores:
		print('Bigram: %s Score: %s' % (item[0], item[1]))

def main():
	measure = sys.argv[2]
	ngrams = ingest_corpus(sys.argv[1])
	ngram_counts = count_ngrams(ngrams)

	if measure == 'PMI':
		compute_pmi(ngram_counts)
	else:
		compute_chi_square(ngram_counts)

if __name__ == '__main__':
	main()