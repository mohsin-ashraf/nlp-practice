# Preprocessing the text data from any irrelevent tags.

from sklearn.feature_extraction.text import CountVectorizer

def preprocess(review_text):
	review_text = review_text.str.replace('<br />','')
	review_text = review_text.str.replace('(<a).*(>).*(</a>)','')
	review_text = review_text.str.replace('&amp','')
	review_text = review_text.str.replace('&gt','')
	review_text = review_text.str.replace('&lt','')
	review_text = review_text.str.replace('(\xa0)',' ')
	return review_text

def get_top_n_words(corpus,n=None):
	vectorizer = CountVectorizer().fit(corpus)
	bag_of_words = vectorizer.transform(corpus)
	sum_words = bag_of_words.sum(axis=0) # adding up the same words along all the vectors
	words_freq = [(word,sum_words[0,idx]) for word,idx in vectorizer.vocabulary_.items()]
	words_freq = sorted(words_freq,key=lambda x: x[1], reverse=True)
	return words_freq[:n]

def get_top_n_bigrams(corpus,n=None):
	vectorizer = CountVectorizer(ngram_range=(2,2)).fit(corpus)
	bag_of_words = vectorizer.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word,sum_words[0,idx]) for word, idx in vectorizer.vocabulary_.items()]
	words_freq = sorted(words_freq,key=lambda x: x[1], reverse=True)
	return words_freq[:n]

def get_top_n_trigrams(corpus,n=None):
	vectorizer = CountVectorizer(ngram_range=(3,3)).fit(corpus)
	bag_of_words = vectorizer.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word,sum_words[0,idx]) for word,idx in vectorizer.vocabulary_.items()]
	words_freq = sorted(words_freq,key=lambda x: x[1], reverse=True)
	return words_freq[:n]

