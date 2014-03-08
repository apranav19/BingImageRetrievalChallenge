from stemming.porter2 import stem
from nltk.corpus import stopwords

class TextPreprocessor(object):

	def __init__(self):
		self.stopwords_list = stopwords.words('english')

	def stem_query_term(self, query_term):
		stemmed_term = stem(query_term)
		return stemmed_term

	def stem_query_terms(self, query_terms):
		stemmed_terms = map(self.stem_query_term, query_terms)
		return stemmed_terms

	def remove_stop_words(self, query_terms):
		filtered_terms = [term for term in query_terms if term not in self.stopwords_list]
		return filtered_terms

	def process_text(self, query_string):
		query_terms = query_string.split()
		# Remove stopwords
		filtered_terms = self.remove_stop_words(query_terms)
		# Stem filtered words
		stemmed_terms = self.stem_query_terms(filtered_terms)

		return stemmed_terms
