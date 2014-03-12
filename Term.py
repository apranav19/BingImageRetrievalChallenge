import math

class Term(object):
	def __init__(self, value=None, click_count=None):
		self.value = value
		self.click_count = click_count
		self.frequency = 0
		self.probability = 0.0
		self.click_counts = list()
		self.term_factor = 0.0

	def increment_frequency(self):
		self.frequency += 1

	def get_probability(self):
		return self.probability

	def set_probability(self, probability):
		self.probability = probability

	def set_frequency(self, frequency):
		self.frequency = frequency

	def increment_frequency(self):
		self.frequency += 1

	def get_frequency(self):
		return self.frequency

	def get_click_counts(self):
		return self.click_counts

	def add_click_count(self, cc):
		self.click_counts.append(cc)

	def get_mean_normalized_click_count(self):
		sum_ccs = sum([math.log(cc + 1, 10) for cc in self.click_counts])
		mean = sum_ccs/len(self.click_counts)
		return mean

	def get_normalized_click_count(self):
		norm_cc = math.log(self.click_count + 1, 10)
		return norm_cc