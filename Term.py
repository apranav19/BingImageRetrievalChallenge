class Term(object):
	def __init__(self, value=None):
		self.value = value
		self.frequency = 1
		self.probability = 0.0

	def increment_frequency(self):
		self.frequency += 1

	def get_probability(self):
		return self.probability

	def set_probability(self, probability):
		self.probability = probability

	def get_frequency(self):
		return self.frequency