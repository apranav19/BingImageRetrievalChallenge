import os, math
from TextPreprocessor import *
from Term import *
from Query import *

def init_file():
	training_file_path = os.getcwd() + "/TrainClick.txt"
	return training_file_path

def clean_data(element):
	return element.rstrip()

if __name__ == '__main__':
	training_file_path = init_file()
	training_file = open(training_file_path, "r")
	results_file = open('results.txt', 'w')
	image_ctr = 1
	current_img = ''

	preprocessor = TextPreprocessor()
	image_dict = dict()
	vocabulary = set()
	query_dict = None
	
	for line in training_file:
		line = line.rstrip()
		clean_line = line.split("\t")
		image_id = clean_line[0]
		query_terms = preprocessor.process_text(clean_line[1].rstrip())
		click_count = clean_line[2]

		if image_id not in image_dict:
			image_dict[image_id] = list()
			# transfer query term info into image_dict
			if query_dict is not None:
				total_term_frequencies = sum([obj.get_frequency() for term, obj in query_dict.items()]) # Compute total term frequencies

				for term in sorted(query_dict.keys()): # Compute probabilities for every term and insert into image table
					vocabulary.add(term) # Add unique term to vocabulary set
					computed_probability = (1.0 * query_dict[term].get_frequency())/total_term_frequencies
					log_prob = math.log(computed_probability + 1.0)
					query_dict[term].set_probability(log_prob)
					image_dict[current_img].append(query_dict[term])
			else:
				query_dict = dict()

			current_img = image_id

		for term in query_terms:
			if term not in query_dict:
				query_dict[term] = Term(term)
			else:
				query_dict[term].increment_frequency()

	vocab_index = list(sorted(vocabulary))

	image_dict2 = dict()

	for img in image_dict.keys():
		image_dict2[img] = [Term()]*len(vocab_index)
		for term in image_dict[img]:
			idx = vocab_index.index(term.value)
			image_dict2[img].insert(idx, term)

	id_ct = 1
	for img, terms in image_dict2.items():
		output_line = str(id_ct)
		id_ct += 1
		for term in terms:
			output_line += "\t" + str(term.probability)
		output_line += "\n"
		results_file.write(output_line)

	results_file.close()