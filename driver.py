import os, math
from TextPreprocessor import *
from Term import *
from Query import *

def format_line(line):
	clean_line = line.rstrip().split('\t') # Remove trailing chars & tokenize line
	return clean_line

def transfer_data(query_dict):
	sorted_query_terms = [query_dict[term].get_mean_normalized_click_count() for term in sorted(query_dict.keys())]
	vocab = set([term for term in sorted(query_dict.keys())])

	return sorted_query_terms, vocab

def vectorize_query(query_dict, image_vec, query_data):
	len_img_vec = len(image_vec)
	query_vector = [[0.0] * len_img_vec] * len(query_data)
	tmp_keys_list = sorted(query_dict.keys())

	for l_idx, qt_list in enumerate(query_data):
		tmp_list = [0.0] * len_img_vec
		for qt in qt_list:
			idx = tmp_keys_list.index(qt.value)
			tmp_list[idx] = qt.get_normalized_click_count()
		query_vector[l_idx] = tmp_list

	return query_vector

if __name__ == '__main__':
	'''
		File I/O declarations
	'''
	training_file_path = 'TrainClick.txt'
	training_file = list(open(training_file_path, "r"))
	#results_file = open('image_vector.txt', 'w')
	#results_file_2 = open('vocab_map.txt', 'w')
	#results_file_3 = open('image_query_click_counts.txt', 'w')
	results_file_4 = open('query_vector.txt', 'w')

	image_ctr = 1
	current_img = ''

	preprocessor = TextPreprocessor() # Initalize text preprocessor

	'''
		Image related collections
	'''
	image_dict = dict()
	image_cc = list()

	'''
		Query related collections
	'''
	query_vecs = dict()
	query_tfs = dict()
	query_dict = dict()

	vocabulary = set()

	line_ctr = 0

	for x in xrange(0, len(training_file)+1):
		if x == 10000: # Stop and transfer last piece of data
			image_dict[current_img], tmp_vocab = transfer_data(query_dict)
			vocabulary.union(tmp_vocab)
			query_tfs[current_img] = vectorize_query(query_dict, image_dict[current_img], query_vecs[current_img])
			break

		clean_line = format_line(training_file[x])
		image_id, click_count  = clean_line[0], clean_line[2]
		query_terms = preprocessor.process_text(clean_line[1].rstrip()) # generates a list of tokens
		image_cc.append(click_count) # Store click counts for every image-query pair

		if image_id not in query_vecs:
			query_vecs[image_id] = list()

		term_list = [Term(qt, int(click_count)) for qt in query_terms]
		query_vecs[image_id].append(term_list)	# For each image, store a list of queries associated

		if image_id not in image_dict:
			image_dict[image_id] = list()
			if len(query_dict) > 0:
				# copy all query terms in the query_dict into image_dict
				image_dict[current_img], tmp_vocab = transfer_data(query_dict)
				vocabulary.union(tmp_vocab)
				query_tfs[current_img] = vectorize_query(query_dict, image_dict[current_img], query_vecs[current_img])

			query_dict = dict()
			current_img = image_id

		for term in query_terms:
			if term not in query_dict:
				query_dict[term] = Term(term)
			query_dict[term].add_click_count(int(click_count))

	'''
		vocab_index = list(sorted(vocabulary))

		# Print all unique words on to text file
		
		for word_voc in vocab_index:
			results_file_2.write(word_voc + "\n")
		results_file_2.close() # Close stream
0G5LtudP4n5DwQ
	    # Print out all the image-query click counts
		for cc in image_cc:
			results_file_3.write(str(cc) + "\n")
		results_file_3.close() # Close stream
	'''

	# Print out image as vector
	for img, terms in image_dict.items():
		for qv_term in query_tfs[img]:	# Print out query as a vector
			q_output = img
			q_output += '\t' + str(qv_term) + '\n'
			#print q_output
			results_file_4.write(q_output)
	results_file_4.close()

	'''
			output_line = img
			for term in terms:
				output_line += '\t' + str(term)
			output_line += '\n'
			results_file.write(output_line)
	'''

	#results_file.close() # Close stream
	#results_file_4.close()

	'''
		image_dict2 = dict()

		for img in image_dict.keys():
			image_dict2[img] = [Term()]*len(vocab_index)
			for term in image_dict[img]:
				idx = vocab_index.index(term.value)
				image_dict2[img][idx] = term

		id_ct = 1
		for img, terms in image_dict2.items():
			output_line = str(id_ct)
			id_ct += 1
			for term in terms:
				output_line += "\t" + str(term.get_frequency())
			output_line += "\n"
			results_file.write(output_line)
		
		results_file.close()
	'''