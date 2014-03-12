import math
from scipy.spatial.distance import cosine, jaccard, cityblock
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
from numpy import linalg
import pylab as pl 


binarizer = preprocessing.Binarizer()
linear_reg = LinearRegression()

def compute_cosine_similarity(query, image): # Computes the cosine similarity
	cosine_sim = np.vdot(query, image)
	norm_q = linalg.norm(query) + 1
	norm_i = linalg.norm(image) + 1
	return cosine_sim/(norm_q * norm_i)

def compute_jaccard_distance(query, image):  # Computes the Jaccard  dissimilarity
	q_bin = binarizer.transform(query)
	i_bin = binarizer.transform(image)
	jaccard_distance = jaccard(q_bin, i_bin)
	return jaccard_distance

def compute_probability(query, image):
	term_likelihood = 1
	for idx, q in enumerate(query):
		tf_q = image[idx]
		total_tfs = sum(image)
		term_likelihood *= (tf_q/total_tfs)

	return term_likelihood

def map_log_transform(el):
	return math.log(el + 1, 10)

def log_transform(matrix):
	for i, instance in enumerate(matrix):
		matrix[i] = map(map_log_transform, instance)

	return matrix

def map_flatten(el):
	return 1 if el <= 1 else el

def res_flatten(matrix):
	for i, instance in enumerate(matrix):
		matrix[i] = map(map_flatten, instance)
	return matrix

if __name__ == '__main__':
	image_v_file = open('image_vector.txt', 'r')
	image_v_file2 = open('image_vector3.txt', 'r')
	query_v_file2 = open('query_vector2.txt', 'r')
	query_v_file = open('query_vector.txt', 'r')
	iq_click_file = open('image_query_click_counts.txt', 'r')

	res_zscore = open('result_zscore_with_log_likelihood_only_features.txt', 'w')

	image_v = dict() # Image Vector Collection
	image_v2 = dict()
	query_v2 = dict()
	query_v = dict() # Query Vector Collection
	image_query_click_counts = dict() # Image-Query click counts

	training_matrix = np.zeros((10000, 6))

	cosine_sim_table = list()
	for line in image_v_file: # Load all image vectors
		line = line.rstrip().split('\t')
		image_id = line[0]
		image_v2[image_id] = [float(t) for t in line[1].split()]
		image_v[image_id] = [float(t) for t in line[1].split()]
	image_v_file.close() # Close image_vector file
	image_v_file2.close()

	for line in query_v_file: # Load all query vectors
		line = line.rstrip().split('\t')
		image_id = line[0]

		if image_id not in query_v:
			query_v[image_id] = list()
		query_v[image_id].append([float(t) for t in line[1].split()])
	query_v_file.close() # Close query vector file

	for line in query_v_file2: # Load all query vectors
		line = line.rstrip().split('\t')
		image_id = line[0]

		if image_id not in query_v2:
			query_v2[image_id] = list()
		query_v2[image_id].append([float(t) for t in line[1].split()])
	query_v_file2.close()

	for line in iq_click_file: # Load all img_query_ccs
		line = line.rstrip().split('\t')
		image_id = line[0]

		if image_id not in image_query_click_counts:
			image_query_click_counts[image_id] = list()
		image_query_click_counts[image_id] = [int(cc) for cc in line[1].split()]
	iq_click_file.close() # Close img_query_ccs file

	row_ct = 0
	for img_id, img_vec in sorted(image_v.items()):
		click_count_list = image_query_click_counts[img_id]
		#popularity = len(click_count_list) # Num of queries associated with this image_id
		for idx, qv in enumerate(query_v[img_id]):
			click_count = click_count_list[idx]
			cos_sim = compute_cosine_similarity(qv, img_vec)
			jaccard_distance = compute_jaccard_distance(qv, img_vec)
			query_likelihood = compute_probability(query_v2[img_id][idx], image_v2[img_id])
			#city_block = cityblock(qv, img_vec)
			popularity = len(query_v[img_id])
			#popularity2 = sum(click_count_list)/len(img_vec)
			#popularity3 = len(image_v2[img_id])/(sum(image_v2[img_id]))
			qv_bin = binarizer.transform(query_v2[img_id][idx])
			sum4 = sum(qv_bin)
			if sum(qv_bin) == 0:
				sum4 = len(qv_bin)/2
			popularity4 = (click_count_list[idx])/sum4
			training_matrix[row_ct] = [cos_sim, jaccard_distance, query_likelihood, popularity, popularity4, click_count]
			row_ct += 1

	training_results = training_matrix[:, 5:]
	#test_results = training_matrix
	training_data = training_matrix[:, :5]
	training_data = log_transform(training_data)
	#training_data = preprocessing.scale(training_matrix[:, :4])

	linear_reg.fit(training_data, training_results)

	res_matrix = linear_reg.predict(training_data)
	#training_results = log_transform(training_results)
	res_matrix = res_flatten(res_matrix)
	#res_matrix = log_transform(res_matrix)
	actual_max = max(training_results)
	actual_min = min(training_results)

	axes = pl.gca()
	axes.set_xlim(0, 500)
	axes.set_ylim(-2, 800)

	pl.plot(np.array(range(10000)), training_results)
	pl.hold(True)
	pl.plot(np.array(range(10000)), res_matrix, color='red')
	pl.show()
	

	#print linear_reg.predict(training_data[7])
	
	''' 
		for row in res_matrix:
		output_line = str(row[0]) + '\n'
		res_zscore.write(output_line)
	'''
