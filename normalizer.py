import numpy as np
from sklearn import preprocessing

if __name__ == '__main__':
	data_file = "image_term_probs.txt"
	out_file = "std_image_term_probs.txt"
	matrix = np.genfromtxt(data_file)

	#min_max_scaler = preprocessing.MinMaxScaler()
	#pre_processed_matrix = min_max_scaler.fit_transform(training_matrix)


	np.savetxt(out_file, training_matrix)

	#pre_processed_matrix = preprocessing.scale(training_matrix)

	#np.savetxt(out_file, pre_processed_matrix)

