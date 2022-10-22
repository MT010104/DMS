<<<<<<< HEAD
import os
import sys
import argparse

import keras
import numpy as np
from tqdm import tqdm
import pandas as pd
from keras.optimizers import SGD, adadelta

import tools.data_utils as DataUtils
import tools.model_utils as ModelUtils

if __name__ == "__main__":

	# os.environ["CUDA_VISIBLE_DEVICES"] = ""
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", "-d", help="Dataset", type=str, default="mnist")
	parser.add_argument("--network", "-n", help="Network", type=str, default="lenet5")

	args = parser.parse_args()
	dataset = args.dataset
	network = args.network

	for target in ['nature', 'cw', 'bim', 'fgsm', 'jsma']:

		x_test, y_test = DataUtils.get_candidate_general(target=target, name=dataset, network=network)
		all_sample_size = x_test.shape[0]
		model = ModelUtils.load_model(network=network, dataset=dataset)
		y_predict = np.argmax(model.predict(x_test, verbose=0), axis=1)

		x_true = []
		x_false = []
		for i in range(all_sample_size):
			if y_test[i] == y_predict[i]:
				x_true.append(i)
			else:
				x_false.append(i)

		trueOrFalse = []
		for i in range(all_sample_size):
			if y_test[i] == y_predict[i]:
				trueOrFalse.append(0)
			else:
				trueOrFalse.append(1)


		res_dir = f"finetuned_prediction/{dataset}_{network}"
		for iteration in range(40):

			combined_prediction = []
			for iter in range(iteration):
				if target == 'nature':
					prediction = np.load(f"{res_dir}/iter_{iter}_models/prediction.npy")
				else:
					prediction = np.load(f"{res_dir}/iter_{iter}_models/prediction_target_{target}.npy")
				# print(prediction.shape)
				if np.any(combined_prediction):
					combined_prediction = np.vstack((combined_prediction, prediction))
				else:
					combined_prediction = prediction

			mutate_matrix = []
			for i, item in tqdm(enumerate(combined_prediction)):
				row = np.zeros(all_sample_size, np.int)
				for j in range(all_sample_size):
					if y_predict[j] != item[j]:
						row[j] = 1
				mutate_matrix.append(row)
			mutate_matrix = np.array(mutate_matrix)
			print(mutate_matrix.shape)

			save_dir = f"finetuned_prediction/{dataset}_{network}"
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			np.save(f"{save_dir}/prediction_{target}.npy", mutate_matrix)

			prediction_trans = []
			for item in tqdm(combined_prediction.T):
				label_count = np.zeros(10)
				for label in item:
					label_count[label] += 1
				label_count /= np.sum(label_count)
				prediction_trans.append(label_count)
			np.save(f'finetuned_prediction/{dataset}_{network}/prediction_trans_{target}.npy', prediction_trans)
=======
import os
import sys
import argparse

import keras
import numpy as np
from tqdm import tqdm
import pandas as pd
from keras.optimizers import SGD, adadelta

import tools.data_utils as DataUtils
import tools.model_utils as ModelUtils

if __name__ == "__main__":

	# os.environ["CUDA_VISIBLE_DEVICES"] = ""
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", "-d", help="Dataset", type=str, default="mnist")
	parser.add_argument("--network", "-n", help="Network", type=str, default="lenet5")

	args = parser.parse_args()
	dataset = args.dataset
	network = args.network

	for target in ['nature', 'cw', 'bim', 'fgsm', 'jsma']:

		x_test, y_test = DataUtils.get_candidate_general(target=target, name=dataset, network=network)
		all_sample_size = x_test.shape[0]
		model = ModelUtils.load_model(network=network, dataset=dataset)
		y_predict = np.argmax(model.predict(x_test, verbose=0), axis=1)

		x_true = []
		x_false = []
		for i in range(all_sample_size):
			if y_test[i] == y_predict[i]:
				x_true.append(i)
			else:
				x_false.append(i)

		trueOrFalse = []
		for i in range(all_sample_size):
			if y_test[i] == y_predict[i]:
				trueOrFalse.append(0)
			else:
				trueOrFalse.append(1)


		res_dir = f"finetuned_prediction/{dataset}_{network}"
		for iteration in range(40):

			combined_prediction = []
			for iter in range(iteration):
				if target == 'nature':
					prediction = np.load(f"{res_dir}/iter_{iter}_models/prediction.npy")
				else:
					prediction = np.load(f"{res_dir}/iter_{iter}_models/prediction_target_{target}.npy")
				# print(prediction.shape)
				if np.any(combined_prediction):
					combined_prediction = np.vstack((combined_prediction, prediction))
				else:
					combined_prediction = prediction

			mutate_matrix = []
			for i, item in tqdm(enumerate(combined_prediction)):
				row = np.zeros(all_sample_size, np.int)
				for j in range(all_sample_size):
					if y_predict[j] != item[j]:
						row[j] = 1
				mutate_matrix.append(row)
			mutate_matrix = np.array(mutate_matrix)
			print(mutate_matrix.shape)

			save_dir = f"finetuned_prediction/{dataset}_{network}"
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			np.save(f"{save_dir}/prediction_{target}.npy", mutate_matrix)

			prediction_trans = []
			for item in tqdm(combined_prediction.T):
				label_count = np.zeros(10)
				for label in item:
					label_count[label] += 1
				label_count /= np.sum(label_count)
				prediction_trans.append(label_count)
			np.save(f'finetuned_prediction/{dataset}_{network}/prediction_trans_{target}.npy', prediction_trans)
>>>>>>> 1dffd00419da92d924b616008c876798ac08764e
