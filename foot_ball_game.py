import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import math

############ check data types of feature values ############
def check_data_types():
	features = train_df.head(1)
	features_list = list(features)
	print(len(features_list))
	for each in features_list:
		print(each, train_df[str(each)].dtypes) # object in pandas is a string


############################################################
def find_properties_of_data():
	HT = train_df['HomeTeam']
	HT_unique = HT.unique()
	print(len(HT_unique))
	print(type(HT_unique))

	AT = train_df['AwayTeam']
	AT_unique = AT.unique()
	print(len(AT_unique))
	print(type(AT_unique))

	HTAT = train_df['HomeTeam'] + '-' + train_df['AwayTeam']
	HTAT_unique = HTAT.unique()
	print(len(HTAT_unique))
	print(type(HTAT_unique))

	AT = train_df['league']
	AT_unique = AT.unique()
	print(len(AT_unique))
	print(type(AT_unique))


############################################################
def visualize_data():
	plt.figure(figsize=(20,10)) 
	sns_plot = sns.heatmap(train_df.corr(method='pearson'), annot= True)
	fig = sns_plot.get_figure()
	fig.savefig('corr_mat.png')

	from pandas.plotting import scatter_matrix
	scatter = scatter_matrix(train_df, diagonal = 'kde', figsize=(20, 20), alpha=0.5)
	plt.savefig('scatter.png')


############################################################
def data_preprocessing():
	'''
	Make dictionary of {'team name': 1, 'team name' : 2 ... 'FTR' : np.asarray([1, 0, 0])}
	Initialize 12786x16x159 nparray: all_train
	Initialize 12786x1x3 nparray: all_valid
	For every row in df:
		row 1: 1 on home team and 1 on away team and rest 0s
		row 2: 1 on home team and rest 0s (special reference to home team)
		For all 14 features(rows) except FTR : 159 cols each with corresponding number in home and away team and rest 0s. 
	
		all_valid will take [100], [010], [001] for (win, loose or draw) from df['FTR'].
	'''
	
	AT = train_df['AwayTeam']
	AT_unique = AT.unique()
	
	unique_item_dict = {}
	for i in range(len(AT_unique)):
		unique_item_dict[AT_unique[i]] = i
	unique_item_dict['H'] = [1, 0, 0]
	unique_item_dict['A'] = [0, 1, 0]
	unique_item_dict['D'] = [0, 0, 1]

	all_train = np.zeros((len(train_df), 16, 159))
	all_valid = np.zeros((len(train_df), 3))

	for i in train_df.index:
		for k in range(16): # For 16 rows of each data point
			value = unique_item_dict[train_df['HomeTeam'][i]]
			all_train[i][k][value] = 1

		all_valid[i] = unique_item_dict[train_df['FTR'][i]]

	# Train Valid data split
	X_train, X_valid, y_train, y_valid = train_test_split(all_train, all_valid, 
                                                    test_size = 0.3,
                                                    random_state = 2,
                                                    stratify = all_valid)
	print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

	return X_train, X_valid, y_train, y_valid

def recompute_for_sklearn():
	#scikit learns logistic regression handles only 1 dim array for each datapoint.
	a, b, c = X_train.shape
	X_train_1 = X_train.reshape((a, b*c))
	a, b, c = X_valid.shape
	X_valid_1 = X_valid.reshape((a, b*c))
	
	#scikit_learn's logistic regression could not take categorical values for lables.
	y_train_1 = np.zeros((len(y_train), 1))
	for i in range(len(y_train)):
		if y_train[i][0] == 1:
			y_train_1[i] = 1
		elif y_train[i][1] == 1:
			y_train_1[i] = 2
		elif y_train[i][2] == 1:
			y_train_1[i] = 3

	y_valid_1 = np.zeros((len(y_valid), 1))
	for i in range(len(y_valid)):
		if y_valid[i][0] == 1:
			y_valid_1[i] = 1
		elif y_valid[i][1] == 1:
			y_valid_1[i] = 2
		elif y_valid[i][2] == 1:
			y_valid_1[i] = 3

	return X_train_1, X_valid_1, y_train_1, y_valid_1

def find_class_weight():
	count_H = 0
	count_A = 0
	count_D = 0
	for i in range(len(y_train)):
		if y_train[i][0] == 1:
			count_H += 1
		elif y_train[i][1] == 1:
			count_A += 1
		elif y_train[i][2] == 1:
			count_D += 1
	print(count_H, count_A, count_D)

	# Inverse number of samples to deal with sample weights of multi class classification.
	total_samples = count_H + count_A + count_D
	H_weight = math.log(total_samples / count_H)
	A_weight = math.log(total_samples / count_A)
	D_weight = math.log(total_samples / count_D)
	print(H_weight, A_weight, D_weight)

	class_weights = {1: H_weight, 2: A_weight, 3: D_weight}
	
	return class_weights

def analyse_results(y_valid_1, Y_pred, file_name):
	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(y_valid_1, Y_pred)
	sns_plot = sns.heatmap(cm, annot=True,fmt='d')
	fig = sns_plot.get_figure()
	fig.savefig(file_name)

def logistic_regression(X_train, X_valid, y_train, y_valid):
	from sklearn.linear_model import LogisticRegression
	# With equal class weights
	classifier = LogisticRegression(random_state = 0)
	print(vars(classifier))
	eq_wt_classifier = classifier.fit(X_train_1, y_train_1) # Sample weight of each sample is equal.
	Y_pred = eq_wt_classifier.predict(X_valid_1)
	file_name = 'cm_lr_eq_wt.png'
	analyse_results(y_valid_1, Y_pred, file_name)

	# Class weight follows INS (Inverse of number of samples).
	Y_pred = class_wt_classifier.predict(X_valid_1)
	classifier = LogisticRegression(random_state = 0, class_weight=class_weights)
	class_wt_classifier = classifier.fit(X_train_1, y_train_1) 
	file_name = 'cm_lr_class_wt'
	analyse_results(y_valid_1, Y_pred, file_name)

def random_forest():
	from sklearn.ensemble import RandomForestClassifier
	# classifier = RandomForestClassifier(criterion='gini', 
 #                             n_estimators=700,
 #                             min_samples_split=10,
 #                             min_samples_leaf=1,
 #                             max_features='log2',
 #                             oob_score=True,
 #                             random_state=1,
 #                             n_jobs=-1)
	# classifier.fit(X_train_1, y_train_1)
	# Y_pred = classifier.predict(X_valid_1)
	# file_name = 'cm_rf_max_features_log2'
	# analyse_results(y_valid_1, Y_pred, file_name)

	# Class weight follows INS (Inverse of number of samples).
	class_weights = {1: 1, 2: 2, 3: 2}
	classifier = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='log2',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1,
                             class_weight=class_weights)
	class_wt_classifier = classifier.fit(X_train_1, y_train_1) 
	Y_pred = class_wt_classifier.predict(X_valid_1)
	file_name = 'cm_rf_class_wt_1'
	analyse_results(y_valid_1, Y_pred, file_name)

def compute_sample_weights(class_weight):
	sample_weights = []
	for i in y_train_1:
		if i == 1:
			sample_weights.append(class_weight[1])
		elif i == 2:
			sample_weights.append(class_weight[2])
		elif i == 3:
			sample_weights.append(class_weight[3])

	return sample_weights

def xgboost():
	from xgboost import XGBClassifier
	classifier = XGBClassifier(colsample_bytree=0.8,
              gamma=0.4,
              min_child_weight=3,
              n_estimators=40,
              reg_alpha=1e-05,
              seed=2,
              subsample=0.8)

	a = classifier.fit(X_train_1, y_train_1, sample_weight=sample_weights)
	Y_pred = classifier.predict(X_valid_1)
	file_name = 'cm_xgboost_2_cl_wts'
	analyse_results(y_valid_1, Y_pred, file_name)

def svm():
	from sklearn.svm import SVC
	classifier = SVC(kernel = 'rbf',random_state = 0)
	classifier.fit(X_train_1, y_train_1, sample_weight=sample_weights)
	Y_pred = classifier.predict(X_valid_1)
	file_name = 'cm_SVM_cl_wt'
	analyse_results(y_valid_1, Y_pred, file_name)

def neural_networks(): #LSTM, Bidirectional LSTM and RNN
	### Build model ###
	import tensorflow as tf
	from tensorflow import keras
	from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, SimpleRNN
	from tensorflow.keras.models import Model

	input_shape = (X_train_1[0].shape)
	input_tensor = Input(input_shape)
	
	x_1 = Embedding(input_dim=2544, output_dim=100)(input_tensor)
	x_2 = SimpleRNN(128)(x_1)
	output = Dense(3, activation='softmax')(x_2)
	model = Model(inputs=[input_tensor], outputs=[output])

	model.summary()

	### Training part ###
	model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

	epochs = 2
	batch_size = 10

	class_weights = {0: 1, 1: 2, 2: 2}
	history = model.fit(X_train_1[:10], y_train[:10], epochs=epochs, batch_size=batch_size, 
		validation_split = 0.2, class_weight = class_weights)

	# plt.title('Loss')
	# plt.plot(history.history['loss'], label='train')
	# plt.plot(history.history['val_loss'], label='test')
	# plt.legend()
	# plt.show()

	# num_batches = int(len(X_train_1)/batch_size)

	# train_loss_per_epoch = open('./LSTM/train_loss_per_epoch.txt', 'a')
	# train_acc_per_epoch = open('./LSTM/train_acc_per_epoch.txt', 'a')
	# valid_loss_per_epoch = open('./LSTM/valid_loss_per_epoch.txt', 'a')
	# valid_acc_per_epoch = open('./LSTM/valid_acc_per_epoch.txt', 'a')
	# for i in range(epochs):
	# 	batch_loss_per_epoch = 0
	# 	batch_acc_per_epoch = 0
	# 	for batch in range(num_batches):
	# 		batch_train_data = np.zeros((batch_size, len(X_train_1[0])))
	# 		batch_train_label = np.zeros((batch_size, 3))
	# 		j = 0
	# 		for k in range(batch*batch_size, min((batch+1) * batch_size, len(X_train_1))):
	# 			batch_train_data[j,:] = X_train_1[k]
	# 			batch_train_label[j,:] = y_train[k]
	# 			j += 1

	# 		loss, acc = model.train_on_batch(batch_train_data, batch_train_label)

	# 		# print(('epoch_num: %d batch_num: %d accuracy: %f\n' % (i, batch, acc)))
	# 		batch_loss_per_epoch += loss
	# 		batch_acc_per_epoch += acc

	# 	train_loss_per_epoch.write("%f \n" %(batch_loss_per_epoch / num_batches))
	# 	train_acc_per_epoch.write("%f \n" %(batch_acc_per_epoch / num_batches))

	# 	### Valildation at every epoch ###
	# 	valid_loss, valid_acc = model.evaluate(X_valid_1, y_valid)
	# 	valid_loss_per_epoch.write("%f \n" %valid_loss)
	# 	valid_acc_per_epoch.write("%f \n" %valid_acc)

	# 	print(valid_acc)

		


############################################################
train_csv_path = "/home/ada/Preethi/ML_Interviews/Football/train.csv"
train_df = pd.read_csv(train_csv_path)
# find_properties_of_data()
# visualize_data()
X_train, X_valid, y_train, y_valid = data_preprocessing() #y_valid is categorical input
class_weights = find_class_weight()
X_train_1, X_valid_1, y_train_1, y_valid_1 = recompute_for_sklearn() #y_valid_1 is int input
sample_weights = compute_sample_weights({1: 1.5, 2: 2.5, 3: 3})

# logistic_regression()
# random_forest()
# xgboost()
# svm()
X_train_1 = np.append(X_train_1, X_valid_1, 0)
y_train = np.append(y_train, y_valid, 0)
neural_networks()