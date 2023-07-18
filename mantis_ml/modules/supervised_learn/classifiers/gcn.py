import pandas as pd
import numpy as np
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import keras

import keras.backend as K

from tensorflow.keras.utils import to_categorical


from mantis_ml.config_class import Config
from mantis_ml.modules.supervised_learn.core.prepare_train_test_sets import PrepareTrainTestSets
from mantis_ml.modules.supervised_learn.classifiers.generic_classifier import GenericClassifier
from mantis_ml.modules.supervised_learn.classifiers.ensemble_lib import ensemble_clf_params

import sklearn

import networkx as nx
from networkx import DiGraph

from tensorflow.keras import layers, optimizers, losses, metrics, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping

import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.data import UnsupervisedSampler, BiasedRandomWalk
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator, FullBatchNodeGenerator, RelationalFullBatchNodeGenerator
from stellargraph.layer import Node2Vec, link_classification, GCN, GAT, RGCN


from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt

import random

# import bikg
# from grex.features import FeatureSet
# from grex.store_client import Store

import stellargraph
import pickle
from sklearn.metrics import roc_auc_score


class GCNClassifier(GenericClassifier):

	def __init__(self, cfg, X_train, y_train, X_test, y_test,
				 train_gene_names, test_gene_names, data, bikg_subgraph, clf_params, clf_id):


		GenericClassifier.__init__(self, cfg, X_train, y_train, X_test, y_test, train_gene_names, test_gene_names, data, clf_id)
		
		if clf_id=='GCN':
			self.epochs = self.cfg.n_epochs
			self.learning_rate = self.cfg.learning_rate
			self.n_filters = self.cfg.n_filters
			self.dropout_ratio = self.cfg.dropout_ratio
			
			self.regl = clf_params['regl']
			self.hidden_layer_nodes = clf_params['hidden_layer_nodes']
			self.add_dropout = clf_params['add_dropout']
			self.optimizer = clf_params['optimizer']
			self.num_hidden_layers = len(clf_params['hidden_layer_nodes'])
			self.batch_size = clf_params['batch_size']
		elif clf_id=='SGCN':
			self.SGCN_max_iter = clf_params['max_iter']
			self.SGCN_C = self.cfg.SGCN_C
			self.SGCN_one_max = self.cfg.SGCN_one_max
			self.SGCN_summation = self.cfg.SGCN_summation
		
		self.n_layers = self.cfg.n_layers
		self.verbose = clf_params['verbose']
		self.make_plots = clf_params['make_plots']
		self.model = None
		self.conc_pred_df = None
		self.clf_id = clf_id
		#genes = set(list(bikg_subgraph.source_label.values)).union(set(list(bikg_subgraph.target_label.values)))
		genes = set(list(train_gene_names.values)).union(set(list(test_gene_names.values)))
		
		
		
		self.bikg_subgraph = bikg_subgraph[bikg_subgraph.source_label.isin(genes) & bikg_subgraph.target_label.isin(genes)].copy()
		self.data = data


	def build_model(self):
		
		# node_features = pd.concat([
		# 	pd.DataFrame(self.X_train, index = self.train_gene_names.gene_name.values), 
		# 	pd.DataFrame(self.X_test, index = self.test_gene_names.gene_name.values)
		# ], axis=0)
		
		node_features = self.data.copy()
		print(node_features.shape)        
		node_features.index = node_features.Gene_Name.values
		node_features = node_features.drop(['known_gene', 'Gene_Name'], axis=1)
		node_features = (node_features-node_features.mean())/node_features.std()
		nodes_dict = {}
		
		
		all_genes = list(set(self.train_gene_names.values).union(self.test_gene_names.values))
		
		
		
		
		nodes_dict["Gene Target"] = node_features[node_features.index.isin(all_genes)]
		
		
		if 'relation' in self.bikg_subgraph.columns.values and 'relation_label' not in self.bikg_subgraph.columns.values: 
			self.bikg_subgraph = self.bikg_subgraph.rename(columns = {'relation': 'relation_label'})
		
		
		
		
		# Code for SGCN using StellarGraph
# 		generator = FullBatchNodeGenerator(G, method="sgc", k=1)
# 		print(self.data.Gene_Name.values[in_training].shape, train_binary_targets.shape)
# 		print(self.data.Gene_Name.values[in_testing].shape, test_binary_targets.shape)
# 		train_gen = generator.flow(self.data.Gene_Name.values[in_training], train_binary_targets)
# 		test_gen = generator.flow(self.data.Gene_Name.values[in_testing], test_binary_targets)
# 		sgc = GCN(layer_sizes=[train_binary_targets.shape[1]],generator=generator,bias=True,dropout=0.5,activations=["sigmoid"],kernel_regularizer=regularizers.l2(1e-5))
# 		x_inp, predictions = sgc.in_out_tensors()
# 		gcn_model = Model(inputs=x_inp, outputs=predictions)
# 		#gcn_model.compile(optimizer=optimizers.Adam(learning_rate=0.1),loss=losses.binary_crossentropy,metrics=["acc"])
# 		gcn_model.compile(optimizer=optimizers.Ftrl(learning_rate=self.learning_rate),loss=losses.binary_crossentropy, metrics=["accuracy"])
# 		gcn_history = gcn_model.fit(train_gen,epochs=self.epochs,verbose=0,shuffle=False)
# 		gcn_model.evaluate(train_gen)
		
		in_training = self.data.Gene_Name.isin(self.train_gene_names.values)
		in_testing = self.data.Gene_Name.isin(self.test_gene_names.values)#~in_training
		
		if self.clf_id=='SGCN':
			train_gene_names, test_gene_names = self.data.Gene_Name[in_training], self.data.Gene_Name[in_testing]
			all_gene_vec = np.hstack([train_gene_names, test_gene_names])
			n_train = len(train_gene_names)

			X = node_features.loc[train_gene_names].values
			y_dash = self.data.set_index('Gene_Name').known_gene.loc[train_gene_names]

			#A = np.asarray(G.to_adjacency_matrix(train_gene_names).todense())#np.minimum(,1)
			
			
			def get_A(subgraph, gene_names):
				train_subgraph = subgraph[subgraph.source_label.isin(gene_names) & subgraph.target_label.isin(gene_names)]
				A_df = pd.crosstab(train_subgraph.source_label, train_subgraph.target_label)
				idx = gene_names
				A_df = A_df.reindex(index = idx, columns=idx, fill_value=0)
				A_dash = A_df.values
				A_dash = A_dash + A_dash.T
				A_dash = A_dash.astype(float)
				if self.SGCN_one_max:
					A_dash = np.minimum(A_dash, 1)
				return A_dash
			
			def get_S_mat(A):
				A_tilde = A + np.diag(np.ones(len(A)))
				D_tilde_sqrt_inv = np.diag(1/(A_tilde.sum(0)**0.5))
				if self.SGCN_summation:
					S = A_tilde
				else:
					S = D_tilde_sqrt_inv.dot(A_tilde).dot(D_tilde_sqrt_inv)
				return S
			
			A = get_A(self.bikg_subgraph, train_gene_names)
			S_mat = get_S_mat(A)
			X_dash = np.linalg.matrix_power(S_mat, self.n_layers).dot(X)
			clf = LogisticRegression(C=self.SGCN_C,fit_intercept=True, max_iter=self.SGCN_max_iter)
			clf.fit(X_dash,y_dash)
			self.model = clf
			
			
			X_t = node_features.loc[all_gene_vec].values
			#A_test = np.asarray(G.to_adjacency_matrix(all_gene_vec).todense())#np.minimum(,1)
			A_test = get_A(self.bikg_subgraph, all_gene_vec)
			
			S_mat_dash = get_S_mat(A_test)
			X_dash_test = np.linalg.matrix_power(S_mat_dash, self.n_layers).dot(X_t)[n_train:,:]
			y_dash_test = self.data.set_index('Gene_Name').known_gene.loc[all_gene_vec][n_train:]

			train_acc = [0,clf.score(X_dash,y_dash)]
			test_acc = [0,clf.score(X_dash_test,y_dash_test)]

			y_pred = np.array(to_categorical(np.round(clf.predict(X_dash_test)).astype(int).ravel(), 2))
			y_prob = clf.predict_proba(X_dash_test)
		elif self.clf_id=='GCN':
			
			G = StellarGraph(
				nodes=nodes_dict,
				edges=self.bikg_subgraph,
				source_column="source_label",
				edge_type_column="relation_label",
				target_column="target_label"
			)


			target_encoding = preprocessing.LabelBinarizer()
			train_binary_targets = target_encoding.fit_transform(self.data.known_gene.values[in_training])
			test_binary_targets = target_encoding.transform(self.data.known_gene.values[in_testing])
			
			
			# Code for GCN using StellarGraph
			generator = FullBatchNodeGenerator(G, method="gcn")
			print(self.data.Gene_Name.values[in_training].shape, train_binary_targets.shape)
			print(self.data.Gene_Name.values[in_testing].shape, test_binary_targets.shape)
			train_gen = generator.flow(self.data.Gene_Name.values[in_training], train_binary_targets)
			test_gen = generator.flow(self.data.Gene_Name.values[in_testing], test_binary_targets)

			# train_gen = generator.flow(train_gene_names.values, train_binary_targets)
			# test_gen = generator.flow(data.Gene_Name.values[in_testing], test_binary_targets)

			gcn = GCN(layer_sizes=[self.n_filters]*self.n_layers, activations=["relu"]*self.n_layers, generator=generator, dropout=self.dropout_ratio)
			gcn_x_in, gcn_x_out = gcn.in_out_tensors()
			hidden = layers.Dense(units=32, activation="relu")(gcn_x_out)
			predictions = layers.Dense(units=train_binary_targets.shape[1], activation="sigmoid")(hidden)
			gcn_model = Model(inputs=gcn_x_in, outputs=predictions)
			gcn_model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),loss=losses.binary_crossentropy, metrics=["accuracy"])
			gcn_history = gcn_model.fit(train_gen, epochs=self.epochs,  verbose=0, shuffle=True, use_multiprocessing=False, workers=1)
			
			y_pred = np.array(to_categorical(np.round(gcn_model.predict(test_gen)).astype(int).ravel(), 2))
			y_prob = pd.DataFrame(gcn_model.predict(test_gen).ravel(), columns = ['positive']).assign(negative = lambda x: [1 - y for y in x.positive.values])[['negative', 'positive']].values
			train_acc = gcn_model.evaluate(train_gen, verbose = 0)
			test_acc = gcn_model.evaluate(test_gen, verbose = 0)
		else:
			raise ValueError('Invalid GCN specified')
		
		self.y_pred = y_pred 
		self.y_prob = y_prob 
		
		self.conc_pred_df = pd.concat([pd.DataFrame(np.array(to_categorical(self.data.known_gene.values[in_testing], 2))), pd.DataFrame(self.y_prob)], axis=1)
		# conc_pred_df = pd.concat([pd.DataFrame(np.array(to_categorical(data.known_gene.values[in_testing], 2))), pd.DataFrame(y_prob)], axis=1)
		self.conc_pred_df.columns = ['test_0', 'test_1', 'pred_0', 'pred_1']
		
		self.train_acc = train_acc 
		self.test_acc = test_acc

		self.train_acc = round(self.train_acc[1] * 100, 2)
		self.test_acc = round(self.test_acc[1] * 100, 2)

		y_test_roc = self.data.known_gene.values[in_testing]
		y_pred_roc = self.y_prob[:, 1]
		self.auc_score = roc_auc_score(y_test_roc, y_pred_roc)
		print(f"Training | Test Accuracy: {self.train_acc}% | {self.test_acc}%")
		print(f"AUC: {self.auc_score}")
		
		self.test_gene_names = self.data.loc[in_testing].Gene_Name



	def run(self):
		
		tf.config.threading.set_inter_op_parallelism_threads(1)
		tf.config.threading.set_intra_op_parallelism_threads(1)
		sess = tf.Session()
		K.set_session(sess)
		
		self.build_model()
		# gcn_model.build_model()
		self.aggregate_predictions()



if __name__ == '__main__':

	config_file = '../../../config.yaml'
	cfg = Config(config_file)

	set_generator = PrepareTrainTestSets(cfg)

	data = pd.read_csv(cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')
	train_dfs, test_dfs = set_generator.get_balanced_train_test_sets(data)


	# select random balanced dataset
	i = 8 #random.randint(0, len(train_dfs))
	print(f"i: {i}")
	train_data = train_dfs[i]
	test_data = test_dfs[i]
	print(f"Training set size: {train_data.shape[0]}")
	print(f"Test set size: {test_data.shape[0]}")

	X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data,
																									test_data)

	# convert target variables to 2D arrays
	y_train = np.array(to_categorical(y_train, 2))
	y_test = np.array(to_categorical(y_test, 2))

	dnn_params = ensemble_clf_params['GCN']
	dnn_params['clf_id'] = 'GCN'

	dnn_model = GCNClassifier(cfg, X_train, y_train, X_test, y_test, train_gene_names, test_gene_names, **dnn_params)
	
	(self, cfg, X_train, y_train, X_test, y_test,
				 train_gene_names, test_gene_names, clf_id,
				 regl, hidden_layer_nodes,
				 add_dropout, dropout_ratio, optimizer,
				 epochs, batch_size, verbose, make_plots)

	dnn_model.run()
	print(dnn_model.auc_score)
