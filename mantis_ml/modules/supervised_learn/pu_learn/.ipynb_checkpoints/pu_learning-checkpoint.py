import pandas as pd
import numpy as np
import random
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from multiprocessing import Process, Manager

from mantis_ml.config_class import Config
from mantis_ml.modules.supervised_learn.core.prepare_train_test_sets import PrepareTrainTestSets
from mantis_ml.modules.supervised_learn.classifiers.ensemble_lib import ensemble_clf_params
from mantis_ml.modules.supervised_learn.classifiers.sklearn_extended_classifier import SklearnExtendedClassifier
import xgboost as xgb

from matplotlib import pyplot as plt
import seaborn as sns

from scipy.stats import zscore
from datetime import datetime

sklearn_extended_classifiers = ['RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier', 'SVC', 'XGBoost','LogisticRegression']
feature_imp_classifiers = ['RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier', 'XGBoost','LogisticRegression','SGCN']

class PULearning:

	def __init__(self, cfg, data, clf_id, final_level_classifier=None):
		self.cfg = cfg
		self.data = data
		self.clf_id = clf_id
		self.final_level_classifier = final_level_classifier


	
	def train_gcn_on_subset(self, train_data, test_data, data, pos_genes_dict, neg_genes_dict, feature_dataframe_list, train_acc_list, test_acc_list, auc_score_list, pos_y_prob_dict, bikg_subgraph):
		
			from mantis_ml.modules.supervised_learn.classifiers.gcn import GCNClassifier
			from tensorflow.keras.utils import to_categorical

			set_generator = PrepareTrainTestSets(self.cfg)
			# set_generator = PrepareTrainTestSets(cfg)
			X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data, test_data)
	
			
			clf_params = ensemble_clf_params[self.clf_id]
			#clf_params['clf_id'] = self.clf_id
			
			gcn_model = GCNClassifier(self.cfg, X_train, y_train, X_test, y_test, train_gene_names, test_gene_names, data, bikg_subgraph,  clf_params, self.clf_id)
			# gcn_model = GCNClassifier(cfg, X_train, y_train, X_test, y_test, train_gene_names, test_gene_names, data, bikg_subgraph,  **gcn_params)
			
			gcn_model.run()
			print(gcn_model.train_acc, gcn_model.test_acc, gcn_model.auc_score)
			
			if self.clf_id=='SGCN':
				cur_model = gcn_model.model
				cur_feature_imp = cur_model.coef_[0,:]
				
				important_feature_indexes = list(range(X_train.shape[1]))
				feature_cols = train_data.iloc[:, important_feature_indexes].columns.values
				
				tmp_feature_dataframe = pd.DataFrame({'features': feature_cols, self.clf_id: cur_feature_imp})
				feature_dataframe_list.append(tmp_feature_dataframe)
			
			
			positive_y_prob = gcn_model.y_prob[:, 1]
			
			for g in range(len(gcn_model.test_gene_names)):
				cur_gene = gcn_model.test_gene_names.values[g]
	
				if cur_gene not in pos_y_prob_dict:
					pos_y_prob_dict[cur_gene] = [positive_y_prob[g]]
				else:
					pos_y_prob_dict[cur_gene] = pos_y_prob_dict[cur_gene] + [positive_y_prob[g]]
	
	
			train_acc_list.append(gcn_model.train_acc)
			test_acc_list.append(gcn_model.test_acc)
			auc_score_list.append(gcn_model.auc_score)
	
			tmp_pos_genes = list(gcn_model.tp_genes.values)
			tmp_pos_genes.extend(gcn_model.fp_genes.values)
	
			tmp_neg_genes = list(gcn_model.tn_genes.values)
			tmp_neg_genes.extend(gcn_model.fn_genes.values)
	
			for p in sorted(tmp_pos_genes):
				pos_genes_dict[p] = pos_genes_dict.get(p, 0) + 1
	
			for n in sorted(tmp_neg_genes):
				neg_genes_dict[n] = neg_genes_dict.get(n, 0) + 1
				
	
	def train_extended_sklearn_clf_on_subset(self, train_data, test_data, pos_genes_dict, neg_genes_dict, feature_dataframe_list, train_acc_list, test_acc_list, auc_score_list, pos_y_prob_dict):

		set_generator = PrepareTrainTestSets(self.cfg)
		# set_generator = PrepareTrainTestSets(cfg)
		X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data, test_data)
	
		# Select Boruta confirmed features, if specified in config.yaml
		important_feature_indexes = list(range(X_train.shape[1]))
	
		make_plots = False
		verbose = False

		clf_params = ensemble_clf_params[self.clf_id]
		# clf_params = ensemble_clf_params[clf_id]
	
		#t1 = datetime.now()
		model = SklearnExtendedClassifier(self.cfg, X_train, y_train, X_test, y_test,
										  train_gene_names, test_gene_names, self.clf_id, clf_params, make_plots, verbose)
		
		# model = SklearnExtendedClassifier(cfg, X_train, y_train, X_test, y_test,
		# 								  train_gene_names, test_gene_names, data, clf_id, clf_params, make_plots, verbose)
	
		if self.clf_id == 'XGBoost':
			model.model = xgb.XGBClassifier(**clf_params)
		else:
			model.build_model()
			
		model.train_model()
	
		#print(datetime.now())
	
	
		#t2 = datetime.now()
		# get feature importance
		# print('important_feature_indexes:', important_feature_indexes)
		feature_cols = train_data.iloc[:, important_feature_indexes].columns.values
		if self.clf_id == 'XGBoost':
			cur_model = model.model
			cur_feature_imp = cur_model.feature_importances_ 
			tmp_feature_dataframe = pd.DataFrame({'features': feature_cols, self.clf_id: cur_feature_imp})
			feature_dataframe_list.append(tmp_feature_dataframe)
		elif self.clf_id in feature_imp_classifiers:
			model.get_feature_importance(X_train, y_train, feature_cols, self.clf_id)
			# model.get_feature_importance(X_train, y_train, feature_cols, clf_id)
			feature_dataframe_list.append(model.feature_dataframe)

		model.process_predictions()
		
		#print(datetime.now()-t2)
		#t3 = datetime.now()
	
		positive_y_prob = model.y_prob[:, 1]
		
		#print(positive_y_prob)
	
		for g in range(len(model.test_gene_names)):
			cur_gene = model.test_gene_names.values[g]
	
			if cur_gene not in pos_y_prob_dict:
				pos_y_prob_dict[cur_gene] = [positive_y_prob[g]]
			else:
				pos_y_prob_dict[cur_gene] = pos_y_prob_dict[cur_gene] + [positive_y_prob[g]]
	
		#print(datetime.now()-t3)
		
	
		train_acc_list.append(model.train_acc)
		test_acc_list.append(model.test_acc)
		auc_score_list.append(model.auc_score)
	
		tmp_pos_genes = list(model.tp_genes.values)
		tmp_pos_genes.extend(model.fp_genes.values)
	
		tmp_neg_genes = list(model.tn_genes.values)
		tmp_neg_genes.extend(model.fn_genes.values)
	
		for p in sorted(tmp_pos_genes):
			pos_genes_dict[p] = pos_genes_dict.get(p, 0) + 1
	
		for n in sorted(tmp_neg_genes):
			neg_genes_dict[n] = neg_genes_dict.get(n, 0) + 1
	
	
	def train_rgcn_on_subset(self, train_data, test_data, pos_genes_dict, neg_genes_dict, train_acc_list, test_acc_list, auc_score_list, pos_y_prob_dict, bikg_subgraph):
		
			from mantis_ml.modules.supervised_learn.classifiers.rgcn import RGCNClassifier
			from tensorflow.keras.utils import to_categorical

			set_generator = PrepareTrainTestSets(self.cfg)
			X_train, y_train, train_gene_names, X_test, y_test, test_gene_names = set_generator.prepare_train_test_tables(train_data, test_data)
	
			
			gcn_params = ensemble_clf_params['GCN']
			gcn_params['clf_id'] = 'GCN'
			
			gcn_model = RGCNClassifier(self.cfg, X_train, y_train, X_test, y_test, train_gene_names, test_gene_names, bikg_subgraph,  **gcn_params)
			
			gcn_model.run()
			print(gcn_model.train_acc, gcn_model.test_acc, gcn_model.auc_score)
			
			positive_y_prob = gcn_model.y_prob[:, 1]
			
			for g in range(len(test_gene_names)):
				cur_gene = test_gene_names.values[g]
	
				if cur_gene not in pos_y_prob_dict:
					pos_y_prob_dict[cur_gene] = [positive_y_prob[g]]
				else:
					pos_y_prob_dict[cur_gene] = pos_y_prob_dict[cur_gene] + [positive_y_prob[g]]
	
	
			train_acc_list.append(gcn_model.train_acc)
			test_acc_list.append(gcn_model.test_acc)
			auc_score_list.append(gcn_model.auc_score)
	
			tmp_pos_genes = list(gcn_model.tp_genes.gene_name.values)
			tmp_pos_genes.extend(gcn_model.fp_genes.gene_name.values)
	
			tmp_neg_genes = list(gcn_model.tn_genes.gene_name.values)
			tmp_neg_genes.extend(gcn_model.fn_genes.gene_name.values)
	
			for p in sorted(tmp_pos_genes):
				pos_genes_dict[p] = pos_genes_dict.get(p, 0) + 1
	
			for n in sorted(tmp_neg_genes):
				neg_genes_dict[n] = neg_genes_dict.get(n, 0) + 1
				
	
	
	def run_pu_learning(self, selected_base_classifiers=None, final_level_classifier=None):
		
# 		bikg_subgraph = pd.read_csv(os.path.join(self.cfg.data_dir, 'bikg_subgraph', 'metabase_graph.csv'), sep = ',', index_col = 0).reset_index(drop=True)
		
		# bikg_subgraph = pd.read_csv(os.path.join(self.cfg.data_dir, 'bikg_subgraph', 'bikg_subgraph.csv'), sep = '\t')
		if self.clf_id in ['GCN','SGCN','RGCN']:
			bikg_subgraph = pd.read_csv(self.cfg.bikg_net, sep = '\t')
		# bikg_subgraph = pd.read_csv(os.path.join(cfg.data_dir, 'bikg_subgraph', 'bikg_subgraph.csv'), sep = '\t')
# 		# bikg_subgraph = pd.read_csv(os.path.join(cfg.data_dir, 'bikg_subgraph', 'metabase_graph.csv'), sep = ',', index_col = 0).reset_index(drop=True)
		
		if self.clf_id in ['GCN']:
			genes = set(list(bikg_subgraph.source_label.values)).union(set(list(bikg_subgraph.target_label.values)))
			self.data = self.data.loc[lambda x: x.Gene_Name.isin(np.array(list(genes)))]
		# data = data.loc[lambda x: x.Gene_Name.isin(np.array(list(genes)))]
		
		self.data = self.data.reset_index(drop=True)
		self.data = self.data.sample(frac = 1.0).reset_index(drop=True)
		
		# data = data.reset_index(drop=True)
		# data = data.sample(frac = 1.0).reset_index(drop=True)
		
		manager = Manager()
		pos_genes_dict = manager.dict()
		neg_genes_dict = manager.dict()
		pos_y_prob_dict = manager.dict()
	
		feature_dataframe_list = manager.list()
		train_acc_list = manager.list()
		test_acc_list = manager.list()
		auc_score_list = manager.list()
		process_jobs = []
		

		total_subsets = None
		for iter_num in range(1, self.cfg.iterations + 1):
			print('-----------------------------------------------> Iteration:', iter_num)
	
			process_jobs = []
	
			# get random partition of the entire dataset
			iter_random_state = random.randint(0, 1000000000)

			set_generator = PrepareTrainTestSets(self.cfg)
			# set_generator = PrepareTrainTestSets(cfg)
			train_dfs, test_dfs = set_generator.get_balanced_train_test_sets(self.data, random_state=iter_random_state)
			
			
			# train_dfs, test_dfs = set_generator.get_balanced_train_test_sets(data, random_state=iter_random_state)
			if total_subsets is None:
				total_subsets = len(train_dfs)
	
	
			# Loop through all balanced datasets from the entire partitioning of current iteration
			# only keep up to max_sets 
			
			if self.cfg.practise_run:
				max_sets = 5
				train_dfs = train_dfs[:np.min([len(train_dfs), max_sets])]
				test_dfs = test_dfs[:np.min([len(test_dfs), max_sets])]
			
			# check if any genes are always present in the train data
			always_genes = (
				pd.concat(
					train_dfs
				)
				[['Gene_Name', 'known_gene']]
				.groupby(['Gene_Name'])
				.count()
				.reset_index()
				.loc[lambda x: x.known_gene == len(train_dfs)]
				.Gene_Name.values
			)
			if len(always_genes) > 0: 
				train_dfs[-1] = train_dfs[-1].loc[lambda x: ~x.Gene_Name.isin(always_genes)]
		
		
			for i in range(len(train_dfs)): 
			#for i in range(10): # Debugging only
				train_data = train_dfs[i]
				test_data = test_dfs[i]
				
# 				for col in train_data.drop(['Gene_Name', 'known_gene'], axis=1).columns.values: 
# 					train_data[col] = zscore(train_data[col].values)
					
# 				train_data = train_data.dropna(axis=1, how = 'all').dropna()
				
# 				for col in test_data.drop(['Gene_Name', 'known_gene'], axis=1).columns.values: 
# 					test_data[col] = zscore(test_data[col].values)
				
# 				test_data = test_data.dropna(axis=1, how = 'all').dropna()
				
				
				print("Training set size: {} x {}".format(*train_data.shape))
				print("Test set size: {} x {}".format(*train_data.shape))
				
				if self.clf_id in sklearn_extended_classifiers:
					process_args = (train_data, test_data, pos_genes_dict, neg_genes_dict, feature_dataframe_list, train_acc_list, test_acc_list, auc_score_list, pos_y_prob_dict)
					#breakpoint()
					if self.cfg.nthreads!=1:
						p = Process(target=self.train_extended_sklearn_clf_on_subset, args=process_args)
					else:
						p = Process()
						self.train_extended_sklearn_clf_on_subset(*process_args)
				elif self.clf_id in ['GCN','SGCN']:
					#breakpoint()
					process_args = (train_data, test_data, self.data, pos_genes_dict, neg_genes_dict, feature_dataframe_list, train_acc_list, test_acc_list, auc_score_list, pos_y_prob_dict, bikg_subgraph)
					#p = Process(target=self.train_gcn_on_subset, args=)
					if self.cfg.nthreads!=1:
						p = Process(target=self.train_gcn_on_subset, args=process_args)
					else:
						p = Process()
						self.train_gcn_on_subset(*process_args)
				elif self.clf_id == 'RGCN':
					p = Process(target=self.train_rgcn_on_subset, args=(train_data, test_data, pos_genes_dict, neg_genes_dict, train_acc_list, test_acc_list, auc_score_list, pos_y_prob_dict, bikg_subgraph))

				process_jobs.append(p)
				p.start()

				if len(process_jobs) >= self.cfg.nthreads:
					for p in process_jobs:
						p.join()
					process_jobs = []
	
		for p in process_jobs:
			p.join()
	
	
		if self.clf_id in feature_imp_classifiers:
			feat_list_file = str(self.cfg.superv_out / ('PU_' + self.clf_id + '.feature_dfs_list.txt'))
			# cleanup
			if os.path.exists(feat_list_file):
				os.remove(feat_list_file)
			with open(feat_list_file, 'a') as f:	 
				for df in feature_dataframe_list:		 
					del df['features']
					df.to_csv(f)
	
			# Aggregate feature importance
			avg_feature_dataframe = None
			for feature_df in feature_dataframe_list:
				feature_df.index = feature_df['features']
				del feature_df['features']
	
				if avg_feature_dataframe is None:
					avg_feature_dataframe = feature_df
				else:
					avg_feature_dataframe.add(feature_df, fill_value=0)
	
			avg_feature_dataframe = avg_feature_dataframe/len(feature_dataframe_list)
			avg_feature_dataframe.to_csv(self.cfg.superv_out / ('PU_'+self.clf_id+'.avg_feature_importance.tsv'), sep='\t')
	
		avg_train_acc = round(sum(train_acc_list) / len(train_acc_list), 2)
		avg_test_acc = round(sum(test_acc_list) / len(test_acc_list), 2)
		avg_auc_score = round(sum(auc_score_list) / len(auc_score_list), 2)
	
		#print('\nAvg. training accuracy: ' + str(avg_train_acc) + ' %')
		#print('Avg. test accuracy: ' + str(avg_test_acc) + ' %')
		print('Avg. AUC score: ' + str(avg_auc_score))
	
		metrics_df = pd.DataFrame(list(zip(train_acc_list, test_acc_list, auc_score_list)), columns=['Train_Accuracy', 'Test_Accuracy', 'AUC'])
		metrics_df.to_csv(self.cfg.superv_out / ('PU_'+self.clf_id+'.evaluation_metrics.tsv'), sep='\t')
	
		neg_genes_df = pd.DataFrame(list(neg_genes_dict.values()), index=(neg_genes_dict.keys()),
									columns=['negative_genes'])
		neg_genes_df.sort_index(inplace=True)
		print(neg_genes_df.shape)
	
		pos_genes_df = pd.DataFrame(list(pos_genes_dict.values()), index=(pos_genes_dict.keys()),
									columns=['positive_genes'])
		pos_genes_df.sort_index(inplace=True)
		print(pos_genes_df.shape)
	
		all_genes_df = neg_genes_df.join(pos_genes_df, how='outer')
		all_genes_df.fillna(0, inplace=True)
	
		print(all_genes_df.head())
		print(all_genes_df.shape)
		all_genes_df.to_csv(self.cfg.superv_out / ('PU_'+self.clf_id+'.all_genes_predictions.tsv'), sep='\t')
	
		gene_proba_dict = pos_y_prob_dict.copy()
		gene_proba_df = pd.DataFrame.from_dict(gene_proba_dict, orient='index').T
		gene_proba_df = gene_proba_df.reindex(gene_proba_df.mean().sort_values(ascending=False).index, axis=1)
		
		gene_proba_df = gene_proba_df.transpose().mean(axis=1).to_frame()
		gene_proba_df.columns = ['predicted_proba']
	
		gene_proba_df.to_csv(self.cfg.superv_proba_pred / (self.clf_id + '.all_genes.predicted_proba.csv'))
		self.feature_dataframe_list = feature_dataframe_list
		self.train_acc_list = train_acc_list
		self.test_acc_list = test_acc_list
		self.auc_score_list = auc_score_list
		
		

	def run(self):
		self.run_pu_learning()


def aggregate_results(data_path, clf_id): 
	
	aucs = [] 
	for disease in os.listdir(data_path): 
		aucs_fname = os.path.join(data_path, disease, 'output_{}'.format(clf_id), 'supervised-learning', 'PU_{}.evaluation_metrics.tsv'.format(clf_id))
		
		if os.path.exists(aucs_fname): 
			aucs_i = pd.read_csv(aucs_fname, sep = '\t').assign(disease = disease)
			aucs.append(aucs_i)
	
	aucs = pd.concat(aucs)

	fig = plt.figure(figsize=(10, 5))
	sns.boxplot(
		x = 'disease', 
		y = 'AUC', 
		data = aucs
	)
	plt.xticks([])
	
	ranks = [] 
	for disease in os.listdir(data_path): 
		ranks_fname = os.path.join(data_path, disease, 'output_{}'.format(clf_id), 'rank_tests.csv')
		
		if os.path.exists(ranks_fname): 
			ranks_i = pd.read_csv(ranks_fname, sep = ',').assign(disease = disease)
			ranks.append(ranks_i)
	
	ranks = pd.concat(ranks)

	fig = plt.figure(figsize=(10, 5))
	sns.barplot(
		x = 'disease', 
		y = 'auc_rank', 
		data = ranks
	)
	plt.xticks([])
	
	return aucs, ranks


if __name__ == '__main__':
	# clf_id = 'XGBoost'
	#clf_id = 'RandomForestClassifier'
	clf_ids =  ['SGCN']#'LogisticRegression','XGBoost','SGCN','GCN']
	# clf_id = 'GCN'
	# clf_id = sys.argv[2]
	# clf_id = 'RGCN'
	# disease_name = 'Chronic_kidney_disease'
	# disease_name = 'Pain'
	# disease = disease_name
# 	disease_name = sys.argv[1]
	disease_name = "Type_I_diabetes_mellitus"
	disease_name = "Amyotrophic_lateral_sclerosis"
	disease_path = os.path.join(data_path, disease_name)
	# config_file = os.path.join(disease_path, '{}.yaml'.format(disease_name))
	# outdir = os.path.join(disease_path, 'output_{}_w_embeddings_128_lr_0.001'.format(clf_id))
	
	np.random.seed(1)
	
	seed_genes_source = 'HPO'
	exact_terms = None
	null_imp = 'zero'
	fuzzy_terms = None
	exclude_terms = None
	iterations = 1
	nthreads = 10
	use_bifeater = True
	# modify default config paramters when provided with respective parameters
	seed_genes_source=seed_genes_source
	learning_rate = 0.01
	dropout_ratio = 0.5 
	use_bifeater = False 
	n_epochs = 500
	n_filters = 16 
	n_layers = 2
	max_sets = 20
	custom_known_genes_file=None
	fast_run_option=False
	
	SGCN_one_max = True
	SGCN_summation = True
	SGCN_C = 10
	
	dts = []
	for clf_id in clf_ids:
		

		print('\n\n\n######################  running in output dir: {}\n\n\n'.format(output_dir))
		
		cfg = Config(disease_name, exact_terms, null_imp, features_table, fuzzy_terms, exclude_terms, output_dir)
		cfg.nthreads = int(nthreads)
		cfg.learning_rate = learning_rate
		cfg.dropout_ratio = float(dropout_ratio)
		cfg.use_bifeater = use_bifeater
		cfg.n_epochs = int(n_epochs)
		cfg.n_filters = int(n_filters)
		cfg.n_layers = int(n_layers)
		cfg.iterations = int(iterations)
		cfg.seed_genes_source = seed_genes_source
		cfg.custom_known_genes_file = custom_known_genes_file
		cfg.exp_inter = exp_inter
		cfg.inf_inter = inf_inter
		cfg.bikg_net = bikg_net
		cfg.nthreads = nthreads
		cfg.iterations = iterations
		cfg.max_sets = max_sets
		cfg.SGCN_one_max = SGCN_one_max
		cfg.SGCN_summation = SGCN_summation
		cfg.SGCN_C = SGCN_C
	# 	cfg.max_sets = int(max_sets)

	# 	cfg = Config(config_file, outdir)

		data = pd.read_csv(os.path.join(disease_path, 'processed-feature-tables', 'processed_feature_table.tsv'), sep='\t')
		# data.columns.values
		# data = data.drop(['Experimental_seed_genes_overlap'], axis=1)

	# 	embeddings = pd.read_csv('node_embeddings.csv', index_col = 0)
		embeddings = embeddings.rename(columns = {'default_label': 'Gene_Name'}).groupby(['Gene_Name']).mean().reset_index()
	# 	embeddings = embeddings.assign(Gene_Name = lambda x: x.index.values)

		data = data.merge(embeddings, on = 'Gene_Name')
	# 	data.known_gene.value_counts()


		print('nthreads:', cfg.nthreads)
		print('Stochastic iterations:', cfg.iterations)

		tic = datetime.now()
		pu = PULearning(cfg, data, clf_id)
		pu.run()
		dts += [datetime.now()-tic]
	
	# pd.concat(pu.feature_dataframe_list).groupby(['features']).mean().reset_index().sort_values(clf_id, ascending=False).head(n=40)
	
	# Run validation against phewas data
	

	gene_predictions_matrix = os.path.join(outdir, 'supervised-learning', 'gene_proba_predictions', '{}.all_genes.predicted_proba.csv'.format(clf_id))
	
	# parse phewas data
	phewas = parse_phewas_data(phewas_dir)
	
	# parse icd10 coding
	coding = parse_icd10_coding(icd10_coding)
	
	# parse gene predictions
	pred = parse_gene_predictions(gene_predictions_matrix)
	
	enrichment, stepwise_enrichment_out, rank_tests = run_validation(phewas, outdir, disease_name, coding, pred, icd10_mapping)
	
	enrichment.to_csv(os.path.join(outdir, 'supervised-learning', 'gene_proba_predictions', 'enrichment.csv'))
	stepwise_enrichment_out.to_csv(os.path.join(outdir, 'supervised-learning', 'gene_proba_predictions', 'stepwise_enrichment_out.csv'))
	rank_tests.to_csv(os.path.join(outdir, 'supervised-learning', 'gene_proba_predictions', 'rank_tests.csv'))
	
# 	##########  post processing - NOTES


	bikg_subgraph
	
	bikg_subgraph.relation_label.value_counts()


	clf_id = 'RandomForestClassifier'
	clf_id = 'GCN'

	
	aucs_rf, ranks_rf = aggregate_results(data_path, 'RandomForestClassifier')
	aucs_rf = aucs_rf.assign(model = 'RandomForestClassifier')
	ranks_rf = ranks_rf.assign(model = 'RandomForestClassifier')
	
	aucs_gcn, ranks_gcn = aggregate_results(data_path, 'GCN')
	aucs_gcn_01 = aucs_gcn.assign(model = 'GCN').assign(learning_rate = 0.1)
	ranks_gcn_01 = ranks_gcn.assign(model = 'GCN').assign(learning_rate = 0.1)
	
	aucs_gcn, ranks_gcn = aggregate_results(data_path, 'GCN')
	aucs_gcn_001 = aucs_gcn.assign(model = 'GCN').assign(learning_rate = 0.001)
	ranks_gcn_001 = ranks_gcn.assign(model = 'GCN').assign(learning_rate = 0.001)
	
	aucs_gcn, ranks_gcn = aggregate_results(data_path, 'GCN')
	aucs_gcn_def = aucs_gcn.assign(model = 'GCN').assign(learning_rate = 0.01)
	ranks_gcn_def = ranks_gcn.assign(model = 'GCN').assign(learning_rate = 0.01)
	
	aucs_gcn, ranks_gcn = aggregate_results(data_path, 'GCN')
	aucs_gcn_noembed = aucs_gcn.assign(model = 'GCN').assign(learning_rate = '0.01_no_embeddings')
	ranks_gcn_noembed = ranks_gcn.assign(model = 'GCN').assign(learning_rate = '0.01_no_embeddings')
	
	aucs = pd.concat([aucs_gcn_01, aucs_gcn_001, aucs_gcn_def, aucs_gcn_noembed])
	ranks = pd.concat([ranks_gcn_01, ranks_gcn_001, ranks_gcn_def, ranks_gcn_noembed])
	
	
	aucs_gcn, ranks_gcn = aggregate_results(data_path, 'GCN')
	aucs_gcn_200 = aucs_gcn.assign(model = 'GCN').assign(epochs = 200)
	ranks_gcn_200 = ranks_gcn.assign(model = 'GCN').assign(epochs = 200)
	
	aucs_gcn, ranks_gcn = aggregate_results(data_path, 'GCN')
	aucs_gcn_noembed = aucs_gcn.assign(model = 'GCN').assign(epochs = '200_no_embeddings')
	ranks_gcn_noembed = ranks_gcn.assign(model = 'GCN').assign(epochs = '200_no_embeddings')
	
	aucs = pd.concat([aucs_gcn_10, aucs_gcn_50, aucs_gcn_200, aucs_gcn_noembed])
	ranks = pd.concat([ranks_gcn_10, ranks_gcn_50, ranks_gcn_200, ranks_gcn_noembed])
	
	sns.boxplot(x = 'model', y = 'AUC', data = aucs, hue = 'learning_rate')
	plt.title('GCN with 128-dim gene embeddings')
	sns.boxplot(x = 'model', y = 'auc_rank', data = ranks, hue = 'learning_rate')
	
	aucs_w_embeddings_epochs_10 = pd.concat([aucs_rf, aucs_gcn]).assign(embeddings = 'dim_16')
	ranks_w_embeddings_16 = pd.concat([ranks_rf, ranks_gcn]).assign(embeddings = 'dim_16')
	
	aucs_w_embeddings_32 = pd.concat([aucs_rf, aucs_gcn]).assign(embeddings = 'dim_32')
	ranks_w_embeddings_32 = pd.concat([ranks_rf, ranks_gcn]).assign(embeddings = 'dim_32')
	
	aucs_w_embeddings_128 = pd.concat([aucs_rf, aucs_gcn]).assign(embeddings = 'dim_128')
	ranks_w_embeddings_128 = pd.concat([ranks_rf, ranks_gcn]).assign(embeddings = 'dim_128')
	
	aucs_no_embeddings = pd.concat([aucs_rf, aucs_gcn]).assign(embeddings = 'no')
	ranks_no_embeddings = pd.concat([ranks_rf, ranks_gcn]).assign(embeddings = 'no')
	
	aucs = pd.concat([aucs_no_embeddings, aucs_w_embeddings_16, aucs_w_embeddings_32, aucs_w_embeddings_128])
	ranks = pd.concat([ranks_no_embeddings, ranks_w_embeddings_16, ranks_w_embeddings_32, ranks_w_embeddings_128])
	

	sns.boxplot(x = 'model', y = 'AUC', data = aucs, hue = 'embeddings')
	sns.boxplot(x = 'model', y = 'auc_rank', data = ranks, hue = 'embeddings')
	
	sns.distplot(aucs.groupby(['disease', 'model']).mean().reset_index().pivot_table(index = 'disease', columns = ['model'], values = 'AUC').reset_index().assign(dif_auc = lambda x: x.GCN - x.RandomForestClassifier).dif_auc.values)
	plt.xlabel('dif_auc')
	
	sns.distplot(ranks.groupby(['disease', 'model']).mean().reset_index().pivot_table(index = 'disease', columns = ['model'], values = 'auc_rank').reset_index().assign(dif_rank = lambda x: x.GCN - x.RandomForestClassifier).dif_rank.values)
	plt.xlabel('dif_rank')
