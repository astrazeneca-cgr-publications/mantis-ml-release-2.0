import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys, os
import glob
import pandas as pd
import ntpath
import pickle
from download import download
from argparse import RawTextHelpFormatter
from mantis_ml.modules.pre_processing.data_compilation.process_features_filtered_by_disease import ProcessFeaturesFilteredByDisease
from gensim.models import KeyedVectors

class Bioword2vec_embeddings(ProcessFeaturesFilteredByDisease): 
	
	def __init__(self, cfg): 
		
		ProcessFeaturesFilteredByDisease.__init__(self, cfg)
		
		wv_file = os.path.join(self.cfg.data_dir, 'bioword2vec_embeddings/bio_embedding_intrinsic')
		
		if not os.path.isfile(wv_file):
			#x = input('\n\nEmbedding file missing. Download? [required only once, ~1.8Gb] [Y/n]:')
			print('Embedding files missing. Downloading now...')
			#if x=='Y':
			os.makedirs(os.path.join(self.cfg.data_dir, 'bioword2vec_embeddings'), exist_ok=True)
			url = 'https://ndownloader.figshare.com/files/12551759'
			path = download(url, wv_file, progressbar=True)
			#else:
			#	print('exiting early...')
			#	sys.exit()
		
		
		self.embeddings = KeyedVectors.load_word2vec_format(wv_file , binary=True)

class MantisMl:

	def __init__(self, disease_name, exact_terms, null_imp, features_table, fuzzy_terms, exclude_terms, output_dir, 
				 seed_genes_source='HPO', nthreads=4, learning_rate = 0.01, dropout_ratio = 0.5, n_epochs = 200, 
				 n_filters = 16, n_layers = 1, iterations=10, 
				 practise_run=False, custom_known_genes_file=None, 
				 #interactions = 'in_web',
				 bikg_net = None,
				 fast_run_option=False, superv_models=None, no_nlp=False, shuffle_features=False,
				 SGCN_one_max = False, SGCN_summation = False, SGCN_C = 1): 
		
		from mantis_ml.config_class import Config
		self.exact_terms = exact_terms
		self.null_imp = null_imp
		self.features_table = features_table
		self.fuzzy_terms = fuzzy_terms
		self.exclude_terms = exclude_terms
		self.output_dir = output_dir

		self.cfg = Config(disease_name, exact_terms, null_imp, features_table, fuzzy_terms, exclude_terms, self.output_dir)

		if bikg_net is None:
			bikg_net = os.path.join(self.cfg.data_dir, 'in_web/inweb_pairwise_gene_interactions_for_gcn.tsv')
		
		# modify default config paramters when provided with respective parameters
		self.cfg.nthreads = int(nthreads)
		self.cfg.learning_rate = learning_rate
		self.cfg.dropout_ratio = float(dropout_ratio)
		self.cfg.n_epochs = int(n_epochs)
		self.cfg.n_filters = int(n_filters)
		self.cfg.n_layers = int(n_layers)
		self.cfg.iterations = int(iterations)
		self.cfg.seed_genes_source = seed_genes_source
		self.cfg.practise_run = bool(practise_run)
		self.cfg.no_nlp = bool(no_nlp)
		self.cfg.shuffle_features = bool(shuffle_features)
		self.cfg.SGCN_one_max = bool(SGCN_one_max)
		self.cfg.SGCN_summation = bool(SGCN_summation)
		self.cfg.SGCN_C = float(SGCN_C)
		
		if self.cfg.no_nlp and (len(self.cfg.fuzzy_terms)>0):
			print('[Warning] --no_nlp over-riding fuzzy terms (treated as empty)\n')

		if fast_run_option:
			self.cfg.classifiers = ['ExtraTreesClassifier', 'RandomForestClassifier', 'SVC', 'GradientBoostingClassifier']

		if superv_models:
			models_dict = { 'et': 'ExtraTreesClassifier',	
					'rf': 'RandomForestClassifier',
					'svc': 'SVC',	
					'gb': 'GradientBoostingClassifier',
					'xgb': 'XGBoost',
					'dnn': 'DNN',
					'gcn': 'GCN',
					'sgcn': 'SGCN',
					'lr' : 'LogisticRegression',
					'stack': 'Stacking' }

			try:
				self.cfg.classifiers = list(set([ models_dict[k] for k in superv_models.split(',') ]))
			except:
				print('[Warning] -m option args are not correct.\n\t  Currently going ahead with mantis-ml run using the 6 default classifiers (unless -f has also been specified which will integrate 4 classifiers only).\n')


		self.cfg.custom_known_genes_file = custom_known_genes_file
		#self.cfg.exp_inter = exp_inter
		#self.cfg.inf_inter = inf_inter
		# self.cfg.interactions = interactions
		# if self.cfg.interactions not in ['in_web','bikg']:
		# 	print('[Warning] invalid --interactions argument. Proceeding with in_web')
		# 	self.cfg.interactions = 'in_web'
		
		self.cfg.bikg_net = bikg_net

		print('nthreads:', self.cfg.nthreads)
		print('practise_run:', self.cfg.practise_run)
		print('Stochastic iterations:', self.cfg.iterations)
		print('Classifiers:', self.cfg.classifiers)
		print('Custom known genes:', self.cfg.custom_known_genes_file)
		print('output dir: {}'.format(self.output_dir))
		print('Turn off NLP: {}'.format(self.cfg.no_nlp))


# 		# Run profiler and store results to ouput dir
# 		run_cmd = "mantisml-profiler -v " + " -d " + "'" + disease_name + "'" + " -g " + seed_genes_source + " -l " + null_imp
# 		if exact_terms is not None: 
# 			run_cmd = run_cmd + " -e " + "'" + exact_terms + "'" 
# 		if features_table is not None: 
# 			run_cmd = run_cmd + " -t " + "'" + features_table + "'" 
# 		if fuzzy_terms is not None: 
# 			run_cmd = run_cmd + " -z " + "'" + fuzzy_terms + "'" 
# 		if exclude_terms is not None: 
# 			run_cmd = run_cmd + " -z " + "'" + exclude_terms + "'" 
			
# 		run_cmd = run_cmd +  " -o " + self.output_dir + " > " + str(self.cfg.out_root) + "/profiler_metadata.out"
	
# 		os.system(run_cmd)



	def get_clf_id_with_top_auc(self):

		auc_per_clf = {}

		metric_files = glob.glob(str(self.cfg.superv_out / 'PU_*.evaluation_metrics.tsv'))

		for f in metric_files:
			clf_id = ntpath.basename(f).split('.')[0].replace('PU_', '')

			tmp_df = pd.read_csv(f, sep='\t', index_col=0)
			avg_auc = tmp_df.AUC.median()
			auc_per_clf[clf_id] = avg_auc

		top_clf = max(auc_per_clf, key=auc_per_clf.get)
		print('Top classifier:', top_clf)

		return top_clf




	def run(self, clf_id=None, final_level_classifier='DNN', run_feature_compiler=False, run_eda=False, run_pu=False,
				  run_aggregate_results=False, run_merge_results=False,
				  run_boruta=False, run_unsupervised=False):

		# *** Load required modules ***
		from mantis_ml.modules.supervised_learn.pu_learn.pu_learning import PULearning
		from mantis_ml.modules.pre_processing.eda_wrapper import EDAWrapper
		from mantis_ml.modules.pre_processing.feature_table_compiler import FeatureTableCompiler
		from mantis_ml.modules.unsupervised_learn.dimens_reduction_wrapper import DimensReductionWrapper
		from mantis_ml.modules.post_processing.process_classifier_results import ProcessClassifierResults
		from mantis_ml.modules.post_processing.merge_predictions_from_classifiers import MergePredictionsFromClassifiers
		from mantis_ml.modules.supervised_learn.feature_selection.run_boruta import BorutaWrapper



		# ========= Compile feature table =========
		if run_feature_compiler:
			feat_compiler = FeatureTableCompiler(self.cfg)
			wv = Bioword2vec_embeddings(self.cfg)
			feat_compiler.run(wv.embeddings)


		# ========= Run EDA and pre-processing =========
		if run_eda:
			eda_wrapper = EDAWrapper(self.cfg)
			eda_wrapper.run()

		data = pd.read_csv(self.cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')





		# ================== Supervised methods ==================
		# ************ Run PU Learning ************
		if run_pu:
			for clf_id in self.cfg.classifiers:
				print('Classifier:', clf_id)
				pu = PULearning(self.cfg, data, clf_id, final_level_classifier)
				pu.run()


		# ************ Process predictions per classifier ************
		if run_aggregate_results:
			aggr_res = ProcessClassifierResults(self.cfg, show_plots=True)
			aggr_res.run()
			
# 			clf_fname = self.cfg.superv_proba_pred / (self.clf_id + '.all_genes.predicted_proba.csv')
# 			df = pd.read_csv(clf_fname, index_col = 0)
			
# 			df.head()
# 			df.shape


		# ************ Merge results from all classifiers ************
		if run_merge_results:
			merger = MergePredictionsFromClassifiers(self.cfg)
			merger.run()


		# ************ Run Boruta feature seleciton algorithm ************
		if run_boruta:
			boru_wrapper = BorutaWrapper(self.cfg)
			boru_wrapper.run()


		# ========= Unsupervised methods =========
		# PCA, sparse PCA and t-SNE
		if run_unsupervised:
			recalc = False # default: False
		
			if clf_id is None:
					highlighted_genes = self.cfg.highlighted_genes
			else:
				top_genes_num = 40
				novel_genes = pd.read_csv(str(self.cfg.superv_ranked_by_proba / (clf_id + '.Novel_genes.Ranked_by_prediction_proba.csv')), header=None, index_col=0)
				highlighted_genes = novel_genes.head(top_genes_num).index.values

			dim_reduct_wrapper = DimensReductionWrapper(self.cfg, data, highlighted_genes, recalc)
			dim_reduct_wrapper.run()

			
			

		
	def run_non_clf_specific_analysis(self):
		""" run_tag: pre """

		args_dict = {'run_feature_compiler': True, 'run_eda': True, 'run_unsupervised': self.cfg.run_unsupervised}
		self.run(**args_dict)



	def run_boruta_algorithm(self):
		""" run_tag: boruta """

		args_dict = {'run_boruta': True}
		self.run(**args_dict)


		
	def run_pu_learning(self):
		""" run_tag: pu """
		
		args_dict = {'run_pu': True}
		self.run(**args_dict)


		
	def run_post_processing_analysis(self):
		""" run_tag: post """
		
		args_dict = {'run_aggregate_results': True, 'run_merge_results': True}
		self.run(**args_dict)


		
	def run_clf_specific_unsupervised_analysis(self, clf_id):
		""" run_tag: post_unsup """
		
		args_dict = {'clf_id': clf_id, 'run_unsupervised': True}
		self.run(**args_dict)

		
	# ---------------------- Run Full pipeline ------------------------
	def run_all(self):
		""" run_tag: all """

		args_dict = {'run_feature_compiler': True, 'run_eda': True, 'run_pu': True,
				  'run_aggregate_results': True, 'run_merge_results': True,
				  'run_boruta': False, 'run_unsupervised': True}
		self.run(**args_dict)
	# -----------------------------------------------------------------
		





def main():

	parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
#	parser.add_argument("-c", dest="config_file", help="Config file (.yaml) with run parameters [Required]\n\n", required=True)
	parser.add_argument("-d", dest="disease_name", default=None, help="Disease name. [Required]\n\n", required = True)
	parser.add_argument("-t", dest="features_table", default=None, help="Extra features table to merge with\n\n")
	parser.add_argument("-l", dest="null_imp", default="median", help="Null imputation method if external features table is given. Can be either zero, or median, or the path to a tsv file specifying the imputation method for each column separately\n\n")
	parser.add_argument("-e", dest="exact_terms", default=None, help="Terms to match against using regular expression matching\n\n")
	parser.add_argument("-z", dest="fuzzy_terms", default=None, help="Terms to match against using NLP\n\n")
	parser.add_argument("-x", dest="exclude_terms", default=None, help="Terms to exclude\n\n")

	parser.add_argument("-o", dest="output_dir", help="Output directory name\n(absolute/relative path e.g. ./CKD, /tmp/Epilepsy-testing, etc.)\nIf it doesn't exist it will automatically be created [Required]\n\n", required=True)
	parser.add_argument("-r", dest="run_tag", choices=['all', 'pre', 'boruta', 'pu', 'post', 'post_unsup'], default='all', help="Specify type of analysis to run (default: all)\n\n")

	parser.add_argument("-f", "--fast", action="count", help="Fast training using only 4 classifiers: Extra Trees, Random Forest, SVC and Gradient Boosting.\nBy default, mantis-ml uses 6 supervised models for training: Extra Trees, Random Forest, SVC, Gradient Boosting, XGBoost and Deep Neural Net.\n\n")

	parser.add_argument("-m", dest="superv_models", default=None, help="Explicitly specify which supervised models to be used for training. This overrides the '-f/--fast' option.\n- Options:\n et: Extra Trees\n rf: Random Forest\n gb: Gradient Boosting\n xgb: XGBoost\n svc: Support Vector Classifier\n dnn: Deep Neural Net\n stack: Stacking classifier\n\nMultiple models may be specified using a ',' separator, e.g. -m et,rf,stack\nWhen this option is not specified, 6 models are trained by default with mantis-ml: Extra Trees, Random Forest, SVC, Gradient Boosting, XGBoost and Deep Neural Net. \n\n")

	parser.add_argument("-k", dest="known_genes_file", help="File with custom list of known genes used for training (new-line separated)\n\n")

	parser.add_argument("-n", dest="nthreads", default=4, help="Number of threads (default: 4)\n\n")
	parser.add_argument("-i", dest="iterations", default=10, help="Number of stochastic iterations for semi-supervised learning (default: 10)\n\n")
	parser.add_argument("-g", dest="seed_genes_source", default='HPO', help="Resource to extract the seed genes from. either HPO, OT, or GEL (default: HPO)\n\n")
	#parser.add_argument("-s", dest="max_sets", default=-1)# help="maximum number of train/test sets (default: 5)\n\n")
	
	#parser.add_argument("--exp_inter", default='in_web/experimental_pairwise_interactions_bikg.tsv', help="File containing gene to gene interactions in tab delimited format\n\n")
	#parser.add_argument("--inf_inter", default='in_web/inferred_pairwise_interactions_bikg.tsv', help="File containing gene to gene interactions in tab delimited format\n\n")
	parser.add_argument("--practise_run", action="store_true", help="Specifies whether to run on a subset of 5 stochastic partitions (default: use all partitions)\n\n")
	#parser.add_argument("--interactions", default='in_web', help="Specifies whether to use in_web or bikg pairwise interactions (default: in_web)\n\n")
	parser.add_argument("--n_epochs", default=200, help="Number of epochs to train the GCN/DNN for (default: 200)\n\n") # previous iteration had this at 200
	parser.add_argument("--learning_rate", default=0.01, help="Learning rate for DNN/GCN (default: 0.01)\n\n") # previous iteration had this at 0.01
	parser.add_argument("--n_filters", default=16, help="Number of filters for the GCN (default: 16)\n\n")
	parser.add_argument("--n_layers", default=2, help="Number of hidden layers for the GCN or power of adjacency matrix in SGCN (1 recommended for GCN) (default: 2)\n\n")
	parser.add_argument("--dropout_ratio", default=0.5, help="Dropout ratio for the GCN (default: 0.5)\n\n")
	parser.add_argument("--bikg_net", default=None, help="Network to use with the GCN - InWeb used if none provided otherwise (default: None)\n\n")
	parser.add_argument("--no_nlp", action="store_true", help="Optionally remove NLP and revert to string matching\n\n")
	parser.add_argument("--shuffle_features", action="store_true", help="Can use shuffled features (must be already generated)\n\n")
	parser.add_argument("--SGCN_one_max", action="store_true", help="Whether to ensure the maximum of all adjacency variables is 1 for SGCN (default: False)\n\n")
	parser.add_argument("--SGCN_summation", action="store_true", help="Whether to aggregate by mean or summation (default: mean)\n\n")
	parser.add_argument("--SGCN_C", default=1, help="Regularisation parameter for SGCN - smaller the stronger the regularisation (default: 1)\n\n")
	

	if len(sys.argv)==1:
		parser.print_help(sys.stderr)
		sys.exit(1)


	args = parser.parse_args()
	print(args)



	disease_name = args.disease_name
	features_table = args.features_table
	exact_terms = args.exact_terms
	null_imp = args.null_imp
	fuzzy_terms = args.fuzzy_terms
	exclude_terms = args.exclude_terms


	if disease_name is None: 
		sys.exit("Missing input arguments. -d is required.")


	output_dir = args.output_dir
	# output_dir = outdir
	run_tag = args.run_tag
	fast_run_option = bool(args.fast)
	superv_models = args.superv_models

	custom_known_genes_file = args.known_genes_file
	#exp_inter = args.exp_inter
	#inf_inter = args.inf_inter
	#interactions = args.interactions
	bikg_net = args.bikg_net
	nthreads = args.nthreads
	learning_rate = args.learning_rate
	dropout_ratio = args.dropout_ratio
	n_epochs = args.n_epochs
	n_filters = args.n_filters
	n_layers = args.n_layers
	practise_run = args.practise_run
	iterations = args.iterations
	seed_genes_source = args.seed_genes_source
	no_nlp = args.no_nlp
	shuffle_features = args.shuffle_features
	SGCN_one_max = args.SGCN_one_max
	SGCN_summation = args.SGCN_summation
	SGCN_C = args.SGCN_C


	mantis = MantisMl(disease_name,
			  exact_terms,
			  null_imp, 
			  features_table,
			  fuzzy_terms,
			  exclude_terms,
			  output_dir, 
			  seed_genes_source = seed_genes_source,
			  nthreads=nthreads, 
			  learning_rate = learning_rate,
			  dropout_ratio = dropout_ratio,
			  n_epochs = n_epochs,
			  n_filters = n_filters,
			  n_layers = n_layers,
			  practise_run=practise_run,
			  iterations=iterations, 
			  custom_known_genes_file=custom_known_genes_file,
			  #exp_inter = exp_inter,
			  #inf_inter = inf_inter,
			  #interactions = interactions,
			  bikg_net = bikg_net,
			  fast_run_option = fast_run_option,
			  superv_models = superv_models,
			  no_nlp = no_nlp,
			  shuffle_features = shuffle_features,
			  SGCN_one_max = SGCN_one_max,
			  SGCN_summation = SGCN_summation,
			  SGCN_C = SGCN_C)


	if run_tag == 'all':
		mantis.run_all()
	elif run_tag == 'pre':
		mantis.run_non_clf_specific_analysis()
	elif run_tag == 'pu':
		mantis.run_pu_learning()
	elif run_tag == 'post':
		mantis.run_post_processing_analysis()
	elif run_tag == 'post_unsup':
		top_clf = mantis.get_clf_id_with_top_auc()
		mantis.run_clf_specific_unsupervised_analysis(top_clf)
	elif run_tag == 'boruta':
		mantis.run_boruta_algorithm()



if __name__ == '__main__':
	main()
