import os, sys
try:
	user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
	user_paths = []

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
import yaml
from pathlib import Path
from shutil import copyfile
import pandas as pd
from random import randint
import string
import ntpath

class Config:

	def __init__(self, disease_name, exact_terms, null_imp, features_table, fuzzy_terms, exclude_terms, output_dir, verbose=False): 
		self.config_dir_path = os.path.dirname(os.path.realpath(__file__))
		self.output_dir = output_dir
		self.verbose = verbose


		# remove any tralining '/' from output dir
		self.output_dir = re.sub(r"\/$", "", self.output_dir)

		# Read static .config YAML file
		static_config_file = Path(self.config_dir_path + '/conf/.config')  
		with open(static_config_file, 'r') as ymlfile:
			static_conf = yaml.load(ymlfile, Loader=yaml.FullLoader)
		if self.verbose:
			print('\n> Static config:')
			for k,v in static_conf.items():
				print(k+':\t', v)

		#if exact_terms is not None: 
			#exact_terms = [x.replace(' ', '') for x in exact_terms.split(',') if x != '']
		self.features_table = features_table
		self.exact_terms = exact_terms
		self.null_imp = null_imp
		self.disease_name = disease_name

		#if fuzzy_terms is not None: 
			#fuzzy_terms = [x.replace(' ', '') for x in fuzzy_terms.split(',') if x != '']
		self.fuzzy_terms = fuzzy_terms

		#if exclude_terms is not None: 
			#exclude_terms = [x.replace(' ', '') for x in exclude_terms.split(',') if x != '']
		self.exclude_terms = exclude_terms


		input_conf = {'disease name': disease_name, 'exact terms': exact_terms, 'fuzzy terms': fuzzy_terms, 'exclude terms': exclude_terms, 'features_table': features_table, 'null_imp': null_imp}

		self.conf = {**static_conf, **input_conf}
		if self.verbose:
			print('\n> Full config:')
			for k,v in self.conf.items():
				print(k+':\t', v)


		# Custom file with known genes list 
		# -- Initiliase to None; Read with -k argument when calling mantisml
		self.custom_known_genes_file = None

		self.init_variables()
		self.init_directories()


	def get_valid_filename_from_str(self, str_val):

		valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
		valid_chars = frozenset(valid_chars)

		filename = ''.join(c for c in str_val if c in valid_chars)
		filename = filename.replace(' ', '_')

		return filename


	def init_variables(self, list_delim=',\s+'):


		# Specify target variable
		self.Y = self.conf['static']['Y_label']
		self.gene_name = self.conf['static']['gene_name']


		# >> Mandatory config field
		required_field_error_msg = "\n[Error] Please provide 'disease name' in command line arguments and re-run."
		if 'disease name' not in self.conf:
			sys.exit(required_field_error_msg)
		if self.conf['disease name'] is None:
			sys.exit(required_field_error_msg)

		self.disease_name = re.split(list_delim, self.conf['disease name'])
		if self.conf['exact terms'] is not None: 
			self.exact_terms = re.split(list_delim, self.conf['exact terms'])
		else: 
			self.exact_terms = [] 
		if self.conf['fuzzy terms'] is not None: 
			self.fuzzy_terms = re.split(list_delim, self.conf['fuzzy terms'])
		else: 
			self.fuzzy_terms = [] 

		if self.conf['exclude terms'] is not None: 
			self.exclude_terms = re.split(list_delim, self.conf['exclude terms'])
		else: 
			self.exclude_terms = [] 


		#self.seed_include_terms


		self.phenotype = self.get_valid_filename_from_str(ntpath.basename(self.output_dir))
		print('Phenotype/Output dir:', self.phenotype)
		


		# >> Optional input parameters
		# Additional feature terms to look up
#		if 'Additional associated terms' not in self.conf:
#			self.additional_include_terms = []
#		else:
#			self.additional_include_terms = self.conf['Additional associated terms']
#			if self.additional_include_terms is None:
#				self.additional_include_terms = []
#			else:
#				self.additional_include_terms = re.split(list_delim, self.additional_include_terms)
#
#		# Diseases/Phenotypes to exclude from HPO and features
#		if 'Diseases/Phenotypes to exclude' not in self.conf:
#			self.exclude_terms = []
#		else:
#			self.exclude_terms = self.conf['Diseases/Phenotypes to exclude']
#			if self.exclude_terms is None:
#				self.exclude_terms = []
#			else:
#				self.exclude_terms = re.split(list_delim, self.exclude_terms)
#

		# Genes to highlight on plots
		self.highlighted_genes = None # self.conf['Genes to highlight'] -- TODO: include it in next release
		if self.highlighted_genes is None:
			self.highlighted_genes = []
		else:
			self.highlighted_genes = re.split(list_delim, self.highlighted_genes)

		
		if self.verbose:
			#print('Output dirname:', self.phenotype)
			print('Disease name:', self.disease_name)
			print('\nexact/terms to include:', self.exact_terms)
			print('\nfuzzy/terms to include:', self.fuzzy_terms)
			print('\nexclude/terms to include:', self.exclude_terms)
			print('Genes to highlight:', self.highlighted_genes)

		# Run advanced
		self.include_disease_features = self.conf['run_advanced']['include_disease_features']
		self.generic_classifier = self.conf['run_advanced']['generic_classifier']
		if self.generic_classifier == 'None':
			self.generic_classifier = None
		self.hide_seed_genes_ratio = self.conf['run_advanced']['hide_seed_genes_ratio']
		self.seed_pos_ratio = self.conf['run_advanced']['seed_pos_ratio']
		self.random_seeds = self.conf['run_advanced']['random_seeds']

		# PU learning parameters
		self.classifiers = self.conf['pu_params']['classifiers']
		self.iterations = self.conf['pu_params']['iterations']
		self.nthreads = self.conf['pu_params']['nthreads']
		self.max_sets = self.conf['pu_params']['max_sets']


		# Data dir with input feature tables to be processed and compiled
		self.data_dir = Path(self.config_dir_path + '/' + self.conf['static']['data_dir'])


		# clustering params
		self.n_clusters = self.conf['static']['n_clusters']


		self.ntop_terms_enriched = self.conf['static']['ntop_terms_enriched']
		
		# NLP config
		self.ntop_terms_gtex = self.conf['static']['ntop_terms_gtex']
		self.ntop_terms_protein_atlas = self.conf['static']['ntop_terms_protein_atlas']
		self.ntop_terms_gwas = self.conf['static']['ntop_terms_gwas']
		self.ntop_terms_mgi = self.conf['static']['ntop_terms_mgi']
		self.ntop_terms_msigdb = self.conf['static']['ntop_terms_msigdb']
		self.seed_genes_source = self.conf['static']['seed_genes_source']
		#self.exact_string_matching = self.conf['static']['exact_string_matching']

		# Define default gene-set
		self.hgnc_genes_series = pd.read_csv(self.data_dir / 'exac-broadinstitute/all_hgnc_genes.txt', header=None).loc[:, 0]

		## === DIRS ===
		# Root Output path
		#self.out_root = Path(self.config_dir_path + '/../out/' + self.phenotype)
		self.out_root = Path(self.output_dir)
		print(self.out_root)
		
		# Root Figs output dir
		self.figs_dir = self.out_root / "Output-Figures"

		# Output dir to store processed feature tables
		self.processed_data_dir = self.out_root / "processed-feature-tables"

		# Unsupervised learning predictions/output folder
		self.unsuperv_out = self.out_root / 'unsupervised-learning'

		# Supervised learning predictions/output folder
		self.superv_out = self.out_root / 'supervised-learning'
		self.superv_pred = self.superv_out / 'gene_predictions'
		self.superv_proba_pred = self.superv_out / 'gene_proba_predictions'
		self.superv_ranked_by_proba = self.superv_out / 'ranked-by-proba_predictions'

		# Gene Predictions per classifier
		self.superv_ranked_pred = self.out_root / 'Gene-Predictions'

		# Output foldr for classifier benchmarking output
		self.benchmark_out = self.figs_dir / 'benchmarking'

		# EDA output folder for figures
		self.eda_out = self.figs_dir / 'EDA'

		# Unsupervised learning figures folder
		self.unsuperv_figs_out = self.figs_dir / 'unsupervised-learning'

		# Supervised learning figures folder
		self.superv_figs_out = self.figs_dir / 'supervised-learning'
		self.superv_feat_imp = self.superv_figs_out / 'feature-importance'
		self.superv_figs_gene_proba = self.superv_figs_out / 'gene_proba_predictions'

		# Overlap Results (from hypergeometric enrichment) 
		# figures per classifier
		self.overlap_out_dir = self.out_root / 'Overlap-Enrichment-Results'
		self.hypergeom_figs_out = self.overlap_out_dir / 'hypergeom-enrichment-figures'
		self.overlap_gene_predictions = self.overlap_out_dir / 'Gene-Predictions-After-Overlap'

		# Run steps (remove/add boruta and/or unsupervised steps)
		self.run_boruta = self.conf['run_steps']['run_boruta']
		self.run_unsupervised = self.conf['run_steps']['run_unsupervised']

		# Boruta feature selection output data & figures
		self.boruta_figs_dir = self.figs_dir / 'boruta'
		self.feature_selection_dir = self.out_root / 'feature_selection'
		self.boruta_tables_dir = self.feature_selection_dir / 'boruta'

		# Read filter args
		self.discard_highly_correlated = self.conf['eda_filters']['discard_highly_correlated']
		self.create_plots = self.conf['eda_filters']['create_plots']
		self.drop_missing_data_features = self.conf['eda_filters']['drop_missing_data_features']
		self.drop_gene_len_features = self.conf['eda_filters']['drop_gene_len_features']
		self.manual_feature_selection = self.conf['eda_filters']['manual_feature_selection']

		# Read other parameters for EDA
		self.missing_data_thres = self.conf['eda_parameters']['missing_data_thres']
		self.high_corr_thres = self.conf['eda_parameters']['high_corr_thres']

		# Read parameters for supervised learning
		self.feature_selection = self.conf['supervised_filters']['feature_selection']
		self.boruta_iterations = self.conf['supervised_filters']['boruta_iterations']
		self.boruta_decision_thres = self.conf['supervised_filters']['boruta_decision_thres']
		self.add_original_features_in_stacking = self.conf['supervised_filters']['add_original_features_in_stacking']
		self.test_size = self.conf['supervised_filters']['test_size']
		self.balancing_ratio = self.conf['supervised_filters']['balancing_ratio']
		self.random_fold_split = self.conf['supervised_filters']['random_fold_split']
		self.kfold = self.conf['supervised_filters']['kfold']

		print("P/U Balancing ratio: ", self.balancing_ratio)

		# randomisation
		self.random_state = 2018  

		# ============================
		# Dir with compiled feature tables
		self.out_data_dir = Path(self.out_root / 'data')
		self.compiled_data_dir = Path(self.out_data_dir / 'compiled_feature_tables')

		# Define input feature tables
		self.generic_feature_table = Path(self.compiled_data_dir / 'generic_feature_table.tsv')
		self.filtered_by_disease_feature_table = Path(self.compiled_data_dir / 'filtered_by_disease_feature_table.tsv')
		self.ckd_specific_feature_table = Path(self.compiled_data_dir / 'ckd_specific_feature_table.tsv')
		self.cardiov_specific_feature_table = Path(self.compiled_data_dir / 'cardiov_specific_feature_table.tsv')

		self.complete_feature_table = Path(self.compiled_data_dir / 'complete_feature_table.tsv')



	def init_directories(self):
		'''
		- Create output dirs
		- Copy config.yaml to out_root directory
		:return: 
		'''

		dirs = [self.out_root, self.compiled_data_dir, self.out_root, self.out_data_dir, self.figs_dir, 
			self.processed_data_dir, self.eda_out, self.unsuperv_out, self.unsuperv_figs_out, 
			self.superv_out, self.superv_pred, self.superv_ranked_pred, self.superv_ranked_by_proba, 
			self.superv_proba_pred, self.superv_figs_out, self.superv_feat_imp, self.superv_figs_gene_proba, 
			self.overlap_out_dir, self.hypergeom_figs_out, self.overlap_gene_predictions, self.benchmark_out, 
			self.boruta_figs_dir, self.feature_selection_dir, self.boruta_tables_dir]


		for d in dirs:
			if not os.path.exists(d):
				os.makedirs(d)

		# Copy input config.yaml to output dir
		#src_conf = str(self.config_file)
		dest_conf = str(self.out_root / 'config.yaml')
		#copyfile(src_conf, dest_conf)

		self.locked_config_path = dest_conf
		# print('Locked config path:', self.locked_config_path)


if __name__ == '__main__':

	conf_file = sys.argv[1]
	cfg = Config(conf_file)
