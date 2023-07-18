from collections import Counter
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import sys, os
import re
import pandas as pd

from mantis_ml.modules.pre_processing.data_compilation.process_features_filtered_by_disease import ProcessFeaturesFilteredByDisease
from mantis_ml.config_class import Config
from gensim.models import KeyedVectors

class Bioword2vec_embeddings(ProcessFeaturesFilteredByDisease): 
	
	def __init__(self, cfg): 
		
		ProcessFeaturesFilteredByDisease.__init__(self, cfg)
		
		wv_file = os.path.join(self.cfg.data_dir, 'bioword2vec_embeddings/bio_embedding_intrinsic')
		self.embeddings = KeyedVectors.load_word2vec_format(wv_file , binary=True)


class MantisMlProfiler:

	def __init__(self, disease_name, exact_terms, null_imp, features_table, fuzzy_terms, exclude_terms, seed_genes_source, output_dir, verbose=False):

		self.output_dir = output_dir
		self.disease_name = disease_name
		self.seed_genes_source = seed_genes_source
		self.exact_terms = exact_terms
		self.null_imp = null_imp
		self.features_table = features_table
		self.fuzzy_terms = fuzzy_terms
		self.exclude_terms = exclude_terms

		self.verbose = verbose

		# common strings to exclude from profiling
		self.eng_stopwords = self.get_english_stopwords()
		self.custom_bullet = '=' * 5 
		self.line_spacer = '\n' * 6
	

	# Disable print to stdout
	def blockPrint(self):
		sys.stdout = open(os.devnull, 'w')

	# Restore print to stdout
	def enablePrint(self):
		sys.stdout = sys.__stdout__

	def bordered(self, text):

		lines = text.splitlines()
		width = max(len(s) for s in lines)

		border = '-' * width 
		res = [border]
		res.append(text)
		res.append(border)

		return '\n'.join(res)



	def assess_hpo_filtered_output(self, proc_obj, cfg):
		print(self.line_spacer + "-----------------   Assessing HPO filtering [config parameters: 'seed_include_terms']	-----------------\n")

		print("- Provided 'exact_terms':")
		print(cfg.exact_terms)

		if self.seed_genes_source == 'HPO': 
			seed_df, hpo_selected_terms = proc_obj.process_hpo(cfg.disease_name, cfg.exclude_terms, cfg.phenotype)
		elif self.seed_genes_source == 'OT': 
			seed_df, hpo_selected_terms = proc_obj.process_OT(cfg.disease_name, cfg.exclude_terms, cfg.phenotype)
		elif self.seed_genes_source == 'GEL': 
			seed_df, hpo_selected_terms = proc_obj.process_GEL(cfg.disease_name, cfg.exclude_terms, cfg.phenotype)
		else: 
			seed_df, hpo_selected_terms = proc_obj.process_hpo(cfg.disease_name, cfg.exclude_terms, cfg.phenotype)
			
			
		selected_genes = seed_df['Gene_Name'].tolist()
		if self.verbose:
			print('\n' + self.bordered(self.custom_bullet + '  Selected HPO genes  ' + self.custom_bullet))
			print(selected_genes)
			seed_df[['Gene_Name']].to_csv(os.path.join(self.output_dir, 'hpo_genes.csv'), index=None, header=False)

		print('\n' + self.bordered(self.custom_bullet + '  Selected HPO disease-associated terms  ' + self.custom_bullet))
		print(sorted(list(hpo_selected_terms)))

		hpo_selected_terms_expanded = [s.split() for s in hpo_selected_terms]
		hpo_selected_terms_expanded = [item.lower() for sublist in hpo_selected_terms_expanded for item in sublist]

		# remove stopwords
		hpo_selected_terms_expanded = [w for w in hpo_selected_terms_expanded if w not in self.eng_stopwords]

		# remove digits
		hpo_selected_terms_expanded = [w for w in hpo_selected_terms_expanded if not w.isdigit()]

		if self.verbose:
			count_hpo_terms = Counter(hpo_selected_terms_expanded)
			print('\n> Most common strings in filtered HPO phenotype terms:')
			for s, count in count_hpo_terms.most_common():
				if count == 1:
					continue
				print(s + ':', count)
				
		pd.DataFrame(hpo_selected_terms, columns = ['hpo_selected_terms']).to_csv(os.path.join(self.output_dir, 'hpo_selected_terms.csv'), index=None, header=False)



	def assess_gtex_filtered_output(self, proc_obj, cfg, wv):

		print(self.line_spacer + "-----------------	Assessing GTEx filtering [config parameters: 'tissue' and 'additional_tissues']	-----------------\n")
		print("\n- Provided 'exact_terms':")
		print(cfg.exact_terms)
		print("\n- Provided 'fuzzy_terms':")
		print(cfg.fuzzy_terms)
		#print("- Provided 'tissue':")
		#print(cfg.tissue)
		#print("\n- Provided 'additional_tissues':")
		#print(cfg.additional_tissues)

		self.blockPrint()
		_, selected_tissue_cols, all_tissue_cols = proc_obj.process_gtex_features(wv)
		self.enablePrint()

		all_tissue_cols = list(all_tissue_cols)
		selected_tissue_cols = list(selected_tissue_cols)
		
		if self.verbose:
			print('\nAvailable GTEx tissues:')
			print(sorted(all_tissue_cols))

		print('\n' + self.bordered(self.custom_bullet + '  Selected GTEx tissues  ' + self.custom_bullet))
		print(sorted(selected_tissue_cols))
		
		pd.DataFrame(selected_tissue_cols, columns = ['selected_tissue_cols']).to_csv(os.path.join(self.output_dir, 'selected_tissue_cols.csv'), index=None, header=False)



	def assess_proteinatlas_filtered_output(self, proc_obj, cfg, wv):

		print(self.line_spacer + "-----------------	Assessing Protein Atlas filtering [config parameters: 'tissue', 'seed_include_terms', 'additional_include_terms']	-----------------\n")
		print("- Provided 'tissue':")
		#print(cfg.tissue)
		print("\n- Provided 'exact_terms':")
		print(cfg.exact_terms)
		print("\n- Provided 'fuzzy_terms':")
		print(cfg.fuzzy_terms)

		self.blockPrint()
		prot_atlas_df, selected_normal_tissues, all_normal_tissues, selected_rna_samples, all_rna_samples = proc_obj.process_protein_atlas_features(wv)
		self.enablePrint()

		if self.verbose:
			print('\nAvailable tissues (normal_tissue.tsv.gz data):')
			print(sorted(all_normal_tissues))

		print('\n' + self.bordered(self.custom_bullet + '  Selected tissues from Protein Atlas (normal_tissue.tsv.gz)  ' + self.custom_bullet))
		print(sorted(selected_normal_tissues))

		if self.verbose:
			print('\nAvailable samples (rna_tissue.tsv.gz data):')
			print(sorted(all_rna_samples))

		print('\n' + self.bordered(self.custom_bullet + '  Selected samples from Protein Atlas (rna_tissue.tsv.gz)  ' + self.custom_bullet))
		print(sorted(selected_rna_samples))
		
		pd.DataFrame(selected_rna_samples, columns = ['selected_rna_samples']).to_csv(os.path.join(self.output_dir, 'selected_rna_samples.csv'), index=None, header=False)




	def assess_msigdb_filtered_output(self, proc_obj, cfg, wv):
		print(self.line_spacer + "-----------------	Assessing MSigDB filtering [config parameters: 'tissue', 'seed_include_terms', 'additional_include_terms']	-----------------\n")
		#print("- Provided 'tissue':")
		#print(cfg.tissue)
		print("\n- Provided 'exact_terms':")
		print(cfg.exact_terms)
		print("\n- Provided 'fuzzy_terms':")
		print(cfg.fuzzy_terms)
		
		if self.seed_genes_source == 'HPO': 
			seed_df, hpo_selected_terms = proc_obj.process_hpo(cfg.disease_name, cfg.exclude_terms, cfg.phenotype)
		elif self.seed_genes_source == 'OT': 
			seed_df, hpo_selected_terms = proc_obj.process_OT(cfg.disease_name, cfg.exclude_terms, cfg.phenotype)
		elif self.seed_genes_source == 'GEL': 
			seed_df, hpo_selected_terms = proc_obj.process_GEL(cfg.disease_name, cfg.exclude_terms, cfg.phenotype)
		else: 
			seed_df, hpo_selected_terms = proc_obj.process_hpo(cfg.disease_name, cfg.exclude_terms, cfg.phenotype)
	
		seed_genes = seed_df['Gene_Name'].tolist()

		self.blockPrint()
		msigdb_go_df, selected_go_terms = proc_obj.process_msigdb_go_features(wv, seed_genes)
		self.enablePrint()

		print('\n' + self.bordered(self.custom_bullet + '  Selected Gene Ontology terms (from MSigDB)  ' + self.custom_bullet))
		print(sorted(selected_go_terms))
		
		pd.DataFrame(selected_go_terms, columns = ['selected_go_terms']).to_csv(os.path.join(self.output_dir, 'selected_go_terms.csv'), index=None, header=False)




	def assess_mgi_filtered_output(self, proc_obj, cfg, wv):
		if cfg.generic_classifier:
			return 0

		print(self.line_spacer + "-----------------	Assessing MGI filtering [config parameters: 'seed_include_terms', 'additional_include_terms']	-----------------\n")
		print("\n- Provided 'exact_terms':")
		print(cfg.exact_terms)
		print("\n- Provided 'fuzzy_terms':")
		print(cfg.fuzzy_terms)

		self.blockPrint()
		_, selected_mgi_phenotypes = proc_obj.process_mgi_features(wv)
		self.enablePrint()
		# print(selected_mgi_phenotypes)

		selected_mgi_phenotypes_expanded = [s.split('|') for s in selected_mgi_phenotypes]
		selected_mgi_phenotypes_expanded = list(set([item.lower() for sublist in selected_mgi_phenotypes_expanded for item in sublist]))
		
		pd.DataFrame(selected_mgi_phenotypes_expanded, columns = ['selected_mgi_phenotypes_expanded']).to_csv(os.path.join(self.output_dir, 'selected_mgi_phenotypes_expanded.csv'), index=None, header=False)
		
		# print(selected_mgi_phenotypes_expanded)

# 		include_pattern = re.compile('|'.join(mgi_include_terms), re.IGNORECASE)

# 		filtered_selected_mgi_phenotypes_expanded = list(filter(lambda x: re.search(include_pattern, x), selected_mgi_phenotypes_expanded))
# 		# print(filtered_selected_mgi_phenotypes_expanded)

# 		print('\n' + self.bordered(self.custom_bullet + '  Selected MGI phenotypes  ' + self.custom_bullet))
# 		print(sorted(filtered_selected_mgi_phenotypes_expanded))



	def get_english_stopwords(self):
		eng_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

		return eng_stopwords




	def run_mantis_ml_profiler(self):

		print('>>> Running mantis-ml config profiling ...')
		print('verbose:', self.verbose)
		print('Output dir:', self.output_dir)
		

		cfg = Config(self.disease_name, self.exact_terms, self.null_imp, self.features_table, self.fuzzy_terms, self.exclude_terms,  self.output_dir)
		wv = Bioword2vec_embeddings(cfg)
		proc_obj = ProcessFeaturesFilteredByDisease(cfg)

		# HPO
		self.assess_hpo_filtered_output(proc_obj, cfg)


		# GTEx
		self.assess_gtex_filtered_output(proc_obj, cfg, wv.embeddings)


		# Protein Atlas
		self.assess_proteinatlas_filtered_output(proc_obj, cfg, wv.embeddings)


		# MSigDB
		self.assess_msigdb_filtered_output(proc_obj, cfg, wv.embeddings)


		# MGI
		self.assess_mgi_filtered_output(proc_obj, cfg, wv.embeddings)
		print('\n\n<<< mantis-ml config profiling complete.')




	
def main():

	parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
	parser.add_argument("-d", dest="disease_name", default=None, help="Disease name. [Required]\n\n", required = True)
	parser.add_argument("-t", dest="features_table", default=None, help="Extra features table to merge with\n\n")
	parser.add_argument("-l", dest="null_imp", default="median", help="Null imputation method if external features table is given. Can be either zero, or median, or the path to a tsv file specifying the imputation method for each column separately\n\n")
	parser.add_argument("-g", dest="seed_genes_source", default='HPO', help="Resource to extract the seed genes from. either HPO, OT, or GEL (default: HPO)\n\n")
	parser.add_argument("-e", dest="exact_terms", default=None, help="Terms to match against using regular expression matching\n\n")
	parser.add_argument("-z", dest="fuzzy_terms", default=None, help="Terms to match against using NLP\n\n")
	parser.add_argument("-x", dest="exclude_terms", default=None, help="Terms to exclude\n\n")

	parser.add_argument("-o", dest="output_dir", help="Output directory name\n(absolute/relative path e.g. ./CKD, /tmp/Epilepsy-testing, etc.)\nIf it doesn't exist it will automatically be created [Required]\n\n", required=True)
	parser.add_argument('-v', '--verbosity', action="count", help="Print verbose output\n\n")     


	if len(sys.argv)==1:
		parser.print_help(sys.stderr)
		sys.exit(1)

	args = parser.parse_args()      
	exact_terms = args.exact_terms
	features_table = args.features_table   
	null_imp = args.null_imp
	fuzzy_terms = args.fuzzy_terms
	exclude_terms = args.exclude_terms

	disease_name = args.disease_name
	seed_genes_source = args.seed_genes_source
	output_dir = args.output_dir
	verbose = bool(args.verbosity)
	
	profiler = MantisMlProfiler(disease_name, exact_terms, null_imp, features_table, fuzzy_terms, exclude_terms, seed_genes_source, output_dir, verbose=verbose)
	profiler.run_mantis_ml_profiler()


if __name__ == '__main__':
	main()
