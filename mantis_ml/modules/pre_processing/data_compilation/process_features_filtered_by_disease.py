
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import re
import os, sys
import random
from scipy.stats import fisher_exact

from gensim.models import KeyedVectors

from mantis_ml.modules.pre_processing.data_compilation.process_generic_features import ProcessGenericFeatures
from mantis_ml.config_class import Config


class Bioword2vec_embeddings(ProcessGenericFeatures): 
	
	def __init__(self, cfg): 
		
		ProcessGenericFeatures.__init__(self, cfg)
		
		wv_file = os.path.join(self.cfg.data_dir, 'bioword2vec_embeddings/bio_embedding_intrinsic')
		self.embeddings = KeyedVectors.load_word2vec_format(wv_file , binary=True)


class ProcessFeaturesFilteredByDisease(ProcessGenericFeatures):

	def __init__(self, cfg):
		ProcessGenericFeatures.__init__(self, cfg)

	def process_hpo(self, include_terms, exclude_terms, annot_descr, conservative=True, save_to_file=False):
		'''
		Process Human Phenotype Ontology
		:param include_go_terms: list of strings that are queried as substrings of disease-associated terms 
		:param exclude_go_terms: list of strings to be excluded if they are substrings of GO terms
		:param annot_descr: disease/phenotype annotation string
		:param conservative: Boolean
		:return: hpo_df
		'''
		print("\n>> Compiling HPO features...")

		if self.cfg.custom_known_genes_file is None:

			df = None

			if not conservative:
				# more inclusive
				df = pd.read_csv(self.cfg.data_dir / 'HPO/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt', sep='\t')
			else:
				# more conservative (default)
				df = pd.read_csv(self.cfg.data_dir / 'HPO/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt', sep='\t')
				# df.head()

			exclude_pattern = re.compile('|'.join(exclude_terms), re.IGNORECASE)
			if len(exclude_terms) > 0:
				df = df.loc[ ~df['HPO-Term-Name'].str.contains(exclude_pattern)]

			include_pattern = re.compile('|'.join(include_terms), re.IGNORECASE)
			seed_df = df.loc[ df['HPO-Term-Name'].str.contains(include_pattern)]
			# seed_df.shape

			hpo_selected_terms = seed_df['HPO-Term-Name']
			hpo_selected_terms = hpo_selected_terms.unique()


			known_genes_df = pd.DataFrame({'Gene_Name': seed_df['entrez-gene-symbol'].unique(), 'known_gene': 1})


			# TO-DO: test that hiding seed genes works for the non-Generic classifier too
			if self.cfg.hide_seed_genes_ratio > 0:
				sample_known_genes_df = known_genes_df.sample(frac=(1-self.cfg.hide_seed_genes_ratio))

				hidden_seed_genes = pd.Series(list(set(known_genes_df.Gene_Name) - set(sample_known_genes_df.Gene_Name)))
				hidden_seed_genes.to_csv(str(self.cfg.out_data_dir / 'hidden_seed_genes.txt'), index=None, header=False)

				known_genes_df = sample_known_genes_df

			if save_to_file:
				known_genes_df.to_csv(self.cfg.data_dir / ("HPO/compiled_known_" + annot_descr + "_genes.tsv"), sep='\t', index=None)

		else:
		
			try:
				known_genes_df = pd.read_csv(self.cfg.custom_known_genes_file, header=None)
				known_genes_df.columns = ['Gene_Name']
				known_genes_df['known_gene'] = 1
				
				print(known_genes_df.head())
				print(known_genes_df.shape)

				hpo_selected_terms = "Not applicable - Using custom known genes list"
				
			except:
				sys.exit("[Error] Could not read input file with custom known genes list.\nPlease provide a file with HGNC gene names (each in a separate line) with the -k option when calling mantisml or mantisml-profiler")	


		print("Total HPO Genes associated with selected pattern: {0}".format(known_genes_df.shape[0]))

		return known_genes_df, hpo_selected_terms

	
	
	def process_OT(self, include_terms, exclude_terms, annot_descr, save_to_file=False):
		'''
		Process OpenTargets
		:param include_go_terms: list of strings that are queried as substrings of disease-associated terms 
		:param exclude_go_terms: list of strings to be excluded if they are substrings of GO terms
		:param annot_descr: disease/phenotype annotation string
		:param conservative: Boolean
		:return: ot_df
		'''
		print("\n>> Compiling OpenTargets features...")

		if self.cfg.custom_known_genes_file is None:

			df = None

			df = pd.read_csv(self.cfg.data_dir / 'opentargets/ot_associations_for_seed_genes.txt', sep='\t').dropna()
			# df = pd.read_csv(cfg.data_dir / 'opentargets/ot_associations_for_seed_genes.txt', sep='\t').dropna()
			# df.head()

			exclude_pattern = re.compile('|'.join(exclude_terms), re.IGNORECASE)
			if len(exclude_terms) > 0:
				df = df.loc[ ~df['disease_name'].str.contains(exclude_pattern)]

			include_pattern = re.compile('|'.join(include_terms), re.IGNORECASE)
			seed_df = df.loc[ df['disease_name'].str.contains(include_pattern)]
			# seed_df.shape

			ot_selected_terms = seed_df['disease_name']
			ot_selected_terms = ot_selected_terms.unique()
			
			###


			known_genes_df = pd.DataFrame({'Gene_Name': seed_df['entrez-gene-symbol'].unique(), 'known_gene': 1})


			# TO-DO: test that hiding seed genes works for the non-Generic classifier too
			if self.cfg.hide_seed_genes_ratio > 0:
				sample_known_genes_df = known_genes_df.sample(frac=(1-self.cfg.hide_seed_genes_ratio))

				hidden_seed_genes = pd.Series(list(set(known_genes_df.Gene_Name) - set(sample_known_genes_df.Gene_Name)))
				hidden_seed_genes.to_csv(str(self.cfg.out_data_dir / 'hidden_seed_genes.txt'), index=None, header=False)

				known_genes_df = sample_known_genes_df

			if save_to_file:
				known_genes_df.to_csv(self.cfg.data_dir / ("opentargets/compiled_known_" + annot_descr + "_genes.tsv"), sep='\t', index=None)

		else:
		
			try:
				known_genes_df = pd.read_csv(self.cfg.custom_known_genes_file, header=None)
				known_genes_df.columns = ['Gene_Name']
				known_genes_df['known_gene'] = 1
				
				print(known_genes_df.head())
				print(known_genes_df.shape)

				ot_selected_terms = "Not applicable - Using custom known genes list"
				
			except:
				sys.exit("[Error] Could not read input file with custom known genes list.\nPlease provide a file with HGNC gene names (each in a separate line) with the -k option when calling mantisml or mantisml-profiler")


		print("Total OT Genes associated with selected pattern: {0}".format(known_genes_df.shape[0]))

		return known_genes_df, ot_selected_terms


	def process_GEL(self, include_terms, exclude_terms, annot_descr, save_to_file=False):
		'''
		Process Genomics England
		:param include_go_terms: list of strings that are queried as substrings of disease-associated terms 
		:param exclude_go_terms: list of strings to be excluded if they are substrings of GO terms
		:param annot_descr: disease/phenotype annotation string
		:param conservative: Boolean
		:return: gel_df
		'''
		print("\n>> Compiling OpenTargets features...")

		if self.cfg.custom_known_genes_file is None:

			df = None

			df = pd.read_csv(self.cfg.data_dir / 'Genomics_England/input_genes_gel.csv', sep='\t').dropna()
			# df = pd.read_csv(cfg.data_dir / 'opentargets/ot_associations_for_seed_genes.txt', sep='\t').dropna()
			# df.head()

			exclude_pattern = re.compile('|'.join(exclude_terms), re.IGNORECASE)
			if len(exclude_terms) > 0:
				df = df.loc[ ~df['disease_name'].str.contains(exclude_pattern)]

			include_pattern = re.compile('|'.join(include_terms), re.IGNORECASE)
			seed_df = df.loc[ df['disease_name'].str.contains(include_pattern)]
			# seed_df.shape

			gel_selected_terms = seed_df['disease_name']
			gel_selected_terms = gel_selected_terms.unique()
			
			###


			known_genes_df = pd.DataFrame({'Gene_Name': seed_df['entrez-gene-symbol'].unique(), 'known_gene': 1})


			# TO-DO: test that hiding seed genes works for the non-Generic classifier too
			if self.cfg.hide_seed_genes_ratio > 0:
				sample_known_genes_df = known_genes_df.sample(frac=(1-self.cfg.hide_seed_genes_ratio))

				hidden_seed_genes = pd.Series(list(set(known_genes_df.Gene_Name) - set(sample_known_genes_df.Gene_Name)))
				hidden_seed_genes.to_csv(str(self.cfg.out_data_dir / 'hidden_seed_genes.txt'), index=None, header=False)

				known_genes_df = sample_known_genes_df

			if save_to_file:
				known_genes_df.to_csv(self.cfg.data_dir / ("genomics_england/compiled_known_" + annot_descr + "_genes.tsv"), sep='\t', index=None)

		else:
		
			try:
				known_genes_df = pd.read_csv(self.cfg.custom_known_genes_file, header=None)
				known_genes_df.columns = ['Gene_Name']
				known_genes_df['known_gene'] = 1
				
				print(known_genes_df.head())
				print(known_genes_df.shape)

				gel_selected_terms = "Not applicable - Using custom known genes list"
				
			except:
				sys.exit("[Error] Could not read input file with custom known genes list.\nPlease provide a file with HGNC gene names (each in a separate line) with the -k option when calling mantisml or mantisml-profiler")


		print("Total GEL Genes associated with selected pattern: {0}".format(known_genes_df.shape[0]))

		return known_genes_df, gel_selected_terms



	def process_omim(self, pattern):
		df = pd.read_csv(self.cfg.data_dir / ('omim/' + pattern + '_genes.txt'), sep='\t', header=None)
		df.columns = ['Gene_Name']
		df[self.cfg.Y] = 1

		if self.cfg.hide_seed_genes_ratio > 0:
			sample_df = df.sample(frac=(1-self.cfg.hide_seed_genes_ratio))

			hidden_seed_genes = pd.Series(list(set(df.Gene_Name) - set(sample_df.Gene_Name)))
			hidden_seed_genes.to_csv(str(self.cfg.out_data_dir / 'hidden_seed_genes.txt'), index=None, header=False)

			df = sample_df

		return df


	def process_gtex_features(self, wv, save_to_file=False):
		'''
		Get Protein TPM expression and Rank from GTEx
		:param pattern: 
		:return: 
		'''
		print("\n>> Compiling GTEx features...")
		disease_name = self.cfg.disease_name
		exact_terms = self.cfg.exact_terms
		fuzzy_terms = self.cfg.fuzzy_terms
		exclude_terms = self.cfg.exclude_terms

		exact_terms = exact_terms + disease_name
		if not self.cfg.no_nlp:
			fuzzy_terms = fuzzy_terms + disease_name
		else:
			fuzzy_terms = []


		#all_patterns.extend(self.cfg.additional_include_terms)
		#print('All patterns:', all_patterns)
		
		exclude_pattern = re.compile('|'.join(self.cfg.exclude_terms), re.IGNORECASE)

		gtex_fname = 'gtex/RNASeq/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_median_tpm.gct'
		if self.cfg.shuffle_features:
			gtex_fname += '.shuffled'
		
		full_df = pd.read_csv(self.cfg.data_dir / gtex_fname, sep='\t')
		#print(full_df.head())
		all_tissue_cols = full_df.columns.values
		print(all_tissue_cols)
		all_selected_tissue_cols = []

		return_gtex_df = pd.DataFrame()
		
		
		if not self.cfg.generic_classifier:
			
			# NLP/fuzzy matching first
			
			term_ids = []
			wv_term = []
			all_selected_tissue_cols = []
			if len(fuzzy_terms)>0:
				for i, term in enumerate(all_tissue_cols): 

					try: 
						w = [x.lower() for x in re.split('\W+|_', term) if x != '']
						profile = np.mean(wv[w], axis=0)
						wv_term.append([profile])
						term_ids.append(term)
					except: 
						pass

				wv_term = np.concatenate(wv_term)
				term_ids = pd.DataFrame(term_ids, columns=['description'], index = term_ids).assign(ind = list(range(len(term_ids))))

				terms_parsed = [x.lower() for term in fuzzy_terms for x in re.split('\W+|_', term) if x != '']
				profile = np.mean(wv[terms_parsed], axis=0) 
				distance = pd.DataFrame(np.sum(np.abs(wv_term - profile), axis=1), columns = ['distance'], index = term_ids.description.values)

				distance = distance.assign(description = lambda x: x.index.values)
				best_matching_terms = distance.sort_values(['distance']).head(n=self.cfg.ntop_terms_gtex)
				all_selected_tissue_cols = list(best_matching_terms.description.values)


			# exact matching

			for pattern in exact_terms:
				selected_tissue_cols = [c for c in full_df.columns if re.compile(pattern, re.IGNORECASE).search(c)]

				if len(self.cfg.exclude_terms) > 0:
					selected_tissue_cols = [c for c in selected_tissue_cols if not re.compile(exclude_pattern).search(c)]

				if len(selected_tissue_cols) == 0: # return if no matching columns exist
					continue
				else:
					print('\n', pattern, ':', selected_tissue_cols)


				#print('Tissue/Disease pattern:', selected_tissue_cols)
				all_selected_tissue_cols.extend(selected_tissue_cols)


			
			if len(self.cfg.exclude_terms) > 0:
					all_selected_tissue_cols = [c for c in all_selected_tissue_cols if not re.compile(exclude_pattern).search(c)]
					
			df = full_df[['Gene_Name'] + all_selected_tissue_cols]   #['Gene_Name', 'gene_id']
			agg_df = df.groupby('Gene_Name').agg('sum')
			
			pattern = 'included_terms'
			
			total_tissue_expr = 'GTEx_' + pattern + '_TPM_expression'
			sum_gtex_df = pd.DataFrame(agg_df.sum(axis=1), columns=[total_tissue_expr])

			# limit to default HGNC gene-set
			sum_gtex_df = sum_gtex_df.reindex(self.cfg.hgnc_genes_series)
			sum_gtex_df.fillna(0, inplace=True)
			sum_gtex_df.sort_values(by=total_tissue_expr, inplace=True, ascending=False)

			sum_gtex_df.reset_index(inplace=True)
			sum_gtex_df.columns.values[0] = 'Gene_Name'

			# Assign Rank = len(default_gene_set) to all genes with total expression less than the median among all genes
			tissue_rank = 'GTEx_' + pattern + '_Expression_Rank'
			sum_gtex_df[tissue_rank] = sum_gtex_df.index + 1
			sum_gtex_df.loc[sum_gtex_df[total_tissue_expr] < int(sum_gtex_df[total_tissue_expr].median()), tissue_rank] = len(self.cfg.hgnc_genes_series)
			#print(sum_gtex_df.head())

			return_gtex_df = sum_gtex_df

			if save_to_file:
				return_gtex_df.to_csv(self.cfg.data_dir / ('gtex/RNASeq/'+ self.cfg.phenotype + '_GTEx_expression_features.tsv'), sep='\t', index=None)
			# TODO: keep expression in each tissue as a separate feature - do not sum
			


		else:

			full_df.drop(['gene_id'], axis=1, inplace=True)

			full_df = full_df.rename(columns={col: col.replace(' ', '_').replace('-', '') for col in full_df.columns if col != self.cfg.gene_name})
			full_df = full_df.rename(columns={col: col.replace('(', '').replace(')', '').replace('__', '_') for col in full_df.columns if col != self.cfg.gene_name})
			full_df = full_df.rename(columns={col: 'GTEx_' + col + '_TPM_expression' for col in full_df.columns if col != self.cfg.gene_name})

			return_gtex_df = full_df.groupby('Gene_Name').agg('sum')
			return_gtex_df.reset_index(inplace=True)

			print(return_gtex_df.head())
			# TODO: keep expression in each tissue as a separate feature - do not sum


		print('GTEx:', return_gtex_df.head())

		return return_gtex_df, all_selected_tissue_cols, all_tissue_cols




	def process_protein_atlas_features(self, wv, verbose=False, save_to_file=False):
		'''
		Get protein expression levels and RNA TPM for Human Protein Atlas
		:param pattern: 
		:return: 
		'''

		#TODO: Currently collapsing values from multiple tissues -- see if this can be untangled in future version

		print("\n>> Compiling Human Protein Atlas features...")

		tissue_str = self.cfg.phenotype.lower()

		disease_name = self.cfg.disease_name
		exact_terms = self.cfg.exact_terms
		fuzzy_terms = self.cfg.fuzzy_terms
		exclude_terms = self.cfg.exclude_terms

		exact_terms = exact_terms + disease_name
		if not self.cfg.no_nlp:
			fuzzy_terms = fuzzy_terms + disease_name
		else:
			fuzzy_terms = []



		# pattern = re.compile('|'.join(include_terms), re.IGNORECASE)
		# pattern = re.compile(pattern, re.IGNORECASE)

		include_pattern = re.compile('|'.join(exact_terms), re.IGNORECASE)
		exclude_pattern = re.compile('|'.join(exclude_terms), re.IGNORECASE)


		# normal_tissue.tsv.gz
		hpa_fname = 'human_protein_atlas/normal_tissue.tsv.gz'
		if self.cfg.shuffle_features:
			hpa_fname += '.shuffled'
		
		normal_df = pd.read_csv(self.cfg.data_dir / hpa_fname, sep='\t')
		print(normal_df.shape)

		all_normal_tissues = normal_df['Tissue'].unique().tolist()
		if verbose:
			print(all_normal_tissues)

		if not self.cfg.generic_classifier:
			
			print('[normal] Keeping only entries from tissue: {0} ...'.format(include_pattern))
			
			if len(exclude_terms) > 0:
				normal_df = normal_df.loc[~normal_df['Tissue'].str.contains(exclude_pattern)]
			
			# if self.cfg.exact_string_matching == False:
			
			term_ids = []
			wv_term = []
			normal_df_fuzzy = pd.DataFrame([])
			if len(fuzzy_terms)>0:
				for i, term in enumerate(all_normal_tissues): 

					try: 
						w = [x.lower() for x in re.split('\W+|_', term) if x != '']
						profile = np.mean(wv[w], axis=0)
						wv_term.append([profile])
						term_ids.append(term)
					except: 
						pass

				wv_term = np.concatenate(wv_term)
				term_ids = pd.DataFrame(term_ids, columns=['description'], index = term_ids).assign(ind = list(range(len(term_ids))))

				terms_parsed = [x.lower() for term in fuzzy_terms for x in re.split('\W+|_', term) if x != '']
				profile = np.mean(wv[terms_parsed], axis=0) 
				distance = pd.DataFrame(np.sum(np.abs(wv_term - profile), axis=1), columns = ['distance'], index = term_ids.description.values)

				distance = distance.assign(description = lambda x: x.index.values)
				best_matching_terms = distance.sort_values(['distance']).head(n=self.cfg.ntop_terms_protein_atlas)

				all_selected_tissues = list(best_matching_terms.description.values)

				normal_df_fuzzy = normal_df.loc[lambda x: x['Tissue'].isin(all_selected_tissues)]
			
			normal_df_exact = normal_df.loc[normal_df['Tissue'].str.contains(include_pattern)]

			normal_df = pd.concat([normal_df_fuzzy, normal_df_exact])



		print("[normal] Removing entries with Reliability = 'Uncertain...'")
		normal_df = normal_df.loc[ normal_df.Reliability != 'Uncertain']
		print(normal_df.shape)
		selected_normal_tissues = normal_df['Tissue'].unique().tolist()

		normal_df = normal_df.iloc[:, [1,3,4]]
		print(normal_df.head())

		def generic_collapse_normal_tissue_expression(df, target_col):
			"""
			# Category coding for aggregation across multiple cell lines in same tissue
			- Not detected: 0
			- Low: 0
			- Medium: 1
			- High: 1
	
			Rule: Keeping the max. value after aggregation by same Gene Name
			"""
			print("Collapsing normal tissue expression from entries with same gene (keeping highest)...")
			df.replace({target_col: {'Not detected': 0, 'Low': 0, 'Medium': 1, 'High': 1}}, inplace=True)

			df[target_col] = df[target_col].astype(str)

			tmp_df = pd.DataFrame(df.groupby(['Gene name', 'Cell type'])[target_col].agg('|'.join))

			# tmp_df['delim_cnt'] = tmp_df[target_col].apply(lambda x: x.count('|'))
			tmp_df['final_level'] = tmp_df[target_col].apply(lambda x: max(x.split('|')))
			tmp_df['final_level'] = tmp_df['final_level'].astype(int)
			tmp_df.drop([target_col], axis=1, inplace=True)
			print(tmp_df.head())

			# tmp_df = tmp_df.unstack(fill_value=0)
			tmp_df.reset_index(inplace=True)
			tmp_df = tmp_df.pivot(index='Gene name', columns='Cell type', values='final_level')
			tmp_df.fillna(0, inplace=True)
			print(tmp_df.head())
			print(tmp_df.shape)

			tmp_df = tmp_df.rename(columns={col: col.replace(' ', '_') for col in tmp_df.columns})
			tmp_df = tmp_df.rename(columns={col: 'ProteinAtlas_' + col + '_Expr_Flag' for col in tmp_df.columns})
			tmp_df.index.names = ['Gene_Name']
			tmp_df.reset_index(inplace=True)


			tissue_str = 'generic'
			if save_to_file:
				tmp_df.to_csv(self.cfg.data_dir / ('human_protein_atlas/human_protein_atlas_' + tissue_str + '_expression_levels.tsv'), sep='\t', index=None)

			return tmp_df


		def collapse_normal_tissue_expression(df, target_col):
			"""
			# Category coding for aggregation across multiple cell lines in same tissue
			- Not detected: 0
			- Low: 1
			- Medium: 3
			- High: 8
	
			Rule: Keeping the max. value after aggregation by same Gene Name
			"""
			print("Collapsing normal tissue expression from entries with same gene (keeping highest)...")
			df.replace({target_col: {'Not detected': 0, 'Low': 1, 'Medium': 3, 'High': 8}}, inplace=True)

			df[target_col] = df[target_col].astype(str)

			tmp_df = pd.DataFrame(df.groupby('Gene name')[target_col].agg('|'.join))

			tmp_df['delim_cnt'] = tmp_df[target_col].apply(lambda x: x.count('|'))
			tmp_df['final_level'] = tmp_df[target_col].apply(lambda x: max(x.split('|')))
			print(tmp_df.head())
			tmp_df['Gene name'] = tmp_df.index.copy()

			final_df = tmp_df[['Gene name', 'final_level']].copy()
			final_df.rename(columns={'final_level': target_col}, inplace=True)

			final_df.replace({target_col: {'0': 'Not_detected', '1': 'Low', '3': 'Medium', '8': 'High'}}, inplace=True)
			final_df.columns = ['Gene_Name', 'ProteinAtlas_gene_expr_levels']
			# print(final_df.head())

			if save_to_file:
				final_df.to_csv(self.cfg.data_dir / ('human_protein_atlas/human_protein_atlas_' + tissue_str + '_expression_levels.tsv'), sep='\t', index=None)
			return final_df

		target_col = 'Level'
		expr_levels_df = None
		if not self.cfg.generic_classifier:
			expr_levels_df = collapse_normal_tissue_expression(normal_df, target_col)
		else:
			expr_levels_df = generic_collapse_normal_tissue_expression(normal_df, target_col)
		print(expr_levels_df.head())
		print(expr_levels_df.shape)



		# =========== rna_tissue.tsv.gz ============
		rna_df = pd.read_csv(self.cfg.data_dir / 'human_protein_atlas/rna_tissue.tsv.gz', sep='\t')
		# print(rna_df.head())
		print(rna_df.shape)

		all_rna_samples = rna_df['Sample'].unique().tolist()

		if not self.cfg.generic_classifier:
			print('[rna] Keeping only entries from tissue: {0} ...'.format(include_pattern))
			# rna_df = rna_df.loc[rna_df.Sample.str.contains(pattern)]
			if len(exclude_terms) > 0:
				rna_df = rna_df.loc[~rna_df['Sample'].str.contains(exclude_pattern)]
			rna_df = rna_df.loc[rna_df['Sample'].str.contains(include_pattern)]

		selected_rna_samples = rna_df['Sample'].unique().tolist()

		def generic_collapse_rna_expression(df, target_col):
			"""
			# Aggregate TPM values across multiple entries of same gene
			"""

			print("Collapsing rna tissue expression from entries with same gene (sum)...")
			print(df.shape)
			tmp_df = df.groupby(['Gene name', 'Sample']).sum()
			print(tmp_df.head())

			tmp_df.reset_index(inplace=True)
			tmp_df = tmp_df.pivot(index='Gene name', columns='Sample', values='Value')
			tmp_df.fillna(0, inplace=True)
			print(tmp_df.head())
			print(tmp_df.shape)

			tmp_df = tmp_df.rename(columns={col: col.replace(' ', '_') for col in tmp_df.columns})
			tmp_df = tmp_df.rename(columns={col: 'ProteinAtlas_' + col + '_RNA_Expr_TPM' for col in tmp_df.columns})
			tmp_df.index.names = ['Gene_Name']
			tmp_df.reset_index(inplace=True)

			if save_to_file:
				tissue_str = 'generic'
				tmp_df.to_csv(self.cfg.data_dir / ('human_protein_atlas/human_protein_atlas_' + tissue_str + '_rna_expression_tpm.tsv'),
							  sep='\t', index=None)

			return tmp_df

		def collapse_rna_expression(df, target_col):
			"""
			# Aggregate TPM values across multiple entries of same gene
			"""

			print("Collapsing rna tissue expression from entries with same gene (sum)...")
			print(df.shape)
			df = df.groupby('Gene name').sum()

			df['Gene name'] = df.index.copy()
			df = df[['Gene name', target_col]]
			df.columns = ['Gene_Name', 'ProteinAtlas_RNA_expression_TMP']

			if save_to_file:
				df.to_csv(self.cfg.data_dir / ('human_protein_atlas/human_protein_atlas_' + tissue_str + '_rna_expression_tpm.tsv'), sep='\t', index=None)

			return df

		target_col = 'Value'
		expr_tpm_df = None
		if not self.cfg.generic_classifier:
			expr_tpm_df = collapse_rna_expression(rna_df, target_col)
		else:
			expr_tpm_df = generic_collapse_rna_expression(rna_df, target_col)
		print(expr_tpm_df.shape)
		print(expr_tpm_df.head())

		# prot_atlas_df = expr_tpm_df # To keep only RNA expression TPM values
		prot_atlas_df = pd.merge(expr_levels_df, expr_tpm_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')


		return prot_atlas_df, selected_normal_tissues, all_normal_tissues, selected_rna_samples, all_rna_samples



		
		

	def process_msigdb_go_features(self, wv, seed_genes, min_go_set_length=150, verbose=False, save_to_file=False):
		'''
		Get GO information from msigdb for genes associated with particular terms
		
		:param include_go_terms: list of strings that are queried as substrings of GO terms 
		:param exclude_go_terms: list of strings to be excluded if they are substrings of GO terms
		:return: 
		'''
		print("\n>> Compiling MsigDB GO features...")
		

		generic_go_file = self.cfg.data_dir / ('msigdb/tables_per_gene_set/generic.min_go_set_len' + str(min_go_set_length) + '_GO_Features.tsv')
		# if os.path.exists(generic_go_file) and self.cfg.generic_classifier:
		# 	msigdb_go_df = pd.read_csv(generic_go_file, sep='\t')
		# 	return msigdb_go_df

		full_go_file = (self.cfg.data_dir / 'msigdb/tables_per_gene_set/c5.all.v6.2.symbols.gmt')
		super_go_file = (self.cfg.data_dir / 'msigdb/tables_per_gene_set/go_reduced.txt')
		
		if self.cfg.shuffle_features:
			full_go_file = str(full_go_file) + '.shuffled'
			super_go_file = str(super_go_file) + '.shuffled'

		disease_name = self.cfg.disease_name
		exact_terms = self.cfg.exact_terms
		fuzzy_terms = self.cfg.fuzzy_terms
		exclude_terms = self.cfg.exclude_terms

		exact_terms = exact_terms + disease_name
		if not self.cfg.no_nlp:
			fuzzy_terms = fuzzy_terms + disease_name
		else:
			fuzzy_terms = []

		
		
		#if self.cfg.exact_string_matching == True:

		gene_lists_per_go_term = dict()
		if not self.cfg.generic_classifier:
			for t in exact_terms:
				gene_lists_per_go_term[t] = []


		selected_go_terms_exact = []
		cnt = 0
		with open(full_go_file) as fh:
			for line in fh:
				line = line.rstrip()
				vals = line.split('\t')
				del vals[1]

				cur_go_field = vals[0]
				if len(exclude_terms) > 0:
					if any(re.compile(excl_term, re.IGNORECASE).search(cur_go_field) for excl_term in exclude_terms):
						continue

				for t in range(len(exact_terms)):
					incl_term = exact_terms[t]
					if (re.compile(incl_term, re.IGNORECASE).search(cur_go_field)):
						selected_go_terms_exact.append(cur_go_field)
						genes = vals[1:]
						if not self.cfg.generic_classifier:
							gene_lists_per_go_term[incl_term].extend(genes)
						else:
							gene_lists_per_go_term[cur_go_field] = genes

						if verbose:
							print(cur_go_field)


		go_terms = [] 
		go_terms_genes = {} 
		with open(full_go_file) as fh:
			for line in fh:
				line = line.rstrip()
				vals = line.split('\t')
				cur_go_field = vals[0]
				go_terms_genes[cur_go_field] = vals[2:]
				go_terms.append(cur_go_field)
		
		all_genes = set([])
		for term in list(go_terms_genes.keys()): 
			all_genes = all_genes.union(set(go_terms_genes[term]))
		
		### check enrichment in seed genes
		enrichment = [] 
		for term in list(go_terms_genes.keys()): 
			if len(set(go_terms_genes[term]).intersection(set(seed_genes))) > 1: 
				odds_ratio, pvalue = fisher_exact(
					np.array([
						[
							len(set(go_terms_genes[term]).intersection(set(seed_genes))), 
							len(set(go_terms_genes[term]).difference(set(seed_genes))), 
						], 
						[
							len(set(seed_genes).difference(set(go_terms_genes[term]))), 
							len(all_genes.difference(set(go_terms_genes[term]).union(set(seed_genes))))
						]
					]), 
					alternative = 'greater'
				)
			else: 
				odds_ratio = 0
				pvalue = 1
			
			enrichment.append([term, odds_ratio, pvalue])
			
		enrichment = pd.DataFrame(enrichment, columns = ['term', 'odds_ratio', 'pvalue'])
		enrichment = enrichment.sort_values(['pvalue']).loc[lambda x: x.pvalue < 0.05]
		enrichment = enrichment.head(n=np.min([len(enrichment), self.cfg.ntop_terms_enriched]))
		
		top_go = list(enrichment.term.values)


		term_ids = []
		wv_term = []
		for i, term in enumerate(go_terms): 

			try: 
				w = [x.lower() for x in re.split('\W+|_', term) if x != '']
				profile = np.mean(wv[w], axis=0)
				wv_term.append([profile])
				term_ids.append(term)
			except: 
				pass

		wv_term = np.concatenate(wv_term)
		term_ids = pd.DataFrame(term_ids, columns=['description'], index = term_ids).assign(ind = list(range(len(term_ids))))

		selected_go_terms_fuzzy = []
		if len(fuzzy_terms)>0:
			terms_parsed = [x.lower() for term in fuzzy_terms for x in re.split('\W+|_', term) if x != '']
			profile = np.mean(wv[terms_parsed], axis=0) 
			distance = pd.DataFrame(np.sum(np.abs(wv_term - profile), axis=1), columns = ['distance'], index = term_ids.description.values)

			distance = distance.assign(description = lambda x: x.index.values)
			best_matching_terms = distance.sort_values(['distance']).head(n=self.cfg.ntop_terms_msigdb)
			selected_go_terms_fuzzy = list(best_matching_terms.description.values)
			if len(exclude_terms) > 0:
				exclude_pattern = re.compile('|'.join(exclude_terms), re.IGNORECASE)
				selected_go_terms_fuzzy = [c for c in selected_go_terms_fuzzy if not re.compile(exclude_pattern).search(c)]
		
		selected_go_terms = selected_go_terms_exact + selected_go_terms_fuzzy

		gene_lists_per_go_term = {} 
		for term in selected_go_terms: 
			gene_lists_per_go_term[term] = go_terms_genes[term]
			
		
		super_go_terms = [] 
		gene_lists_per_super_go_term = {} 
		with open(super_go_file) as fh:
			for line in fh:
				line = line.rstrip()
				vals = line.split('\t')
				cur_go_field = vals[0]
				gene_lists_per_super_go_term[cur_go_field] = vals[2:]
				super_go_terms.append(cur_go_field)
				

		new_gene_lists_per_go_term = dict()
		for term in gene_lists_per_go_term.keys():
			new_gene_lists_per_go_term[term] = list(set(gene_lists_per_go_term[term]))
			if  self.cfg.generic_classifier and (len(new_gene_lists_per_go_term[term]) < min_go_set_length):
				del new_gene_lists_per_go_term[term]
		
		for term in gene_lists_per_super_go_term.keys():
			new_gene_lists_per_go_term[term] = list(set(gene_lists_per_super_go_term[term]))
		
		if len(top_go) > 0: 
			for term in top_go:
				new_gene_lists_per_go_term[term] = go_terms_genes[term]



		gene_lists_per_go_term = new_gene_lists_per_go_term.copy()

		msigdb_go_df = pd.DataFrame()
		for term in gene_lists_per_go_term.keys():
			tmp_df = pd.DataFrame({'Gene_Name': gene_lists_per_go_term[term], term: 1})
			tmp_df = tmp_df[['Gene_Name', term]]
			print(tmp_df.shape)

			if msigdb_go_df.shape[0] > 0:
				msigdb_go_df = pd.merge(msigdb_go_df, tmp_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
			else:
				msigdb_go_df = tmp_df

		msigdb_go_df.fillna(0, inplace=True)
		# print(msigdb_go_df.loc[msigdb_go_df.Gene_Name.isin(['PKD1', 'PKD2', 'NOTCH1'])])


		go_cols = [(x.replace(' ', '_')) for x in msigdb_go_df.columns if x != 'Gene_Name']
		if not self.cfg.generic_classifier:
			go_cols = [('GO_' + x) for x in msigdb_go_df.columns if x != 'Gene_Name']


		go_cols = ['Gene_Name'] + go_cols
		print(go_cols)
		msigdb_go_df.columns = go_cols

		file_prefix = self.cfg.phenotype
		if self.cfg.generic_classifier:
			file_prefix = 'generic.min_go_set_len' + str(min_go_set_length)

		if save_to_file:
			msigdb_go_df.to_csv(self.cfg.data_dir / ('msigdb/tables_per_gene_set/' + file_prefix + '_GO_Features.tsv'), sep='\t', index=None)


		return msigdb_go_df, selected_go_terms



	def process_mgi_features(self, wv, verbose=False, save_to_file=False):
		'''
		Retrieve human genes with mouse orthologs that have a particular phenotype
		:param include_terms: terms to look for in the mammalian phenotype
		:param exclude_terms: terms to exclude from mammalian phenotype search
		:return: 
		'''
		print(">> Compiling MGI phenotype-associated genes features...")

		# include_terms.extend(self.cfg.additional_include_terms)

		disease_name = self.cfg.disease_name
		exact_terms = self.cfg.exact_terms
		fuzzy_terms = self.cfg.fuzzy_terms
		exclude_terms = self.cfg.exclude_terms

		exact_terms = exact_terms + disease_name
		if not self.cfg.no_nlp:
			fuzzy_terms = fuzzy_terms + disease_name
		else:
			fuzzy_terms = []



		mgi_fname = self.cfg.data_dir / 'mgi/hmd_human_pheno.processed.rpt'
		if self.cfg.shuffle_features:
			mgi_fname = str(mgi_fname) + '.shuffled'
		
		query_human_pheno_df = pd.read_csv(mgi_fname, sep='\t')
		query_human_pheno_df.fillna('', inplace=True)

		# exclude query terms
		exclude_pattern = re.compile('|'.join(exclude_terms), re.IGNORECASE)
		if len(exclude_terms) > 0:
			query_human_pheno_df = query_human_pheno_df.loc[~query_human_pheno_df['human_phenotypes'].str.contains(exclude_pattern)]

		
		#######
		
		# if self.cfg.exact_string_matching == True:

		human_phenotypes = np.unique([y for x in query_human_pheno_df['human_phenotypes'].values for y in x.split('|')])

		
		# include query terms
		include_pattern = re.compile('|'.join(exact_terms), re.IGNORECASE)
		query_human_pheno_df_exact = query_human_pheno_df.loc[
		query_human_pheno_df['human_phenotypes'].str.contains(include_pattern)]

		selected_mgi_phenotypes_exact = query_human_pheno_df_exact['human_phenotypes'].unique().tolist()
			
			

		term_ids = []
		wv_term = []
		for i, term in enumerate(human_phenotypes): 

			try: 
				w = [x.lower() for x in re.split('\W+|_', term) if x != '']
				profile = np.mean(wv[w], axis=0)
				wv_term.append([profile])
				term_ids.append(term)
			except: 
				pass

		selected_mgi_phenotypes_fuzzy = []
		if len(fuzzy_terms)>0:
			wv_term = np.concatenate(wv_term)
			term_ids = pd.DataFrame(term_ids, columns=['description'], index = term_ids).assign(ind = list(range(len(term_ids))))

			terms_parsed = [x.lower() for term in fuzzy_terms for x in re.split('\W+|_', term) if x != '']
			profile = np.mean(wv[terms_parsed], axis=0) 
			distance = pd.DataFrame(np.sum(np.abs(wv_term - profile), axis=1), columns = ['distance'], index = term_ids.description.values)

			distance = distance.assign(description = lambda x: x.index.values)
			best_matching_terms = distance.sort_values(['distance']).head(n=self.cfg.ntop_terms_mgi)
			selected_go_terms = list(best_matching_terms.description.values)

			query_human_pheno_df = query_human_pheno_df.loc[lambda x: [np.any([y in z for y in selected_go_terms]) for z in x['human_phenotypes'].values]]

			selected_mgi_phenotypes_fuzzy = query_human_pheno_df['human_phenotypes'].unique().tolist()

		selected_mgi_phenotypes = selected_mgi_phenotypes_exact + selected_mgi_phenotypes_fuzzy
		
			
		####################
			
		if verbose:
			print(selected_mgi_phenotypes)

		mgi_pheno_genes = query_human_pheno_df['Human Marker Symbol'].unique()
		mgi_pheno_df = pd.DataFrame({'Gene_Name': mgi_pheno_genes, 'MGI_mouse_knockout_feature': 1})
		

		return  mgi_pheno_df, selected_mgi_phenotypes



	def process_inweb(self, seed_genes):
		print("\n>> Compiling Graph level features...")
		
		full_df = pd.read_csv(self.cfg.bikg_net, sep = '\t')
		inferred_df = pd.DataFrame(full_df.groupby('source_label')['target_label'].apply(list))
		inferred_df.columns = ['interacting_genes']
		inferred_df.index.name = 'Gene_Name'
		inferred_df.reset_index(inplace=True)
		# agg_experim_df.set_index('Gene_Name',inplace=True)
		inferred_df
		
		def get_seed_genes_overlap(interacting_genes):
			if type(interacting_genes) is str:
				interacting_genes = eval(interacting_genes)
			elif type(interacting_genes) is list:
				pass
			else:
				raise ValueError('Unrecognised type in interacting genes (seed gene overlap calculation)')
			overlapping_genes = list(set(list(interacting_genes)) & set(seed_genes))
			perc_overlap = len(overlapping_genes) / len(interacting_genes)
		
			return perc_overlap
		
		# fname = cfg.data_dir / 'in_web' / 'inferred_pairwise_interactions_bikg.tsv'
		if self.cfg.shuffle_features:
			raise NotImplementedError('Need to shuffle graph before loading in')
			# fname = str(fname) + '.shuffled'
		# inferred_df = pd.read_csv(fname, sep='\t', index_col=None)
		# inferred_df = agg_experim_df
		
		inferred_df['Seed_genes_overlap'] = inferred_df['interacting_genes'].apply(get_seed_genes_overlap)
		inferred_df.drop(['interacting_genes'], axis=1, inplace=True)
		print(inferred_df.head())
		
		inweb_df = inferred_df
		
		inweb_df.fillna(0, inplace=True)
		
		return inweb_df



	def process_single_cell_features(self, wv, verbose=False, save_to_file=False):
		'''
		Retrieve single cell data
		:return: 
		'''
		print(">> Compiling Single Cell genes features...")

		# include_terms.extend(self.cfg.additional_include_terms)

		def get_tissue(disease_name_or_names):
		    """
		    Pass in a string e.g. 'disease name' or list of strings in ['disease name1','disease name2',...]
		    """
		
		    tissue_names = [
		        'Blood',
		        'Brain', 
		        'Breast',
		        'Colon', 
		        'Esophagus',
		        'Heart',
		        'Kidney',
		        'Liver', 
		        'Lung',
		        'Lymph node', 
		        'Marrow',
		        'Muscle', 
		        'Ovary',
		        'Pancreas',
		        'Prostate',
		        'Skin', 
		        'Spleen',
		        'Stomach',
		        'Testis',
		    ]
		    
		
		    if type(disease_name_or_names) is str:
		        disease_names = [disease_name_or_names]
		    else:
		        disease_names = disease_name_or_names
		    
		    term_ids = []
		    wv_term = []
		    for i, term in enumerate(disease_names): 
		    
		    	try: 
		    		w = [x.lower() for x in re.split('\W+|_', term) if x != '']
		    		profile = np.mean(wv[w], axis=0)
		    		wv_term.append([profile])
		    		term_ids.append(term)
		    	except: 
		    		pass
		    
		    wv_term = np.concatenate(wv_term)
		    term_ids = pd.DataFrame(term_ids, columns=['description'], index = term_ids).assign(ind = list(range(len(term_ids))))
		    
		    tissue_ids = []
		    wv_tissue = []
		    for i, term in enumerate(tissue_names): 
		    
		    	try: 
		    		w = [x.lower() for x in re.split('\W+|_', term) if x != '']
		    		profile = np.mean(wv[w], axis=0)
		    		wv_tissue.append([profile])
		    		tissue_ids.append(term)
		    	except: 
		    		pass
		    
		    wv_tissue = np.concatenate(wv_tissue)
		    tissue_ids = pd.DataFrame(tissue_ids, columns=['description'], index = tissue_ids).assign(ind = list(range(len(tissue_ids))))
		    
		    
		    dist_all = [] 
		    # idisease = term_ids.description.values[0]
		    for idisease in term_ids.description.values: 
		    	# itissue = tissue_ids.description.values[0]
		    	dist = []
		    	for itissue in tissue_ids.description.values: 
		    		dist.append([
		    			idisease, 
		    			itissue,
		    			np.sum(np.abs(wv_term[term_ids.loc[idisease].ind] - wv_tissue[tissue_ids.loc[itissue].ind]))
		    		])
		    	dist = pd.DataFrame(dist, columns = ['disease', 'tissue', 'dist']).sort_values(['dist']).head(n=1)
		    	dist_all.append(dist)
		    	
		    dist_all = pd.concat(dist_all)
		    #dist_all = dist_all.assign(disease = lambda x: ['<{}>'.format(y) for y in x.disease.values])
		    
		    dist_all.tissue = dist_all.tissue.replace('Lymph node','Lymph')
		
		    if len(disease_names)==1:
		        return dist_all.iloc[0].tissue
		    else:
		        return dist_all



		
		disease_name = self.cfg.disease_name
		tissue = get_tissue(disease_name)

		print('>>> tissue : ', tissue)

		

		if self.cfg.no_nlp:
			raise NotImplementedError('Need to consider how to handle single cell when NLP is switched off')


		single_cell_fname = self.cfg.data_dir / ("single_cell_expression/%s.tsv" % tissue)
		if self.cfg.shuffle_features:
			raise NotImplementedError('Need to consider how to handle single cell when shuffling features')



		features_table = pd.read_csv(single_cell_fname, sep = '\t')
		if 'gene_id' in features_table.columns.values: 
			features_table = features_table.drop(['gene_id'], axis=1)

		if 'gene' in features_table.columns.values: 
			features_table = features_table.rename(columns = {'gene': 'Gene_Name'})
			


		# if self.cfg.null_imp == 'zero': 
		# 	features_table = features_table.fillna(0)
		# elif self.cfg.null_imp == 'median': 
		# 	features_table = features_table.fillna(features_table.median())
		# else: 
		# 	imp = pd.read_csv(self.cfg.null_imp, sep = '\t', header = None)
		# 	imp.columns = ['feature', 'imp']
		# 	for col in features_table.select_dtypes('number').columns.values: 
		# 		if col in imp.feature.values: 
		# 			imp_method = imp.loc[lambda x: x.feature == col].imp.values[0]
		# 			if imp_method == 'median': 
		# 				features_table[col] = features_table[col].fillna(features_table[col].median())
		# 			else: 
		# 				features_table[col] = features_table[col].fillna(0)
		# 		else: 
		# 			features_table[col] = features_table[col].fillna(features_table[col].median())

		# mgi_pheno_genes = query_human_pheno_df['Human Marker Symbol'].unique()
		# single_cell_df = pd.DataFrame({'Gene_Name': mgi_pheno_genes, 'MGI_mouse_knockout_feature': 1})

		single_cell_df = features_table.groupby(['Gene_Name']).mean().reset_index()		


		return  single_cell_df



	def run_all(self, wv):

		disease_name = self.cfg.disease_name
		exact_terms = self.cfg.exact_terms
		fuzzy_terms = self.cfg.fuzzy_terms
		exclude_terms = self.cfg.exclude_terms
		
		# disease_name = cfg.disease_name
		# exact_terms = cfg.exact_terms
		# fuzzy_terms = cfg.fuzzy_terms
		# exclude_terms = cfg.exclude_terms

		exact_terms = exact_terms + disease_name
		if not self.cfg.no_nlp:
			fuzzy_terms = fuzzy_terms + disease_name
		else:
			fuzzy_terms = []


		#seed_include_terms = self.cfg.seed_include_terms
		#exclude_terms = self.cfg.exclude_terms

		# Get Seed genes (positive data points)
		if not self.cfg.generic_classifier:
			if self.cfg.seed_genes_source == 'HPO': 
				# HPO
				seed_genes_df, _ = self.process_hpo(disease_name, exclude_terms, self.cfg.phenotype)
				print('HPO:', seed_genes_df.shape)
			elif self.cfg.seed_genes_source == 'OT': 
				seed_genes_df, _ = self.process_OT(disease_name, exclude_terms, self.cfg.phenotype)
				print('OT:', seed_genes_df.shape)
			elif self.cfg.seed_genes_source == 'GEL': 
				seed_genes_df, _ = self.process_GEL(disease_name, exclude_terms, self.cfg.phenotype)
				print('OT:', seed_genes_df.shape)
			else: 
				seed_genes_df, _ = self.process_hpo(disease_name, exclude_terms, self.cfg.phenotype)
				print('HPO:', seed_genes_df.shape)
		else:
			seed_genes_df = self.process_omim(self.cfg.generic_classifier)

		print(seed_genes_df.shape)
		print(seed_genes_df.head())
		if seed_genes_df.shape[0] == 0:
			sys.exit('[Error] No seed genes found for current terms:' + ','.join(disease_name))


		# GTEx
		gtex_df, _, _ = self.process_gtex_features(wv)
		print('GTEx:', gtex_df.shape)


		# GWAS
		if not self.cfg.generic_classifier:

			tissue_gwas_df = self.process_gwas_features(wv, disease_name, exact_terms, fuzzy_terms, exclude_terms, search_term=self.cfg.phenotype)
			print('GWAS:', tissue_gwas_df.shape)


		# Protein Atlas

		prot_atlas_df, _, _, _, _ = self.process_protein_atlas_features(wv)
		#prot_atlas_df = pd.DataFrame()
		print('Human Protein Atlas:', prot_atlas_df.shape)
		print(prot_atlas_df.shape)


		# MSigDB
		seed_genes = seed_genes_df['Gene_Name'].tolist()
		if self.cfg.generic_classifier:
			msigdb_include_terms = ['.*']
			#exclude_terms = []
		msigdb_go_df, _ = self.process_msigdb_go_features(wv, seed_genes)
		print('MSigDB:', msigdb_go_df.shape)
		print(msigdb_go_df.iloc[:, 0:10].head())
		print(msigdb_go_df.columns.values[:10])


		# MGI
		if not self.cfg.generic_classifier:

			mgi_pheno_df, _ = self.process_mgi_features(wv)
			print('MGI:', mgi_pheno_df.shape)


		# InWeb_IM
		inweb_df = self.process_inweb(seed_genes)
		print('InWeb_IM:', inweb_df.shape)


		# Single cell
		single_cell_df = self.process_single_cell_features(wv)
		print('Single cell:', single_cell_df.shape)
		

		#breakpoint()

		print("\n>> Merging all data frames together...")
		filtered_by_disease_features_df = pd.merge(inweb_df, seed_genes_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')


		# filtered_by_disease_features_df = pd.concat([inweb_df.set_index('Gene_Name'), seed_genes_df.set_index('Gene_Name'), single_cell_df.set_index('Gene_Name')],axis=1).reset_index()

		print(filtered_by_disease_features_df.shape)
		if gtex_df.shape[0] > 0:
			filtered_by_disease_features_df = pd.merge(filtered_by_disease_features_df, gtex_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
			print(filtered_by_disease_features_df.shape)
		if prot_atlas_df.shape[0] > 0:
			filtered_by_disease_features_df = pd.merge(filtered_by_disease_features_df, prot_atlas_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
			print(filtered_by_disease_features_df.shape)

		if msigdb_go_df.shape[0] > 0:
			filtered_by_disease_features_df = pd.merge(filtered_by_disease_features_df, msigdb_go_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
			print(filtered_by_disease_features_df.shape)

		if not self.cfg.generic_classifier:
			if tissue_gwas_df.shape[0] > 0:
				filtered_by_disease_features_df = pd.merge(filtered_by_disease_features_df, tissue_gwas_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
				print(filtered_by_disease_features_df.shape)

			if mgi_pheno_df.shape[0] > 0:
				filtered_by_disease_features_df = pd.merge(filtered_by_disease_features_df, mgi_pheno_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
				print(filtered_by_disease_features_df.shape)

				

		# ---------------------------------------------------------
		# Impute 'known_gene', 'GO_*', 'MGI_mouse_knockout_feature' & 'ProteinAtlas_gene_expr_levels' with zero, for all genes that don't have a '1' value:
		# these values are not missing data but rather represent a 'False'/zero feature value.
		# filtered_by_disease_features_df.drop(['Inferred_seed_genes_overlap', 'Experimental_seed_genes_overlap'], axis=1, inplace=True)
		filtered_by_disease_features_df['known_gene'].fillna(0,inplace=True)
		
		inweb_df_cols = [c for c in inweb_df.columns.values if c != 'Gene_Name']
		for c in inweb_df_cols:
			if c in filtered_by_disease_features_df.columns:
				filtered_by_disease_features_df[c].fillna(0, inplace=True)

		protatlas_cols = [c for c in prot_atlas_df.columns.values if c != 'Gene_Name']
		for c in protatlas_cols:
			if c in filtered_by_disease_features_df.columns:
				filtered_by_disease_features_df[c].fillna(0, inplace=True)

		go_cols = [c for c in msigdb_go_df.columns.values if c != 'Gene_Name']
		for c in go_cols:
			if c in filtered_by_disease_features_df.columns:
				filtered_by_disease_features_df[c].fillna(0,inplace=True)

		if 'MGI_mouse_knockout_feature' in filtered_by_disease_features_df.columns:
			filtered_by_disease_features_df['MGI_mouse_knockout_feature'].fillna(0,inplace=True)
		# ---------------------------------------------------------


		if single_cell_df.shape[0] > 0:
			filtered_by_disease_features_df = pd.merge(filtered_by_disease_features_df, single_cell_df, how = 'outer', on = 'Gene_Name')
			print(filtered_by_disease_features_df.shape)

		

		if self.cfg.random_seeds:
			print(filtered_by_disease_features_df.loc[ filtered_by_disease_features_df.known_gene == 1, :].shape)

			total_seed_genes = filtered_by_disease_features_df.loc[ filtered_by_disease_features_df.known_gene == 1, :].shape[0]
			# reset known genes labels with '0' value for all genes
			filtered_by_disease_features_df.loc[:, 'known_gene'] = 0
			print(filtered_by_disease_features_df.loc[ filtered_by_disease_features_df.known_gene == 1, :].shape)

			# select random indexes
			random_seed_indexes = np.random.choice(list(range(filtered_by_disease_features_df.shape[0])), size=total_seed_genes, replace=False).tolist()
			print(len(random_seed_indexes))

			# assign '1' value for 'known_gene' label to random genes indicated by the generated random indexes
			filtered_by_disease_features_df.loc[ random_seed_indexes, 'known_gene'] = 1
			print(filtered_by_disease_features_df.loc[ filtered_by_disease_features_df.known_gene == 1, :].shape)



		filtered_by_disease_features_df.to_csv(self.cfg.filtered_by_disease_feature_table, sep='\t', index=None)
		print("Saved to {0}".format(self.cfg.filtered_by_disease_feature_table))
		print(filtered_by_disease_features_df.shape)

		duplicate_gene_names = filtered_by_disease_features_df.Gene_Name[filtered_by_disease_features_df.Gene_Name.duplicated()].unique()
		if len(duplicate_gene_names) > 0:
			print('[Error] Duplicate Gene Names:')
			print(duplicate_gene_names)
			sys.exit(-1)


if __name__ == '__main__':

	config_file = '../../../config.yaml'#sys.argv[1] #
	disease_name = 'kidney disease'
	exact_terms = None
	null_imp = None
	fuzzy_terms = None
	exclude_terms = None
	features_table = None
	
	output_dir = '../../../../test'
	
	# cfg = Config(config_file, 'example-input')
	# config_file = '../../../../config.yaml'
	# cfg = Config(config_file)
	# cfg = Config(config_file, output_dir)
	cfg = Config(
    disease_name,
    exact_terms,
    null_imp,
    features_table,
    fuzzy_terms,
    exclude_terms,
    output_dir,
    verbose=False,
	)
	cfg.no_nlp = False
	cfg.shuffle_features = False
	cfg.exp_inter = cfg.data_dir  / 'in_web/experimental_pairwise_interactions_bikg.tsv'
	cfg.inf_inter = cfg.data_dir  / 'in_web/inferred_pairwise_interactions_bikg.tsv'

	
	#Config(disease_name, exact_terms, fuzzy_terms, exclude_terms, output_dir)
	cfg.seed_genes_source = 'OT'
	wv = Bioword2vec_embeddings(cfg)
	proc = ProcessFeaturesFilteredByDisease(cfg)
	
	proc.run_all(wv.embeddings)
# 	gtex_df, all_selected_tissue_cols, _ = proc.process_gtex_features(wv.embeddings)
# 	gtex_df.shape
# 	gtex_df.head()
# 	all_selected_tissue_cols
	
# 	mgi_pheno_df, selected_mgi_phenotypes = proc.process_mgi_features(wv.embeddings)
# 	mgi_pheno_df.head()
# 	mgi_pheno_df.shape
# 	proc.run_all(wv.embeddings)
	
# 	known_genes_df, hpo_selected_terms =proc.process_hpo(disease_name, [], '', conservative=True, save_to_file=False)
# 	known_genes_df.shape
# 	len(hpo_selected_terms)

# 	include_terms = [disease_name]
# 	exclude_terms = []
# 	annot_descr = ''
# 	known_genes_df, hpo_selected_terms =proc.process_OT(disease_name, [], '', save_to_file=False)
# 	known_genes_df.shape
# 	len(hpo_selected_terms)
	
# 	prot_atlas_df, selected_normal_tissues, all_normal_tissues, selected_rna_samples, all_rna_samples = proc.process_protein_atlas_features(wv.embeddings)
# 	prot_atlas_df.shape
# 	prot_atlas_df.head()
# 	len(selected_normal_tissues)
# 	all_normal_tissues
# 	len(selected_rna_samples)
# 	len(all_rna_samples)
	
# 	msigdb_go_df, selected_go_terms = proc.process_msigdb_go_features(wv.embeddings, seed_genes)
# 	msigdb_go_df.head()
# 	msigdb_go_df.shape
