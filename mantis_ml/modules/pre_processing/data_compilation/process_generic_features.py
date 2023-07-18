import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import sys
import re
from pathlib import Path
import numpy as np
import re
import os, sys
import random

from gensim.models import KeyedVectors


from mantis_ml.config_class import Config




def explode(df, column, sep='|', keep=False):
	'''
	Split the values of a column and expand so the new DataFrame has one value per row. 
	Filters rows where the column is missing.
	
	:param df: pandas.DataFrame - dataframe with a column to split and expand
	:param column: str - the column to split and expand
	:param sep: str - the string used to split the column's values
	:param keep: bool -whether to retain the presplit value as it's own row
	:return: pandas.DataFrame - dataframe with the same columns as `df`.
	'''
	indexes = list()
	new_values = list()
	df = df.dropna(subset=[column])
	for i, presplit in enumerate(df[column].astype(str)):
		values = presplit.split(sep)
		if keep and len(values) > 1:
			indexes.append(i)
			new_values.append(presplit)
		for value in values:
			indexes.append(i)
			new_values.append(value)
	new_df = df.iloc[indexes, :].copy()
	new_df[column] = new_values
	return new_df


class ProcessGenericFeatures:

	def __init__(self, cfg):
		self.cfg = cfg

	def process_exac(self, save_to_file=False):
		'''
		Process ExAC features
		- Generic (Keep only GeneSize -- rest of features are replaced by Gnomad Constraint scores)
		- CNV
		:return: exac_df
		'''

		print("\n>> Compiling ExAC / CNV / mut_prob features...")

		## Base features (static)
		base_df = pd.read_csv(self.cfg.data_dir / 'exac-broadinstitute/all_genes_exac-broad_feature_table.txt', sep='\t')
		# keep only GeneSize -- rest of features are replaced by Gnomad Constraint scores)
		base_df = base_df[['Gene_Name', 'GeneSize']]


		print(base_df.shape)

		## CNV features
		cnv_df = pd.read_csv(self.cfg.data_dir / 'exac-broadinstitute/cnv/exac-final-cnv.gene.scores071316', sep=' ')
		cnv_df.drop(['gene', 'chr', 'start', 'end'], axis=1, inplace=True)

		# Collapse entries with same gene_symbol and get average from all features
		agg_cnv_df = cnv_df.groupby('gene_symbol').mean()
		# Make sure flags in the interval 0<=flag<0.5 are assigned a '0' value...
		# ...and flags between 0.5<=flag<=1 a '1' value, after groupping by mean.
		agg_cnv_df.loc[(agg_cnv_df.flag >= 0.5), 'flag'] = 1
		agg_cnv_df.loc[(agg_cnv_df.flag < 0.5), 'flag'] = 0

		agg_cnv_df.columns = 'ExAC_' + agg_cnv_df.columns
		agg_cnv_df.insert(0, 'Gene_Name', agg_cnv_df.index)
		#agg_cnv_df.to_csv(self.cfg.data_dir / 'exac-broadinstitute/cnv/ExAC_CNV_features.tsv', sep='\t', index=None)
		print(agg_cnv_df.shape)

		exac_df = pd.merge(base_df, agg_cnv_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
		print(exac_df.shape)
		print(exac_df.head())

		if save_to_file:
			exac_df.to_csv(self.cfg.data_dir / 'exac-broadinstitute/compiled_exac_features.tsv', sep='\t', index=None)

		return exac_df


	def process_gnomad(self, save_to_file=False):
		'''
		Process gnomad (2018 release) Constraint features
		:return: gnomad_df
		'''
		print("\n>> Compiling GnomAD constraint features...")
		gnomad_df = pd.read_csv(self.cfg.data_dir / 'gnomad/constraint.txt', sep='\t')

		print(gnomad_df.shape)
		gnomad_df = gnomad_df.loc[gnomad_df.canonical == True, :]
		gnomad_df.reset_index(drop=True, inplace=True)
		gnomad_df.fillna(0, inplace=True)


		# Doc: Some genes have more than one transcripts annotated as 'canonical'
		# In that case, keep the transcript that has the largest sum of: obs_lof + obs_mis + obs_syn
		dupl_genes = gnomad_df.gene[gnomad_df.gene.duplicated()]
		dupl_canonical_df = gnomad_df.loc[ gnomad_df.gene.isin(dupl_genes) ].copy()
		dupl_canonical_df = gnomad_df.loc[gnomad_df.gene.isin(dupl_genes)].copy()

		# keep data frame with unique gene names
		uniq_canonical_df = gnomad_df.loc[~gnomad_df.gene.isin(dupl_genes)].copy()

		dupl_canonical_df['obs_sum'] = dupl_canonical_df['obs_lof'] + dupl_canonical_df['obs_mis'] + dupl_canonical_df['obs_syn']
		dupl_canonical_df = dupl_canonical_df.sort_values(by=['gene', 'obs_sum'], ascending=False)
		dupl_canonical_df.drop_duplicates(subset='gene', keep='first', inplace=True)
		dupl_canonical_df.drop(['obs_sum'], axis=1, inplace=True)

		uniq_gnomad_df = pd.concat([uniq_canonical_df, dupl_canonical_df], axis=0, sort=False)
		uniq_gnomad_df.drop(['transcript', 'canonical', 'obs_syn', 'exp_syn', 'oe_syn', 'oe_syn_lower', 'oe_syn_upper', 'syn_z', 'gene_issues'], axis=1, inplace=True)
		uniq_gnomad_df.columns = ['GnomAD_' + c for c in uniq_gnomad_df.columns.values]
		uniq_gnomad_df.rename(columns={'GnomAD_gene': 'Gene_Name'}, inplace=True)

		if save_to_file:
			gnomad_df.to_csv(self.cfg.data_dir / 'gnomad/compiled_gnomad_features.tsv', sep='\t', index=None)


		return uniq_gnomad_df


	def process_genic_intolerance_scores(self, save_to_file=False):
		'''
		Process RVIS and MTR features
		:return: genic_intol_df
		'''

		print("\n>> Compiling genic-intolerance features...")
		rvis_df = pd.read_csv(self.cfg.data_dir / 'genic-intolerance/GenicIntolerance_v3_12Mar16.txt', sep='\t')
		rvis_df = rvis_df.iloc[:, [0, 4, 23]]
		rvis_df.columns = ['Gene_Name', 'RVIS', 'LoF_FDR_ExAC']
		print(rvis_df.shape)

		rvis_exac_df = pd.read_csv(self.cfg.data_dir / 'genic-intolerance/RVIS_Unpublished_ExAC_May2015.txt', sep='\t')
		rvis_exac_df = rvis_exac_df.iloc[:, [0, 6]]
		rvis_exac_df.columns = ['Gene_Name', 'RVIS_ExAC']
		print(rvis_exac_df.shape)

		rvis_exac_v2_df = pd.read_csv(self.cfg.data_dir / 'genic-intolerance/RVIS_Unpublished_ExACv2_March2017.txt', sep='\t')
		rvis_exac_v2_df = rvis_exac_v2_df.iloc[:, [0, 1, 3, 6]]
		rvis_exac_v2_df.columns = ['Gene_Name', 'geneCov_ExACv2', 'RVIS_ExACv2', 'MTR_ExACv2']
		print(rvis_exac_v2_df.shape)

		genic_intol_df = pd.merge(rvis_df, rvis_exac_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
		genic_intol_df = pd.merge(genic_intol_df, rvis_exac_v2_df, how='outer', left_on='Gene_Name', right_on='Gene_Name')
		print(genic_intol_df.shape)

		if save_to_file:
			genic_intol_df.to_csv(self.cfg.data_dir / 'genic-intolerance/compiled_genic_intolerance_scores.tsv', sep='\t', index=None)

		return genic_intol_df


	def process_essential_mouse_genes(self, save_to_file=False):
		print("\n>> Compiling essential mouse gene features...")

		essential_mouse_df = pd.read_csv(self.cfg.data_dir / 'essential_genes_for_mouse/essential_mouse_gene_features.tsv', sep='\t', index_col=False)
		essential_mouse_df.replace({'Y': 1, 'N': 0}, inplace=True)
		print(essential_mouse_df.shape)

		if save_to_file:
			essential_mouse_df.to_csv(self.cfg.data_dir / 'essential_genes_for_mouse/compiled_essential_mouse_features.tsv', sep='\t', index=None)

		return essential_mouse_df


	def process_gwas_features(self, wv, disease_name, exact_terms, fuzzy_terms, exclude_terms, search_term='All', verbose=False, signif_thres = '5e-08', save_to_file=False):
		print("\n>> Compiling GWAS features...")
		
		exact_terms = exact_terms + disease_name
		fuzzy_terms = fuzzy_terms + disease_name

		df = pd.read_csv(self.cfg.data_dir / 'gwas_catalog/all_associations_v1.0.2.tsv', sep='\t', low_memory=False)

		print('Genes to keep:', search_term)
		
		exclude_pattern = re.compile('|'.join(exclude_terms), re.IGNORECASE)
		include_pattern = re.compile('|'.join(exact_terms), re.IGNORECASE)
		
		if len(exclude_terms) > 0:
			df = df.loc[ ~df['DISEASE/TRAIT'].str.contains(exclude_pattern) ]
		
		###########
		
		# if self.cfg.exact_string_matching == False:
		
		disease_names = df['DISEASE/TRAIT'].unique()
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

		terms_parsed = [x.lower() for term in fuzzy_terms for x in re.split('\W+|_', term) if x != '']
		profile = np.mean(wv[terms_parsed], axis=0) 
		distance = pd.DataFrame(np.sum(np.abs(wv_term - profile), axis=1), columns = ['distance'], index = term_ids.description.values)

		distance = distance.assign(description = lambda x: x.index.values)
		best_matching_terms = distance.sort_values(['distance']).head(n=self.cfg.ntop_terms_protein_atlas)
		
		all_selected_diseases = list(best_matching_terms.description.values)
		
		df_fuzzy = df.loc[lambda x: x['DISEASE/TRAIT'].isin(all_selected_diseases)]
		df_exact = df.loc[df['DISEASE/TRAIT'].str.contains(include_pattern)]

		df = pd.concat([df_fuzzy, df_exact])
		
			
		print(df.shape)
		if df.shape[0] == 0:
			return pd.DataFrame()

		if verbose:
			print(df['DISEASE/TRAIT'].unique())

		# Keep only hits that achieved genome-wide significance
		signif_df = df.loc[df['P-VALUE'].astype(float) < float(signif_thres)]
		signif_df = signif_df[['REPORTED GENE(S)', 'P-VALUE', 'OR or BETA', 'DISEASE/TRAIT', 'MAPPED_TRAIT']]
		print(signif_df.shape)

		# Split comma-separated Genes into separate lines
		expanded_df = explode(signif_df, 'REPORTED GENE(S)', sep=',')
		print(expanded_df.shape)

		# Count hits per gene
		hits_per_gene_df = expanded_df.groupby('REPORTED GENE(S)').agg('count')
		print(hits_per_gene_df.shape)

		# Limit to HGNC genes
		hits_per_gene_df = hits_per_gene_df.reindex(self.cfg.hgnc_genes_series)
		hits_per_gene_df.fillna(0, inplace=True)
		hits_per_gene_df.sort_values(by='P-VALUE', inplace=True, ascending=False)
		hits_per_gene_df.reset_index(inplace=True)
		hits_per_gene_df.columns.values[[0, 1]] = ['Gene_Name', 'GWAS_hits']
		hits_per_gene_df = hits_per_gene_df.iloc[:, [0, 1]]
		print(hits_per_gene_df.shape)

		# Get min and max P-value for each gene
		full_df = hits_per_gene_df.merge(expanded_df, how='left', left_on='Gene_Name', right_on='REPORTED GENE(S)')
		full_df['P-VALUE'] = full_df['P-VALUE'].apply(lambda x: float(x))
		full_df = full_df.rename(columns={'P-VALUE': 'P_VALUE', 'OR or BETA': 'OR'})
		full_df.drop(['REPORTED GENE(S)'], axis=1, inplace=True)

		def custom_agg_func(df, groupby_col, agg_func):
			feature_by_gene = pd.DataFrame(df.groupby('Gene_Name')[groupby_col].agg(agg_func))
			new_colname = 'GWAS_'+agg_func+'_'+groupby_col
			feature_by_gene = feature_by_gene.rename(columns={groupby_col: new_colname}).reset_index()
			# imputation
			feature_by_gene.fillna(feature_by_gene[new_colname].median(), inplace=True)
			return feature_by_gene

		max_pval_df = custom_agg_func(full_df, 'P_VALUE', 'max')
		min_pval_df = custom_agg_func(full_df, 'P_VALUE', 'min')

		max_or_df = custom_agg_func(full_df, 'OR', 'max')
		min_or_df = custom_agg_func(full_df, 'OR', 'min')

		# compile final gwas data frame
		gwas_df = hits_per_gene_df.merge(max_pval_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
		gwas_df = gwas_df.merge(min_pval_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
		gwas_df = gwas_df.merge(max_or_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
		gwas_df = gwas_df.merge(min_or_df, how='left', left_on='Gene_Name', right_on='Gene_Name')

		gwas_df['GWAS_tissue_trait_flag'] = None
		gwas_df.loc[gwas_df.GWAS_hits > 0, 'GWAS_tissue_trait_flag'] = 1
		gwas_df.loc[gwas_df.GWAS_hits == 0, 'GWAS_tissue_trait_flag'] = 0
		total_gwas_genes = gwas_df.loc[gwas_df.GWAS_tissue_trait_flag == 1].shape[0]
		print("Total GWAS genes: {0}".format(total_gwas_genes))
		if search_term != 'All':
			gwas_df.columns = [s.replace('GWAS', search_term + '_GWAS') for s in gwas_df.columns.values]


		return gwas_df



	def process_mgi_essential_features(self, save_to_file=False):
		'''
		Get humang genes with mouse ortholog with a lethality phenotype (essential mouse genes)
		:return: 
		'''
		print(">> Compiling MGI essential genes features...")

		query_human_pheno_df = pd.read_csv(self.cfg.data_dir / 'mgi/hmd_human_pheno.processed.rpt', sep='\t')
		query_human_pheno_df.fillna('', inplace=True)
		# print(query_human_pheno_df.head())

		mgi_lethal_pheno_df = pd.read_csv(self.cfg.data_dir / 'mgi/MGI_LethalityPhenotypes.txt', sep='\t', header=None)
		mgi_lethal_pheno_df.columns = ['MP_ID', 'lethal_phenotype']

		mgi_lethal_phenotypes = mgi_lethal_pheno_df.MP_ID.unique()
		# print(mgi_lethal_phenotypes)

		def check_for_mp_ids(row):
			has_lethal_pheno = False
			mp_ids = row.split(' ')
			for p in mp_ids:
				if p in mgi_lethal_phenotypes:
					return True

			return has_lethal_pheno

		has_lethal_pheno = query_human_pheno_df['High-level Mammalian Phenotype ID (space-delimited)'].apply(check_for_mp_ids)

		mgi_essential_df = pd.DataFrame({'Gene_Name': query_human_pheno_df.loc[ has_lethal_pheno, 'Human Marker Symbol'], 'MGI_essential_gene': 1})
		# de-duplicate
		mgi_essential_df.drop_duplicates(subset='Gene_Name', inplace=True)
		print(mgi_essential_df.shape)

		if save_to_file:
			mgi_essential_df.to_csv(self.cfg.data_dir / 'mgi/mgi_essential_genes.tsv', sep='\t', index=None)

		return  mgi_essential_df


	def run_all(self, wv):
		exac_df = self.process_exac()
		print('ExAC:', exac_df.shape)
		
		gnomad_df = self.process_gnomad()
		print('GnomAD:', gnomad_df.shape)

		genic_intol_df = self.process_genic_intolerance_scores()
		print('Genic-intolerance:', genic_intol_df.shape)

		essential_mouse_df = self.process_essential_mouse_genes()
		print('Essential mouse genes:', essential_mouse_df.shape)
		print(essential_mouse_df.loc[ essential_mouse_df.essential_mouse_knockout == 1 ].shape)

		gwas_df = self.process_gwas_features(wv, self.cfg.disease_name, self.cfg.exact_terms, self.cfg.fuzzy_terms, self.cfg.exclude_terms)
		print('GWAS:', gwas_df.shape)

		# TODO: check if that's redundant with essential_mouse_df
		mgi_essential_df = self.process_mgi_essential_features()
		print('MGI:', mgi_essential_df.shape)
		
		if self.cfg.features_table is not None: 

			features_table = pd.read_csv(self.cfg.features_table, sep = '\t')
			if 'gene_id' in features_table.columns.values: 
				features_table = features_table.drop(['gene_id'], axis=1)

			if 'gene' in features_table.columns.values: 
				features_table = features_table.rename(columns = {'gene': 'Gene_Name'})
				


			if self.cfg.null_imp == 'zero': 
				features_table = features_table.fillna(0)
			elif self.cfg.null_imp == 'median': 
				features_table = features_table.fillna(features_table.median())
			else: 
				imp = pd.read_csv(self.cfg.null_imp, sep = '\t', header = None)
				imp.columns = ['feature', 'imp']
				for col in features_table.select_dtypes('number').columns.values: 
					if col in imp.feature.values: 
						imp_method = imp.loc[lambda x: x.feature == col].imp.values[0]
						if imp_method == 'median': 
							features_table[col] = features_table[col].fillna(features_table[col].median())
						else: 
							features_table[col] = features_table[col].fillna(0)
					else: 
						features_table[col] = features_table[col].fillna(features_table[col].median())

			features_table = features_table.groupby(['Gene_Name']).mean().reset_index()
		# a = pd.DataFrame([1, np.nan, 3], columns = ['col'])
		# a['col'] = a['col'].fillna(a['col'].median())

		print("\n>> Merging all data frames together...")
		generic_features_df = pd.merge(exac_df, gnomad_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
		print(generic_features_df.shape)
		generic_features_df = pd.merge(generic_features_df, genic_intol_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
		print(generic_features_df.shape)
		generic_features_df = pd.merge(generic_features_df, essential_mouse_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
		print(generic_features_df.shape)
		generic_features_df = pd.merge(generic_features_df, gwas_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
		print(generic_features_df.shape)
		generic_features_df = pd.merge(generic_features_df, mgi_essential_df, how='left', left_on='Gene_Name', right_on='Gene_Name')
		print(generic_features_df.shape) 
		if self.cfg.features_table is not None: 
			generic_features_df = pd.merge(generic_features_df, features_table, how = 'left', on = 'Gene_Name')
		print(generic_features_df.shape)
		

		# Impute 'MGI_essential_gene' with zero, for all genes that don't have a '1' value:
		# these values are not missing data but rather represent a 'False'/zero feature value.
		generic_features_df['MGI_essential_gene'].fillna(0,inplace=True)


		generic_features_df.to_csv(self.cfg.generic_feature_table, sep='\t', index=None)
		print("Saved to {0}".format(self.cfg.generic_feature_table))

		print(generic_features_df.shape)



if __name__ == '__main__':

	config_file = sys.argv[1] #'../../../config.yaml'
	out_dir = sys.argv[2]
	cfg = Config(config_file, out_dir)
	wv = Bioword2vec_embeddings(cfg)

	proc = ProcessGenericFeatures(cfg)
	proc.run_all(wv.embeddings)
