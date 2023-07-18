import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 
matplotlib.use('agg')
import pandas as pd
import os, sys
from mantis_ml.config_class import Config

from mantis_ml.modules.unsupervised_learn.dimens_reduction import DimensionalityReduction
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class DimensReductionWrapper(DimensionalityReduction):

	def __init__(self, cfg, data, highlighted_genes, recalc, tsne_perplex=30):
		DimensionalityReduction.__init__(self, cfg)

		self.data = data
		self.highlighted_genes = highlighted_genes
		self.recalc = recalc
		self.tsne_perplex = tsne_perplex

	def run_pca(self, method='PCA'):

		# try:
		# Calculate principal components
		pca, pca_df = self.calc_principal_components(self.data, method=method)
		print(pca_df.head())

		# Plot PCA
		plot_title = "Principal Component Analysis"
		self.plot_embedding_w_labels(pca_df, self.highlighted_genes, 'PC1', 'PC2',
								plot_title=plot_title, filename_prefix=method, figsize=(12, 12))

		# Store Interactive PCA to .html file
		interactive_pca_df = pca_df.copy()
		interactive_pca_df.rename(columns={'PC1': 'x', 'PC2': 'y'}, inplace=True)
		self.plot_interactive_viz(interactive_pca_df, self.highlighted_genes, method, 1, 0)

		# Make Scree Plot
		if method == 'PCA':
			self.make_scree_plot(pca)

		# except Exception as e:
		#	 print('[Exception]:', e)

		# TODO: 3D PCA scatterplot -- https://plot.ly/python/3d-scatter-plots



	def run_tsne(self, data_type='original_data'):

		method_2d = "t-SNE." + data_type + '.perplexity' + str(self.tsne_perplex) + '.ncomp_2'
		method_3d = "t-SNE." + data_type + '.perplexity' + str(self.tsne_perplex) + '.ncomp_3'
		plot_title = "t-SNE (perplexity=" + str(self.tsne_perplex) + ")"

		X_tsne = None
		total_time = -1.0
		stored_Xtsne_data_2d = str(self.cfg.unsuperv_out / ("tSNE.perplexity" + str(self.tsne_perplex) + "." + data_type + '.ncomp_2' +  ".tsv"))
		stored_Xtsne_data_3d = str(self.cfg.unsuperv_out / ("tSNE.perplexity" + str(self.tsne_perplex) + "." + data_type + '.ncomp_3' +  ".tsv"))
		print(stored_Xtsne_data_2d)
		print(stored_Xtsne_data_3d)

		if os.path.exists(stored_Xtsne_data_2d) and os.path.exists(stored_Xtsne_data_3d) and not self.recalc:
			X_tsne_2d = pd.read_csv(stored_Xtsne_data_2d, sep = '\t', index_col = False)
			X_tsne_3d = pd.read_csv(stored_Xtsne_data_3d, sep = '\t', index_col = False)
		else:
			print('Calculating t-SNE with perplexity:', self.tsne_perplex)
			X_tsne_2d, total_time_2d = self.calc_tsne(self.data, n_comp = 2, data_type=data_type, perplexity=self.tsne_perplex)
			X_tsne_3d, total_time_3d = self.calc_tsne(self.data, n_comp = 3, data_type=data_type, perplexity=self.tsne_perplex)
			

		# print(f"[t-SNE] Total time elapsed: {total_time_2d + total_time_3d}s")
		print(X_tsne_2d.head())
		print(X_tsne_3d.head())


		print('Plotting t-SNE embedding with selected gene labels...')
		self.plot_embedding_w_labels(X_tsne_2d, self.highlighted_genes, 'd0', 'd1',
								plot_title=plot_title, filename_prefix=method_2d, figsize=(14, 12))
		
		interactive_X_tsne = X_tsne_2d.copy()
		interactive_X_tsne.rename(columns={'d0': 'x', 'd1': 'y'}, inplace=True)
		self.plot_interactive_viz(interactive_X_tsne, self.highlighted_genes, method_2d, 1, 0)

		# TODO: complete automated nested-clustering
		print('Getting clusters (louvain) on t-SNE...')
		# agglom_cl, tsne_repr, gene_names = get_clustering_from_tsne(X_tsne_2d, n_clusters=20, perplexity=self.tsne_perplex)
		# clustering_2d, tsne_repr_2d, gene_names = self.get_clustering_from_tsne_kmeans(2, X_tsne_2d, n_clusters=self.cfg.n_clusters, perplexity=self.tsne_perplex)
		# clustering_3d, tsne_repr_3d, gene_names = self.get_clustering_from_tsne_kmeans(3, X_tsne_3d, n_clusters=self.cfg.n_clusters, perplexity=self.tsne_perplex)
		
		clustering, tsne_repr_2d, gene_names = self.get_louvain_clustering(self.data, X_tsne_2d, nneighb=10)
		
		tsne_repr_3d = X_tsne_3d.drop([self.cfg.Y, 'Gene_Name'], axis=1)
		tsne_repr_3d.columns = ['x', 'y', 'z']
		
		tsne_repr_3d['Gene_Name'] = X_tsne_3d['Gene_Name'].values
		tsne_repr_3d[self.cfg.Y] = X_tsne_3d[self.cfg.Y].values
		
		tsne_repr_3d = tsne_repr_3d.merge(tsne_repr_2d[['Gene_Name', 'cluster']], on = 'Gene_Name')
		
		tsne_repr_3d.index = tsne_repr_3d['Gene_Name'].values
		tsne_repr_3d = tsne_repr_3d.loc[gene_names]
		tsne_repr_3d = tsne_repr_3d.reset_index(drop=True)
		
		filename_prefix_2d = 't-SNE.perplexity' + str(self.tsne_perplex) + '.with_cluster_annotation' + '.ncomp_2'
		filename_prefix_3d = 't-SNE.perplexity' + str(self.tsne_perplex) + '.with_cluster_annotation' + '.ncomp_3'

		self.plot_embedding_w_clusters(clustering, tsne_repr_2d, gene_list=self.cfg.highlighted_genes,
								gene_names=gene_names,
								filename_prefix=filename_prefix_2d)

		filename_prefix_2d = 'cluster_signatures.perplexity' + str(self.tsne_perplex) + '.with_cluster_annotation' + '.ncomp_2'
		
		self.plot_clusters_signature(self.data, clustering, filename_prefix=filename_prefix_2d, ncomp=2)



	def run(self):
		
		# > UMAP

		print("\n>> Running Unsupervised analysis...")
		# > PCA
		self.run_pca()

		# > Sparse PCA
		# self.run_pca(method='SparsePCA')

		# > t-SNE
		self.run_tsne()

		print("...Unsupervised analysis complete.")

		# TODO: clustering of t-SNE plot and Pathway Enrichment analysis of CoInterest with PANTHER/IPA


if __name__ == '__main__':

	config_file = sys.argv[1] #'../../config.yaml'
	cfg = Config(config_file)


	highlighted_genes = cfg.highlighted_genes
	if len(sys.argv) > 2:
		gene_list_file = sys.argv[2]
		highlighted_genes = pd.read_csv(gene_list_file, header=None)
		highlighted_genes = highlighted_genes.iloc[ :, 0].tolist()

		highlighted_genes = highlighted_genes + cfg.highlighted_genes
	print(highlighted_genes)


	recalc = False # Default: True

	data = pd.read_csv(cfg.processed_data_dir / "processed_feature_table.tsv", sep='\t')
	dim_reduct_wrapper = DimensReductionWrapper(cfg, data, highlighted_genes, recalc=recalc)
	dim_reduct_wrapper.run()
