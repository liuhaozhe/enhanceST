import pandas as pd
from sklearn import metrics
import scanpy as sc
from pathlib import Path
import numpy as np

# dir_input = Path(f'{data_path}/{data_name}/')
# pd1 = pd.DataFrame(adata.obs['clusters'].tolist(), columns=['layer_guess'], index=adata.obs['clusters'].index)
# pd1.to_csv(f'{dir_input}/metadata.tsv', sep='\t', index=True)

def calculate_cluster_score(sample_name, base_path, output_path, results):
    dir_input = Path(f'{base_path}/{sample_name}/')
    dir_output = Path(f'{output_path}/')

    # Read the metadata file
    # df_meta = pd.read_csv(f'{dir_input}/metadata.tsv', sep='\t')
    df_meta = pd.read_csv(f'{dir_input}/metadata.tsv', sep='\t', index_col=0)
    df_meta = df_meta.sort_values(by='layer_guess')
    print(df_meta)

    # Read the refined predictions from SpaGCN results
    adata = sc.read(f"{dir_output}/{results}.h5ad")
    pd2 = pd.DataFrame(adata.obs["DeepST_refine_domain"].tolist(), columns=["DeepST_refine_domain"],
                       index=adata.obs["DeepST_refine_domain"].index)

    # Filter out rows with missing layer_guess
    df_meta = df_meta[~pd.isnull(df_meta['layer_guess'])]
    filtered_pd2 = pd2[pd2.index.isin(df_meta['layer_guess'].index)]
    df_sorted = filtered_pd2.sort_values(by='DeepST_refine_domain')

    # Calculate ARI score
    ari_score = metrics.adjusted_rand_score(df_meta['layer_guess'].values, df_sorted['DeepST_refine_domain'].values)
    nmi_score = metrics.normalized_mutual_info_score(df_meta['layer_guess'].values, df_sorted['DeepST_refine_domain'].values)
    print(f'===== Project: {sample_name} ARI score: {ari_score:.3f} =====')
    print(f'===== Project: {sample_name} NMI score: {nmi_score:.3f} =====')
    return ari_score, nmi_score