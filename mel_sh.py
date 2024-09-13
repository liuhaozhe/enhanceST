from enhanceST import *
from scipy.sparse import csr_matrix
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

def plot_loss(losses_list, val_losses_list, save=None):
    # epoch-loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses_list, label='Training Loss')
    plt.plot(val_losses_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epoch - Loss Graph')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./figures/show{save}', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_genes(hidpi_adata, show_genes, n=4, size=(20, 5), point_size=30, titles=None, cmap=None, save=None):
    if titles == None:
        titles = show_genes
    genes_index = [list(hidpi_adata.var_names).index(gene) for gene in show_genes]
    m = len(show_genes) // n + 1
    plt.figure(figsize=size)
    flag = 0
    for j in range(m):
        show_genes0 = show_genes[j * n:min(len(show_genes), (j + 1) * n)]

        for i in range(n):
            if not i < len(show_genes0):
                break
            plt.subplot(m, n, flag + 1)

            plt.scatter(hidpi_adata.obs['array_col'], hidpi_adata.obs['array_row'],
                        c=np.array(hidpi_adata.X.todense())[:, genes_index[flag]],
                        marker='s', s=point_size, vmin=0, cmap=cmap)
            plt.title(titles[flag])
            plt.xticks([])
            plt.yticks([])
            plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()
            plt.axis('off')
            flag += 1

    plt.savefig(f'./figures/show{save}', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def run_sh (hidpi_adata,data_path,data_name,save_path,layer,batch_size,lr,epoch):
    hidpi_adata

    hidpi_adata.X = csr_matrix(hidpi_adata.X)
    coords = hidpi_adata.obsm['spatial']
    hidpi_adata.obs['array_row'] = coords[:, 0]  # 提取行坐标
    hidpi_adata.obs['array_col'] = coords[:, 1]  # 提取列坐标
    hidpi_adata.var_names_make_unique()
    integral_coords = hidpi_adata.obs[['array_row', 'array_col']]

    integral_coords.loc["5x10", :] = [5, 10]  # We add three spots since they are in tissue but missing.
    integral_coords.loc["10x14", :] = [10, 14]
    integral_coords.loc["15x22", :] = [15, 22]

    # QC
    sc.pp.calculate_qc_metrics(hidpi_adata, inplace=True)
    sc.pp.filter_cells(hidpi_adata, min_genes=200)
    sc.pp.filter_genes(hidpi_adata, min_cells=10)

    # Genes expressed over ten percent of total spots
    good_adata = hidpi_adata[:, hidpi_adata.var["n_cells_by_counts"] > len(hidpi_adata.obs.index) * 0.1]
    n_samples = good_adata.shape[0]
    np.random.seed(42)
    indices = np.random.permutation(n_samples)

    # 按 8:2 比例划分索引
    split_point = int(n_samples * 0.8)
    train_idx, val_idx = indices[:split_point], indices[split_point:]

    # 创建两个新的 AnnData 对象
    good_adata_train = good_adata[train_idx].copy()
    good_adata_val = good_adata[val_idx].copy()

    # Get count matrix and coordinate matrix, they are inputs of DIST algorithm.
    train_counts = np.array(good_adata_train.X.todense())
    train_coords = good_adata_train.obs[['array_row', 'array_col']]
    val_counts = np.array(good_adata_val.X.todense())
    val_coords = good_adata_val.obs[['array_row', 'array_col']]

    # this test is full size
    test_adata = hidpi_adata
    test_counts = np.array(test_adata.X.todense())
    test_coords = test_adata.obs[['array_row', 'array_col']]
    train_set = getSTtrainset(train_counts, train_coords)
    val_set = getSTtrainset(val_counts, val_coords)
    test_set = getSTtestset(test_counts, test_coords)
    position_info, imputed_to_integral_mapping = get_ST_position_info(integral_coords)

    class Config:
        # network meta params
        epoch = 200
        batch_size = 128
        width = 64
        iteration_depth = layer  # layers number of iteration, range from conv layers
        iteration = 2  # times of iteration
        learning_rate = lr
        init_variance = 0.1  # variance of weight initializations, typically smaller when residual learning is on
        test_positive = True  # True means positive data, outputs are clipped by zeros.

        def __init__(self):
            # network meta params that by default are determined (by other params) by other params but can be changed
            self.filter_shape = ([[3, 3, 1, self.width]] +
                                 [[3, 3, self.width, self.width]] * (self.iteration_depth * self.iteration - 1) +
                                 [[3, 3, self.width, 4]])
            self.depth = self.iteration_depth * self.iteration + 1

    config = Config()
    imputed_img, losses_list, val_losses_list = enhanceST(train_set, val_set, test_set, epoch=epoch, batch_size=batch_size,
                                                          conf=config, gpu=0)

    imputed_counts, imputed_coords = img2expr(imputed_img, test_adata.var_names, integral_coords, position_info)

    imputed_adata = ad.AnnData(X=imputed_counts, obs=imputed_coords)
    imputed_adata.X = csr_matrix(imputed_adata.X)
    imputed_adata.var = hidpi_adata.var

    from scipy.interpolate import griddata

    # 从 adata 中提取原始点的行列坐标和空间坐标
    original_coords = hidpi_adata.obs[['array_row', 'array_col']].values
    original_spatial = hidpi_adata.obsm['spatial']
    # 从 imputed_adata 中提取新数据的行列坐标
    new_coords = imputed_adata.obs[['array_row', 'array_col']].values
    # 使用 griddata 插值生成新的点的空间坐标
    new_spatial = griddata(original_coords, original_spatial, new_coords, method='linear')

    # 将生成的空间坐标添加到 imputed_adata 中
    imputed_adata.obsm['spatial'] = new_spatial
    imputed_adata.uns = hidpi_adata.uns


    imputed_adata.write_h5ad(f"{data_path}/{data_name}2.h5ad")

    return imputed_adata,losses_list,val_losses_list

