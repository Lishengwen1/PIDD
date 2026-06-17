import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors


def _reduce_2d_fit_transform(X_real, X_all, method='pca', random_state=42):
    """
    Fit t-SNE on X_real and then project all points (X_all) into the same 2D space.
    For points not in X_real, use weighted average of nearest neighbors in X_real.
    """
    ts = TSNE(
        n_components=2, 
        random_state=random_state, 
        init='pca',
        perplexity=30,           
        early_exaggeration=12.0, 
        learning_rate='auto'     
    )
    # 1. Fit t-SNE on X_real
    emb_real = ts.fit_transform(X_real)
    
    # 2. For other points in X_all, find nearest neighbors in X_real and use weighted average
    if X_all.shape[0] > X_real.shape[0]:
        X_other = X_all[X_real.shape[0]:]
        nbrs = NearestNeighbors(n_neighbors=5).fit(X_real)
        distances, indices = nbrs.kneighbors(X_other)
        weights = 1.0 / (distances + 1e-8)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        emb_other = np.zeros((X_other.shape[0], 2))
        for i in range(X_other.shape[0]):
            for j in range(weights.shape[1]):
                emb_other[i] += weights[i, j] * emb_real[indices[i, j]]
        emb_all = np.vstack([emb_real, emb_other])
    else:
        emb_all = emb_real
            
    return emb_real, emb_all

def _get_dynamic_axis_limits(embeddings, padding_ratio=0.1):
    """
    Dynamically compute axis limits based on the actual distribution of embeddings.
    """
    if len(embeddings) == 0:
        return (-1, 1), (-1, 1)
    
    x_min, x_max = np.min(embeddings[:, 0]), np.max(embeddings[:, 0])
    y_min, y_max = np.min(embeddings[:, 1]), np.max(embeddings[:, 1])
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_padding = x_range * padding_ratio
    y_padding = y_range * padding_ratio
    
    x_limits = (x_min - x_padding, x_max + x_padding)
    y_limits = (y_min - y_padding, y_max + y_padding)
    
    return x_limits, y_limits

def _remove_axes(ax):
    """Remove axes ticks and spines."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def _apply_energy_threshold(energy_grid, threshold_ratio=0.8):
    """
    Apply energy threshold to hide high‑energy regions.
    Only the lowest `threshold_ratio` fraction of energy values are kept.
    """
    if energy_grid.size == 0:
        return energy_grid
    threshold = np.quantile(energy_grid, threshold_ratio)
    masked_energy = energy_grid.copy()
    masked_energy[masked_energy > threshold] = np.nan
    
    return masked_energy

def visualize_energy_to_pdf(features,lables,
                            num_real,
                            reduce_method='tsne',    
                            kde_bw_method=None,      
                            grid_size=200,
                            temp=1.0,
                            figsize=(8,6),
                            cmap='viridis',
                            random_state=42,
                            energy_threshold_ratio=0.8,  
                            save_path='energy.pdf'):
    """
    Inputs:
      - features: numpy or torch array of shape (N,D)
      - lables: array of integer labels indicating group membership
      - num_real: int, first `num_real` samples are considered 'real'
      - save_path: directory path to save output figures (PDF and PNG)

    Outputs:
      - Saves multi‑page PDFs and PNGs for each visualization.
    """
    X_all = np.asarray(features)
    methods_labels_all = np.asarray(lables)
    # X_all = _to_numpy(X_all)
    N, D = X_all.shape
    num_real = int(num_real)

    # ------------------------------------------------------------
    # Label mapping (only a subset of labels are used in this paper):
    #   0  -> Real 
    #   2  -> PIDD
    #   11 -> IID
    #   12 -> IDM
    #   14 -> HDD
    #   15..35 -> Process methods (21 groups, indices 0..20 in code)
    # Other labels (e.g. 1,3,...) are ignored in this visualization.
    # ------------------------------------------------------------

    X_real = X_all[methods_labels_all==0]
    X_pidd = X_all[methods_labels_all==2]
    X_iid = X_all[methods_labels_all==11]
    X_idm = X_all[methods_labels_all==12]
    X_hdd = X_all[methods_labels_all==14]
    X_process = {}
    for k in range(15, 36):
        X_process[k-15] = X_all[methods_labels_all==k]

    # 1) Fit 2D reducer on X_real and transform all points
    emb_real, emb_all = _reduce_2d_fit_transform(X_real, X_all, method=reduce_method, random_state=random_state)
    emb_pidd = emb_all[methods_labels_all==2]
    emb_iid = emb_all[methods_labels_all==11]
    emb_idm = emb_all[methods_labels_all==12]
    emb_hdd = emb_all[methods_labels_all==14]
    emb_process = {}
    for k in range(15, 36):
        emb_process[k-15] = emb_all[methods_labels_all==k]
    
    # Dynamically compute axis limits from all embeddings
    all_embeddings = np.vstack([emb for emb in [emb_real, emb_pidd, emb_iid, emb_idm, emb_hdd, emb_process[0], emb_process[1], emb_process[2], emb_process[3], emb_process[4], emb_process[5], emb_process[6], emb_process[7], emb_process[8], emb_process[9], emb_process[10], emb_process[11], emb_process[12], emb_process[13], emb_process[14], emb_process[15], emb_process[16], emb_process[17], emb_process[18], emb_process[19], emb_process[20]] if len(emb) > 0])
    x_lim, y_lim = _get_dynamic_axis_limits(all_embeddings, padding_ratio=0.1)



    # 2) Fit KDE on emb_real and compute energy for all points
    xy_all = emb_all.T  
    eps = 1e-12
    kde = gaussian_kde(emb_real.T, bw_method=kde_bw_method)

    dens_all = kde(xy_all) + eps
    energy_all = -temp * np.log(dens_all)
    energy_real = energy_all[methods_labels_all==0]
    energy_pidd = energy_all[methods_labels_all==2]
    energy_iid = energy_all[methods_labels_all==11]
    energy_idm = energy_all[methods_labels_all==12]
    energy_hdd = energy_all[methods_labels_all==14]
    energy_process = {}
    for k in range(15, 36):
        energy_process[k-15] = energy_all[methods_labels_all==k]

    # 3) Build energy grid (over range determined by emb_real)
    x_min, x_max = x_lim
    y_min, y_max = y_lim
    x_pad = (x_max - x_min) * 0.05 + 1e-6
    y_pad = (y_max - y_min) * 0.05 + 1e-6
    xi = np.linspace(x_min - x_pad, x_max + x_pad, grid_size)
    yi = np.linspace(y_min - y_pad, y_max + y_pad, grid_size)
    xx, yy = np.meshgrid(xi, yi)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    dens_grid = kde(positions).reshape(grid_size, grid_size) + eps
    energy_grid = -temp * np.log(dens_grid)
    
    # Apply energy threshold to mask high‑energy regions
    energy_grid_masked = _apply_energy_threshold(energy_grid, energy_threshold_ratio)
    

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)


    colors = {
        'real': '#501d8a',      
        'dsam': '#e55709',      
        'pidd': '#e55709',      
        'random': '#e55709',    
        'energy_cmap': 'RdYlGn_r'  
    }

    # ----- Figure 1: Real points only (scatter, not energy‑colored) -----
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(emb_real[:,0], emb_real[:,1], c=colors['real'], s=20, alpha=1.0,edgecolors='black', linewidths=0.3)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    _remove_axes(ax)
    plt.tight_layout()
    
    pp = PdfPages(os.path.join(save_path, "real.pdf"))
    pp.savefig(fig, bbox_inches='tight', dpi=300)
    pp.close()
    plt.savefig(os.path.join(save_path, "real.png"), bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)

    # ----- Figures for each process of PIDD (energy field +  points) -----
    for k in range (21):
        fig, ax = plt.subplots(figsize=figsize)
        ax.contourf(xx, yy, energy_grid_masked, levels=10, cmap=colors['energy_cmap'], alpha=1.0)
        if len(emb_process[k]) > 0:
            sc = ax.scatter(emb_process[k][:,0], emb_process[k][:,1], c=energy_process[k], s=50, cmap=colors['energy_cmap'],  edgecolors='black', linewidths=0.3)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        _remove_axes(ax)
        plt.tight_layout()
        
        pp = PdfPages(os.path.join(save_path, "real_field_process_{}.pdf".format(k)))
        pp.savefig(fig, bbox_inches='tight', dpi=300)
        pp.close()
        plt.savefig(os.path.join(save_path, "real_field_process_{}.png".format(k)), bbox_inches='tight', dpi=300, transparent=True)
        plt.close(fig)



    # ----- Figure: Energy field only (without points) -----
    fig, ax = plt.subplots(figsize=figsize)
    contour = ax.contourf(xx, yy, energy_grid_masked, levels=10, cmap=colors['energy_cmap'], alpha=1.0)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    _remove_axes(ax)
    plt.tight_layout()
    
    pp = PdfPages(os.path.join(save_path, "real_field.pdf"))
    pp.savefig(fig, bbox_inches='tight', dpi=300)
    pp.close()
    plt.savefig(os.path.join(save_path, "real_field.png"), bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)

    # ----- Figure: Energy field + PIDD  -----
    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, energy_grid_masked, levels=10, cmap=colors['energy_cmap'], alpha=1.0)
    if len(emb_pidd) > 0:
        sc = ax.scatter(emb_pidd[:,0], emb_pidd[:,1], c=energy_pidd, s=50, cmap=colors['energy_cmap'], edgecolors='black', linewidths=0.3)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    _remove_axes(ax)
    plt.tight_layout()
    
    pp = PdfPages(os.path.join(save_path, "real_field_PIDD.pdf"))
    pp.savefig(fig, bbox_inches='tight', dpi=300)
    pp.close()
    plt.savefig(os.path.join(save_path, "real_field_PIDD.png"), bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)




    # ----- Figure: Energy field + IID -----
    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, energy_grid_masked, levels=10, cmap=colors['energy_cmap'], alpha=1.0)
    if len(emb_iid) > 0:
        sc = ax.scatter(emb_iid[:,0], emb_iid[:,1], c=energy_iid, s=50, cmap=colors['energy_cmap'], edgecolors='black', linewidths=0.3)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    _remove_axes(ax)
    plt.tight_layout()
    
    pp = PdfPages(os.path.join(save_path, "real_field_IID.pdf"))
    pp.savefig(fig, bbox_inches='tight', dpi=300)
    pp.close()
    plt.savefig(os.path.join(save_path, "real_field_IID.png"), bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)

    # ----- Figure: Real + IID, no energy field -----
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(emb_real[:,0], emb_real[:,1], c=colors['real'], s=20, alpha=1.0, edgecolors='none')
    if len(emb_iid) > 0:
        ax.scatter(emb_iid[:,0], emb_iid[:,1], c=colors['dsam'], s=50, alpha=1.0, edgecolors='black', linewidths=0.3)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    _remove_axes(ax)
    plt.tight_layout()
    
    pp = PdfPages(os.path.join(save_path, "real_IID.pdf"))
    pp.savefig(fig, bbox_inches='tight', dpi=300)
    pp.close()
    plt.savefig(os.path.join(save_path, "real_IID.png"), bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)

    # ----- Figure: Energy field + IDM -----
    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, energy_grid_masked, levels=10, cmap=colors['energy_cmap'], alpha=1.0)
    if len(emb_idm) > 0:
        sc = ax.scatter(emb_idm[:,0], emb_idm[:,1], c=energy_idm, s=50, cmap=colors['energy_cmap'], edgecolors='black', linewidths=0.3)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    _remove_axes(ax)
    plt.tight_layout()
    
    pp = PdfPages(os.path.join(save_path, "real_IDM.pdf"))
    pp.savefig(fig, bbox_inches='tight', dpi=300)
    pp.close()
    plt.savefig(os.path.join(save_path, "real_IDM.png"), bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)


    # ----- Figure: Energy field + HDD -----
    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, energy_grid_masked, levels=10, cmap=colors['energy_cmap'], alpha=1.0)
    if len(emb_hdd) > 0:
        sc = ax.scatter(emb_hdd[:,0], emb_hdd[:,1], c=energy_hdd, s=50, cmap=colors['energy_cmap'], edgecolors='black', linewidths=0.3)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    _remove_axes(ax)
    plt.tight_layout()
    
    pp = PdfPages(os.path.join(save_path, "real_HDD.pdf"))
    pp.savefig(fig, bbox_inches='tight', dpi=300)
    pp.close()
    plt.savefig(os.path.join(save_path, "real_HDD.png"), bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)

# ----- Colorbar for energy scale -----
    energy_min = np.min(energy_all)
    energy_max = np.max(energy_all)
    plot_energy_colorbar(energy_min, energy_max, 
                        cmap=colors['energy_cmap'], 
                        save_path=save_path)

    print(f"Saved visualization to: {save_path}")
    print(f"Each visualization saved as both PDF and PNG formats")
    return 0
def plot_energy_colorbar(energy_min, energy_max, cmap='RdYlBu_r', 
                        figsize=(6, 1), save_path='.'):
    """
    Plot a horizontal colorbar for energy levels.
    """

    fig, ax = plt.subplots(figsize=figsize)
    norm = plt.Normalize(energy_min, energy_max)
    cmap_obj = plt.get_cmap(cmap)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
                     cax=ax, orientation='horizontal')
    cb.set_label('Energy Level', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    
    for spine in cb.ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    pp = PdfPages(os.path.join(save_path, "energy_colorbar.pdf"))
    pp.savefig(fig, bbox_inches='tight', dpi=300)
    pp.close()
    plt.savefig(os.path.join(save_path, "energy_colorbar.png"), 
                bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    
if __name__ == "__main__":
    import torch
    features = torch.load('./features.pt')
    methods_lables = torch.load('./methods_labels.pt')
    out = visualize_energy_to_pdf(features, methods_lables, 
                                  num_real=10000,
                                  reduce_method='tsne', 
                                  kde_bw_method='scott',
                                  grid_size=200,
                                  temp=1.0,
                                  figsize=(8, 6),
                                  energy_threshold_ratio=0.6,  
                                  save_path='./results/')