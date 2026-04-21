import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from ipywidgets import Output, HBox
from IPython.display import display, clear_output
from common.data import DataWithLabels

import plotly.graph_objects as go
import numpy as np

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()




def plot_2d_arrows_stratified(embeddings, labels, samples_per_cluster=50, seed=42):
    """
    Plot 2D arrows with stratified sampling. Legend toggles arrow visibility.
    """
    np.random.seed(seed)
    
    unique_labels = np.unique(labels)
    sampled_idx = []
    
    for lab in unique_labels:
        cluster_idx = np.where(labels == lab)[0]
        n_sample = min(samples_per_cluster, len(cluster_idx))
        sampled_idx.extend(np.random.choice(cluster_idx, n_sample, replace=False))
    
    sampled_idx = np.array(sampled_idx)
    embeddings_sampled = embeddings[sampled_idx]
    labels_sampled = labels[sampled_idx]
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, lab in enumerate(unique_labels):
        mask = labels_sampled == lab
        cluster_emb = embeddings_sampled[mask]
        
        if lab == -1:
            color = '#aaaaaa'
            name = 'Noise (-1)'
        else:
            color = colors[i % len(colors)]
            name = f'Cluster {lab}'
        
        # Create line segments: origin -> point, with None separators
        x_lines, y_lines = [], []
        for emb in cluster_emb:
            x_lines.extend([0, emb[0], None])
            y_lines.extend([0, emb[1], None])
        
        # Single trace per cluster - arrows as lines
        fig.add_trace(go.Scatter(
            x=x_lines,
            y=y_lines,
            mode='lines',
            line=dict(color=color, width=1.5),
            opacity=0.5,
            name=name,
            legendgroup=f'cluster_{lab}',
            hoverinfo='skip',
            
        ))
        
        # Arrow tips (small markers at endpoints for visual arrowhead effect)
        fig.add_trace(go.Scatter(
            x=cluster_emb[:, 0],
            y=cluster_emb[:, 1],
            mode='markers',
            marker=dict(size=4, color=color, symbol='arrow', angle=0),
            legendgroup=f'cluster_{lab}',
            showlegend=False,
            hovertemplate=f'{name}<br>(%{{x:.3f}}, %{{y:.3f}})<extra></extra>'
        ))
    
    # Origin marker
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=10, color='black', symbol='x'),
        name='Origin',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'2D Embedding Arrows ({len(sampled_idx)} samples)',
        xaxis_title='Dim 1',
        yaxis_title='Dim 2',
        width=800,
        height=700,
        xaxis=dict(scaleanchor='y', scaleratio=1),
        legend=dict(x=1.02, y=0.98, itemclick='toggle', itemdoubleclick='toggleothers')
    )
    
    return fig


def plot_3d_arrows_stratified(embeddings, labels, samples_per_cluster=50, seed=42):
    """
    Plot 3D arrows with stratified sampling to ensure all clusters represented.
    """
    np.random.seed(seed)
    
    unique_labels = np.unique(labels)
    sampled_idx = []
    
    for lab in unique_labels:
        cluster_idx = np.where(labels == lab)[0]
        n_sample = min(samples_per_cluster, len(cluster_idx))
        sampled_idx.extend(np.random.choice(cluster_idx, n_sample, replace=False))
    
    sampled_idx = np.array(sampled_idx)
    embeddings_sampled = embeddings[sampled_idx]
    labels_sampled = labels[sampled_idx]
    
    fig = go.Figure()
    
    # Add one trace per cluster for proper legend
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, lab in enumerate(unique_labels):
        mask = labels_sampled == lab
        cluster_emb = embeddings_sampled[mask]
        color = colors[i % len(colors)]
        
        # Lines from origin
        for emb in cluster_emb:
            fig.add_trace(go.Scatter3d(
                x=[0, emb[0]], y=[0, emb[1]], z=[0, emb[2]],
                mode='lines',
                line=dict(color=color, width=2),
                legendgroup=f'cluster_{lab}',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Arrow tips as markers (one per cluster for legend)
        fig.add_trace(go.Scatter3d(
            x=cluster_emb[:, 0],
            y=cluster_emb[:, 1],
            z=cluster_emb[:, 2],
            mode='markers',
            marker=dict(size=4, color=color),
            name=f'Cluster {lab}',
            legendgroup=f'cluster_{lab}',
            hovertemplate='Cluster: %{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>',
            text=[str(lab)] * len(cluster_emb)
        ))
    
    fig.update_layout(
        title=f'3D Embedding Arrows ({len(sampled_idx)} samples)',
        scene=dict(
            xaxis_title='Dim 1',
            yaxis_title='Dim 2',
            zaxis_title='Dim 3',
            aspectmode='cube'
        ),
        width=900,
        height=700,
        legend=dict(x=1.02, y=0.98),
        # legend_font=dict(size=1.2)
    )
    
    return fig


def plot_conf(Configurations, scale_eps = 1, vert: bool = False, figname: str = None, fig_kw: dict = {} ):
    P = Configurations[..., :2]
    A = Configurations[..., 2:]
    norm_P = np.linalg.norm(P, axis=-1)
    norm_A = np.linalg.norm(A, axis=-1)
    max_P = np.max(norm_P)
    max_A = np.max(norm_A)

    scalep = 1. / np.max(norm_P) if max_P > scale_eps else 1.0
    scalea = 1. / np.max(norm_A) if max_A > scale_eps else 1.0

            # Prepare grid
    M, N, _ = P.shape
    x, y = np.meshgrid(np.arange(N), np.arange(M))

    # Recalculate pseudocubic coordinates to cartesian ones
    U, V = P[..., 0], P[..., 1]
    Eta, Xi = A[..., 0], A[..., 1]

    # Compute angles for each field (radians, range -pi to pi)
    angleP = np.arctan2(U, V)
    angleA = np.arctan2(Eta, Xi)

    # Common normalization and colormap for both fields
    norm = plt.Normalize(-np.pi, np.pi)
    cmap = 'hsv'

    # Create subplots
    if vert:
        fig, ax = plt.subplots(2, 1, sharex=True, **fig_kw)
    else:
        fig, ax = plt.subplots(1, 2, sharey=True, **fig_kw)

    q1 = ax[0].quiver(
        x, y, scalep * U, scalep * V, angleP,
        scale=1, scale_units='xy', angles='xy',
        cmap=cmap, norm=norm
    )
    ax[0].set_title(r"Polarization Field P(u, v)")
    ax[0].set_xlim((-1, N+1))
    ax[0].set_ylim((-1, M+1))
    ax[0].set_aspect('equal', adjustable='box')

    q2 = ax[1].quiver(
        x + 0.5, y + 0.5, scalea * Eta, scalea * Xi, angleA,
        scale=1, scale_units='xy', angles='xy',
        cmap=cmap, norm=norm
    )
    ax[1].set_title(r"Hidden Field A($\eta$, $\xi$)")
    ax[1].set_xlim((-1, N+1))
    ax[1].set_ylim((-1, M+1))
    ax[1].set_aspect('equal', adjustable='box')

    if figname:
        plt.savefig(figname, format="png", bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def compare_clusterings(values, labels1, labels2):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    sns.scatterplot(x=values[:, 1], y=values[:, 0], hue=labels1, palette='tab20', alpha=0.7, ax=axs[0])
    axs[0].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    axs[0].set_xlabel('kappa')
    axs[0].set_ylabel('alpha')
    axs[0].set_title('Original Labels', size=16)   
    fig.subplots_adjust(right=0.8)

    sns.scatterplot(x=values[:, 1], y=values[:, 0], hue=labels2, palette='tab10', alpha=0.7, ax=axs[1])
    axs[1].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    axs[1].set_xlabel('kappa')
    # axs[1].set_ylabel('alpha')
    axs[1].set_title('Embedding Space Labels', size=16)
    fig.subplots_adjust(right=0.8)

    plt.show()

class InteractivePhaseDiagram:
    """Interactive phase diagram - click any point to view its configuration."""
    
    def __init__(self, data: DataWithLabels, samples_per_cluster=10000, seed=42):
        self.data = data
        np.random.seed(seed)
        
        # Stratified sampling for performance
        unique_labels = np.unique(data.labels)
        sampled_idx = []
        
        for lab in unique_labels:
            cluster_idx = np.where(data.labels == lab)[0]
            n_sample = min(samples_per_cluster, len(cluster_idx))
            sampled_idx.extend(np.random.choice(cluster_idx, n_sample, replace=False))
        
        self.sampled_idx = np.array(sampled_idx)
        self.values_sampled = data.values[self.sampled_idx]
        self.labels_sampled = data.labels[self.sampled_idx]
        self.unique_labels = unique_labels
        
        # Output widget for configuration plots - match phase diagram size
        self.config_output = Output(layout={'width': '700px', 'min_height': '550px'})
        
    def _create_click_handler(self):
        def on_click(trace, points, state):
            if points.point_inds:
                point_idx = points.point_inds[0]
                original_idx = trace.customdata[point_idx]
                with self.config_output:
                    clear_output(wait=True)
                    print(f"Configuration index: {original_idx}, Label: {self.data.labels[original_idx]}")
                    print(f"Values: {self.data.values[original_idx]},   Energy: {self.data.energies[original_idx]}")
                    self.data.plot(int(original_idx), fig_kw={'figsize': (15, 8)})
        return on_click
    
    def show(self):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        # Use FigureWidget for click interactivity
        fig = go.FigureWidget()
        
        for i, lab in enumerate(self.unique_labels):
            mask = self.labels_sampled == lab
            cluster_values = self.values_sampled[mask]
            cluster_original_idx = self.sampled_idx[mask]
            
            if lab == -1:
                color = '#aaaaaa'
                name = 'Noise (-1)'
            else:
                color = colors[int(lab) % len(colors)]
                name = f'Cluster {lab}'
            
            fig.add_trace(go.Scatter(
                x=cluster_values[:, 1],
                y=cluster_values[:, 0],
                mode='markers',
                marker=dict(size=8, color=color, opacity=0.7),
                name=name,
                customdata=cluster_original_idx,
                hovertemplate=f'{name}<br>x=%{{x:.3f}}, y=%{{y:.3f}}<br>idx=%{{customdata}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Phase diagram',
            xaxis_title='Parameter 1',
            yaxis_title='Parameter 2',
            width=700,
            height=550,
            legend=dict(x=1.02, y=0.98),
            hovermode='closest'
        )
        
        # Attach click handler to all traces
        click_handler = self._create_click_handler()
        for trace in fig.data:
            trace.on_click(click_handler)
        
        display(HBox([fig, self.config_output]))
