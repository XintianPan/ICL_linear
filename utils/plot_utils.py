import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import torch
from matplotlib.patches import FancyArrowPatch

# Create a custom colormap that centers 'cividis' with white at zero
cividis_with_white = LinearSegmentedColormap.from_list(
    'cividis_white_center', 
    ['#0D2758', 'white', '#A32015'],  # Colors at low, middle, and high points in 'cividis'
    N=256
)

def plot_attn_matrices(model, n_layers, d, n_head, adj_factor = [1,1], dy = 1, save = False):
    """
    Plots the Query-Key (QK) and Output-Value (OV) matrices for multiple attention heads, combining QK and OV
    for each head in a single plot with a subplot layout (size ratio 4:1).

    Parameters:
    - model: dict
        A dictionary containing the weights of the attention layer. Expected keys include:
        - 'attention_layer.q_proj.weight': Query projection weight matrix of shape (d_embed * n_head, d_embed).
        - 'attention_layer.k_proj.weight': Key projection weight matrix of shape (d_embed * n_head, d_embed).
        - 'attention_layer.v_proj.weight': Value projection weight matrix of shape (d_embed * n_head, d_embed).
        - 'attention_layer.o_proj.weight': Output projection weight matrix of shape (d_embed, d_embed * n_head).
    - d: int
        Dimensionality of the model (base size \(d\), where \(d_embed = d + 1\)).
    - n_head: int
        Number of attention heads.

    Returns:
    - None
        Displays each head's QK and OV plots in separate figures.
    """
    # Adjust embedding size to account for d + 1
    d_embed = d + dy

    # Extract projection matrices
    W_Q = model['attention_layer.q_proj.weight']  # Shape: (d_embed * n_head, d_embed)
    W_K = model['attention_layer.k_proj.weight']  # Shape: (d_embed * n_head, d_embed)
    W_V = model['attention_layer.v_proj.weight']  # Shape: (d_embed * n_head, d_embed)
    W_O = model['attention_layer.o_proj.weight']  # Shape: (d_embed, d_embed * n_head)

    # Ensure the embedding size is consistent with the number of heads
    assert W_Q.shape[0] == d_embed * n_head, f"Expected W_Q rows to match {d_embed * n_head}, but got {W_Q.shape[0]}"
    assert W_Q.shape[1] == d_embed, f"Expected W_Q columns to match {d_embed}, but got {W_Q.shape[1]}"

    for head_idx in range(n_head):
        # Slice the weights for this head
        W_Q_h = W_Q[head_idx * d_embed:(head_idx + 1) * d_embed, :]  # Shape: (d_embed, d_embed)
        W_K_h = W_K[head_idx * d_embed:(head_idx + 1) * d_embed, :]  # Transpose: Shape (d_embed, d_embed)
        W_V_h = W_V[head_idx * d_embed:(head_idx + 1) * d_embed, :]  # Shape: (d_embed, d_embed)
        W_O_h = W_O[:, head_idx * d_embed:(head_idx + 1) * d_embed]  # Shape: (d_embed, d_embed)

        # Compute QK and OV matrices
        QK = (W_K_h.T @ W_Q_h) / np.sqrt(d_embed)  # Scaled QK matrix
        OV = W_V_h.T @ W_O_h.T  # OV matrix

        QK_np = QK.cpu().detach().numpy() if hasattr(QK, 'cpu') else QK
        OV_np = OV.cpu().detach().numpy() if hasattr(OV, 'cpu') else OV
        #print(QK_np)

        QK_max_abs_value = np.abs(QK_np).max() * adj_factor[0]
        OV_max_abs_value = np.abs(OV_np).max() * adj_factor[1]

        # Create a single figure with two subplots for QK and OV
        fig, (ax_qk, ax_ov) = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [8, 1]})

        # Adjust spacing between plots
        plt.subplots_adjust(wspace=-0.4)

        # --- Plot QK Matrix ---
        im_qk = ax_qk.imshow(QK_np, cmap=cividis_with_white, 
                             vmax = QK_max_abs_value, vmin = - QK_max_abs_value, 
                             aspect='equal', interpolation='none')
        ax_qk.set_title(f'Head {head_idx + 1} QK', fontsize=17)
        fig.colorbar(im_qk, pad=0.04)
        # Add green dotted lines to QK plot
        ax_qk.axvline(x=d-0.5, color='green', linestyle='--', linewidth=5)  # Vertical line
        ax_qk.axhline(y=d-0.5, color='green', linestyle='--', linewidth=5)  # Horizontal line

        # --- Plot OV Matrix ---
        im_ov = ax_ov.imshow(OV_np, cmap=cividis_with_white, 
                             vmax = OV_max_abs_value, vmin = - OV_max_abs_value, 
                             aspect='equal', interpolation='none')
        ax_ov.set_title(f'Head {head_idx + 1} OV', fontsize=17)
        ax_ov.set_xticks([])  # Remove x-axis measures
        fig.colorbar(im_ov, pad=0.3, aspect=25, shrink=1)
        ax_ov.axhline(y=d-0.5, color='green', linestyle='--', linewidth=5)  # Horizontal line

        plt.tight_layout()
        plt.show()
        if save:
            fig.savefig(f"Layer_{n_layers}_Head_{n_head}_d_{d}_attnmtrx_HeadId{head_idx+1}.pdf", format="pdf", bbox_inches='tight')

def plot_heatmap_sequence_all_heads(n_layers, d, tensor_list, indices, training_step, adj_factor=1, save=False):
    """
    Plots a sequence of heatmaps for all heads at specified steps, sharing a single color bar.

    Parameters:
    - tensor_list (list of lists): A list of size (n_head, n_step), where each element is a tensor matrix.
    - indices (list of int): List of indices (steps) specifying which matrices to plot for all heads.
    - training_step (int): Total number of training steps.

    Returns:
    - None
    """
    n_head = len(tensor_list)  # Number of heads
    n_step = len(tensor_list[0]) if n_head > 0 else 0  # Number of steps per head

    # Extract matrices for all heads and valid steps
    all_matrices = []
    for head_idx in range(n_head):
        head_matrices = [tensor_list[head_idx][i].cpu().detach().numpy() for i in indices]
        all_matrices.extend(head_matrices)

    # Compute global color scale
    vmax = max(np.max(matrix) for matrix in all_matrices) * adj_factor

    # Create subplots
    fig, axes = plt.subplots(
        n_head, len(indices), figsize=(5 * len(indices), 5 * n_head),
        sharex=True, sharey=True, squeeze=False
    )

    # Plot each heatmap
    for head_idx in range(n_head):
        for plot_idx, step_idx in enumerate(indices):
            matrix = tensor_list[head_idx][step_idx]
            matrix = matrix.cpu().detach().numpy() if isinstance(matrix, torch.Tensor) else matrix
            ax = axes[head_idx][plot_idx]
            im = ax.imshow(matrix.T, cmap=cividis_with_white, aspect="equal", vmin=-vmax, vmax=vmax)

            # Add titles and labels
            if head_idx == 0:
                if step_idx != 0:
                    ax.set_title(f"Step {int(step_idx / (n_step-1) * training_step / 1000)}k", fontsize=22)
                else:
                    ax.set_title(f"Step 1", fontsize=22)
            if plot_idx == 0:
                ax.set_ylabel(f"Head {head_idx + 1} KQ", fontsize=22)

    # Add a single shared color bar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, shrink=0.96)

    plt.show()
    if save:
            fig.savefig(f"Layer_{n_layers}_Head_{n_head}_d_{d}_attnmtrx_dynamic.pdf", format="pdf", bbox_inches='tight')

def plot_heatmap_sequence_per_head(n_layers, d, tensor_list, indices, training_step, adj_factor=1, save=False):
    """
    Plots a sequence of heatmaps for each head in separate figures.

    Parameters:
    - tensor_list (list of lists): A list of size (n_head, n_step), where each element is a tensor matrix.
    - indices (list of int): List of indices (steps) specifying which matrices to plot for all heads.
    - training_step (int): Total number of training steps.
    - adj_factor (float): Adjustment factor for the color scale.
    - save (bool): Whether to save each figure as a PDF.

    Returns:
    - None
    """
    n_head = len(tensor_list)  # Number of heads
    n_step = len(tensor_list[0]) if n_head > 0 else 0  # Number of steps per head

    # Compute global color scale
    all_matrices = []
    for head_idx in range(n_head):
        head_matrices = [tensor_list[head_idx][i].cpu().detach().numpy() for i in indices]
        all_matrices.extend(head_matrices)
    vmax = max(np.max(matrix) for matrix in all_matrices) * adj_factor

    for head_idx in range(n_head):
        # Create a figure for each head
        fig, axes = plt.subplots(
            1, len(indices), figsize=(5 * len(indices)/2, 5), sharex=True, sharey=True, squeeze=False
        )
        plt.subplots_adjust(wspace=0.01)

        # Plot heatmaps for the current head
        for plot_idx, step_idx in enumerate(indices):
            matrix = tensor_list[head_idx][step_idx]
            matrix = matrix.cpu().detach().numpy() if isinstance(matrix, torch.Tensor) else matrix
            ax = axes[0][plot_idx]
            im = ax.imshow(matrix, cmap=cividis_with_white, aspect="equal", vmin=-vmax, vmax=vmax)

            # Add titles and labels
            if step_idx != 0:
                ax.set_title(f"Step {int(step_idx / (n_step-1) * training_step/1000)}k", fontsize=22)
            else:
                ax.set_title(f"Step 1", fontsize=22)

        axes[0][0].set_ylabel(f"Head {head_idx + 1} OV", fontsize=22)

        # Add a single shared color bar for this head
        cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.1, shrink=1)

        # Show or save the figure
        plt.show()
        if save:
            fig.savefig(f"Layer_{n_layers}_Head_{n_head}_d_{d}_attnmtrx_dynamic_HeadId{head_idx+1}.pdf", format="pdf", bbox_inches="tight")

def plot_2_head_dynamics(model_dict, n_layers, n_head, d, save_fig=False):
    colors= ['#0D2758','#A32015', "#60656F", "#DEA54B"]

    # Create the figure
    plt.figure(figsize=(8, 5))

    for idx, dynamics_list in enumerate(model_dict['qk_dynamics']):
        # Convert to NumPy array
        data = [x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in dynamics_list][:200]
        x = [(i-1) * 200000 / len(data)  for i in range(len(data))]
        
        # Plot the points (small markers) and lines
        plt.plot(x, data, color=colors[idx], linestyle='--', linewidth=2, label=f'Head {idx + 1}')
        plt.scatter(x, data, color=colors[idx], s=4, marker='o')  # Smaller points

    # Add horizontal reference line
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=0.13, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=-0.13, color='black', linestyle='--', linewidth=1)

    # Add a double arrow
    arrow = FancyArrowPatch(
        posA=(120000, 0.135),  # Start position (x, y)
        posB=(120000, -0.005),  # End position (x, y)
        arrowstyle='<->',  # Double-headed arrow
        mutation_scale=15,  # Size of the arrow
        color='gray',  # Color of the arrow
        linewidth=1.2,  # Line width
    )
    plt.gca().add_patch(arrow)

    # Add a double arrow
    arrow = FancyArrowPatch(
        posA=(120000, -0.135),  # Start position (x, y)
        posB=(120000, 0.005),  # End position (x, y)
        arrowstyle='<->',  # Double-headed arrow
        mutation_scale=15,  # Size of the arrow
        color='gray',  # Color of the arrow
        linewidth=1.2,  # Line width
    )
    plt.gca().add_patch(arrow)

    # Customize x-axis ticks
    xticks = np.linspace(0, 150000, num=9)  # Generate ticks (e.g., 0, 100000, 200000, ...)
    plt.xticks(ticks=xticks, labels=[f'{int(tick/1000)}k' for tick in xticks])

    # Customize y-axis ticks
    yticks = plt.yticks()[0]
    plt.yticks(ticks=np.append(yticks, [0.13, -0.13]), labels=[f'{tick:.2f}' if tick in [0.13, -0.13] else f'{tick:.2f}' for tick in np.append(yticks, [0.13, -0.13])])

    plt.text(50000, 0.05, "Positive Heads (1)", fontsize=18, color="black")
    #plt.text(180000, 0.03, "Dummy Heads (3)", fontsize=15, color="black")
    plt.text(48000, -0.22, "Negative Heads (2)", fontsize=18, color="black")

    # Add labels, title, and legend
    plt.xlim(-2000, 150000)
    plt.xlabel('Steps', fontsize=15)
    plt.ylabel('Average of Diagnal Elements of KQ', fontsize=15)
    plt.title('QK Dynamics', fontsize=15)
    plt.legend(fontsize=18, loc ="upper right")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"Layer_{n_layers}_Head_{n_head}_d_{d}_kq_dynamic_main.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    sum_negative = []

    for idx, dynamics_list in enumerate(model_dict['ov_dynamics']):
        # Convert to NumPy array
        data = [x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in dynamics_list][:400]
        x = [(i-1) * 400000 / len(data) for i in range(len(data))]
        #if idx == 0 or idx ==3:
        #    sum_negative.append(data)
        
        # Plot the points (small markers) and lines
        plt.plot(x, data, color=colors[idx], linestyle='--', linewidth=2, label=f'Head {idx + 1}')
        plt.scatter(x, data, color=colors[idx], s=4, marker='o')  # Smaller points

    #sum_negative = [sum_negative[0][i] + sum_negative[1][i] for i in range(len(sum_negative[0]))]
    #plt.plot(x, sum_negative, color="#AD8A64", linestyle=':', linewidth=2, label=f'Sum of Head 1&4')
    #plt.scatter(x, sum_negative, color="#AD8A64", s=4, marker='o')  # Smaller points
        
    # Add horizontal reference line
    plt.axhline(y=0, color='#0D2758', linestyle='--', linewidth=1)
    plt.axhline(y=3.5, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=-3.5, color='black', linestyle='--', linewidth=1)

    # Add a double arrow
    arrow = FancyArrowPatch(
        posA=(350000, 3.55),  # Start position (x, y)
        posB=(350000, -0.01),  # End position (x, y)
        arrowstyle='<->',  # Double-headed arrow
        mutation_scale=15,  # Size of the arrow
        color='gray',  # Color of the arrow
        linewidth=1.2,  # Line width
    )
    plt.gca().add_patch(arrow)

    # Add a double arrow
    arrow = FancyArrowPatch(
        posA=(350000, -3.55),  # Start position (x, y)
        posB=(350000, 0.01),  # End position (x, y)
        arrowstyle='<->',  # Double-headed arrow
        mutation_scale=15,  # Size of the arrow
        color='gray',  # Color of the arrow
        linewidth=1.2,  # Line width
    )
    plt.gca().add_patch(arrow)

    plt.text(170000, 2.5, "Positive Heads (3)", fontsize=18, color="black")
    #plt.text(180000, 0.3, "Dummy Heads (1)", fontsize=15, color="black")
    plt.text(170000, -2.7, "Negative Heads (2)", fontsize=18, color="black")

    xticks = np.linspace(0, 400000, num=9)  # Generate ticks (e.g., 0, 100000, 200000, ...)
    plt.xticks(ticks=xticks, labels=[f'{int(tick/1000)}k' for tick in xticks])

    yticks = plt.yticks()[0]
    plt.yticks(ticks=np.append(yticks, [3.5, -3.5]), labels=[f'{tick:.2f}' if tick in [3.5, -3.5] else f'{tick:.2f}' for tick in np.append(yticks, [3.5, -3.5])])
    # Add labels, title, and legend
    plt.xlim(-10000, 400000)
    plt.xlabel('Steps', fontsize=15)
    plt.ylabel('Value of the Last Element of OV', fontsize=15)
    plt.title('OV Dynamics', fontsize=15)
    plt.legend(fontsize=18, loc ="upper left")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"Layer_{n_layers}_Head_{n_head}_d_{d}_ov_dynamic.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def plot_4_head_dynamics(model_dict, n_layers, n_head, d, save_fig=False):
    colors= ['#0D2758','#A32015', "#60656F", "#DEA54B"]

    # Create the figure
    plt.figure(figsize=(5, 5))

    for idx, dynamics_list in enumerate(model_dict['qk_dynamics']):
        # Convert to NumPy array
        data = [x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in dynamics_list][:200]
        x = [(i-1) * 200000 / len(data)  for i in range(len(data))]
        
        # Plot the points (small markers) and lines
        plt.plot(x, data, color=colors[idx], linestyle='--', linewidth=2, label=f'Head {idx + 1}')
        plt.scatter(x, data, color=colors[idx], s=4, marker='o')  # Smaller points

    # Add horizontal reference line
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=0.13, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=-0.13, color='black', linestyle='--', linewidth=1)

    # Add a double arrow
    arrow = FancyArrowPatch(
        posA=(120000, 0.135),  # Start position (x, y)
        posB=(120000, -0.005),  # End position (x, y)
        arrowstyle='<->',  # Double-headed arrow
        mutation_scale=15,  # Size of the arrow
        color='gray',  # Color of the arrow
        linewidth=1.2,  # Line width
    )
    plt.gca().add_patch(arrow)

    # Add a double arrow
    arrow = FancyArrowPatch(
        posA=(120000, -0.135),  # Start position (x, y)
        posB=(120000, 0.005),  # End position (x, y)
        arrowstyle='<->',  # Double-headed arrow
        mutation_scale=15,  # Size of the arrow
        color='gray',  # Color of the arrow
        linewidth=1.2,  # Line width
    )
    plt.gca().add_patch(arrow)

    # Customize x-axis ticks
    xticks = np.linspace(0, 150000, num=9)  # Generate ticks (e.g., 0, 100000, 200000, ...)
    plt.xticks(ticks=xticks, labels=[f'{int(tick/1000)}k' for tick in xticks])

    # Customize y-axis ticks
    yticks = plt.yticks()[0]
    plt.yticks(ticks=np.append(yticks, [0.13, -0.13]), labels=[f'{tick:.2f}' if tick in [0.13, -0.13] else f'{tick:.2f}' for tick in np.append(yticks, [0.13, -0.13])])

    #plt.text(50000, 0.05, "Positive Heads (1)", fontsize=18, color="black")
    #plt.text(180000, 0.03, "Dummy Heads (3)", fontsize=15, color="black")
    #plt.text(48000, -0.22, "Negative Heads (2)", fontsize=18, color="black")

    # Add labels, title, and legend
    plt.xlim(-2000, 150000)
    plt.ylim(-0.3, 0.3)
    plt.xlabel('Steps', fontsize=15)
    plt.ylabel('Average of Diagnal Elements of KQ', fontsize=15)
    plt.title('QK Dynamics', fontsize=15)
    plt.legend(fontsize=16, loc ="upper right")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"Layer_{n_layers}_Head_{n_head}_d_{d}_kq_dynamic_main.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Create the figure
    plt.figure(figsize=(5, 5))
    sum_negative = []

    for idx, dynamics_list in enumerate(model_dict['ov_dynamics']):
        # Convert to NumPy array
        data = [x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in dynamics_list][:400]
        x = [(i-1) * 400000 / len(data) for i in range(len(data))]
        if idx == 0 or idx ==2:
            sum_negative.append(data)
        
        # Plot the points (small markers) and lines
        plt.plot(x, data, color=colors[idx], linestyle='--', linewidth=2, label=f'Head {idx + 1}')
        plt.scatter(x, data, color=colors[idx], s=4, marker='o')  # Smaller points

    sum_negative = [sum_negative[0][i] + sum_negative[1][i] for i in range(len(sum_negative[0]))]
    plt.plot(x, sum_negative, color="#AD8A64", linestyle='--', linewidth=2, label=f'Sum of Heads')
    plt.scatter(x, sum_negative, color="#AD8A64", s=4, marker='o')  # Smaller points
        
    # Add horizontal reference line
    plt.axhline(y=0, color='#0D2758', linestyle='--', linewidth=1)
    plt.axhline(y=3.5, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=-3.5, color='black', linestyle='--', linewidth=1)

    # Add a double arrow
    arrow = FancyArrowPatch(
        posA=(120000, 3.5),  # Start position (x, y)
        posB=(120000, -0.01),  # End position (x, y)
        arrowstyle='<->',  # Double-headed arrow
        mutation_scale=15,  # Size of the arrow
        color='gray',  # Color of the arrow
        linewidth=1.2,  # Line width
    )
    plt.gca().add_patch(arrow)

    # Add a double arrow
    arrow = FancyArrowPatch(
        posA=(120000, -3.5),  # Start position (x, y)
        posB=(120000, 0.01),  # End position (x, y)
        arrowstyle='<->',  # Double-headed arrow
        mutation_scale=15,  # Size of the arrow
        color='gray',  # Color of the arrow
        linewidth=1.2,  # Line width
    )
    plt.gca().add_patch(arrow)

    #plt.text(65000, 2.2, "Positive Heads (1)", fontsize=18, color="black")
    #plt.text(180000, 0.03, "Dummy Heads (3)", fontsize=15, color="black")
    #plt.text(63000, -2.5, "Negative Heads (2)", fontsize=18, color="black")

    xticks = np.linspace(0, 150000, num=9)  # Generate ticks (e.g., 0, 100000, 200000, ...)
    plt.xticks(ticks=xticks, labels=[f'{int(tick/1000)}k' for tick in xticks])

    yticks = plt.yticks()[0]
    plt.yticks(ticks=np.append(yticks, [3.5, -3.5]), labels=[f'{tick:.2f}' if tick in [3.5, -3.5] else f'{tick:.2f}' for tick in np.append(yticks, [3.5, -3.5])])

    # Add labels, title, and legend
    plt.xlim(-2000, 150000)
    plt.xlabel('Steps', fontsize=15)
    plt.ylabel('Value of the Last Element of OV', fontsize=15)
    plt.title('OV Dynamics', fontsize=15)
    plt.legend(fontsize=16, loc ="upper left")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"Layer_{n_layers}_Head_{n_head}_d_{d}_ov_dynamic_main.pdf", format="pdf", bbox_inches="tight")
    plt.show()