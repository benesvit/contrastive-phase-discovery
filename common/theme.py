import numpy as np
from seaborn import cubehelix_palette

PALETTE = cubehelix_palette(10, rot=3, gamma=1.1, hue=1.4, light=0.8).as_hex()
phases = ["FE", "PE", "AFE A", "AFE A\'", "AFE B"]

PALETTE_PRESET = {phase: color for phase, color in zip(phases, PALETTE)}
PALETTE_CONTINUE = PALETTE[len(PALETTE_PRESET):]
PALETTE_PRESET['Noise'] = "#aaaaaa"

PALETTE_SIAM = PALETTE

# PALETTE = ["#a5a806","#ff0990",
#                  "#37FF00", "#3000F1",
#                  "#00FF99", "#8C7B7BFF",
#                 "#D2BC61", "#55de36"]

# PALETTE_SIAM = ["#06a88a","#ffef0e",
#                  "#2F6121", "#CC8FDE",
#                  "#0066FF", "#E89402",
#                 "#FF159A", "#de3636"]

# PALETTE_PRESET = {
#     'PE': "#06a88a",
#     'FE': "#ffef0e",
#     'AFE A': "#ff4a5c",
#     'AFE B': "#CC8FDE",
#     'Noise': "#aaaaaa",
# }



def build_label_map(labels, cluster_names={}, named_first=True):
    """Return (display_labels, palette_dict, legend_order).

    * Clusters in `cluster_names` get that name as display label.
    * Remaining clusters are renumbered 1, 2, 3, …
    * If a display name exists in PALETTE_PRESET, that color is used automatically.
    * Otherwise colors are drawn from PALETTE in order.
    * `legend_order` controls legend ordering: named clusters first, then numbered.
    """
    unique = sorted(set(int(k) for k in np.unique(labels)))

    # Split into named and unnamed
    named_keys = [k for k in unique if k in cluster_names]
    unnamed_keys = [k for k in unique if k not in cluster_names]

    # Build mapping
    label_map = {}
    for k in named_keys:
        label_map[k] = cluster_names[k]
    for i, k in enumerate(unnamed_keys, start=1):
        label_map[k] = str(i)

    display_labels = np.array([label_map[int(l)] for l in labels])

    # Legend order: named first, then numbered
    if named_first:
        ordered_keys = named_keys + unnamed_keys
    else:
        ordered_keys = unique

    legend_order = [label_map[k] for k in ordered_keys]

    # Build color dict — use PALETTE_PRESET when display name matches
    color_idx = 0
    palette_dict = {}
    for k in ordered_keys:
        dl = label_map[k]
        if dl in palette_dict:
            continue
        if dl in PALETTE_PRESET:
            palette_dict[dl] = PALETTE_PRESET[dl]
        else:
            palette_dict[dl] = PALETTE_CONTINUE[color_idx % len(PALETTE_CONTINUE)]
            color_idx += 1

    return display_labels, palette_dict, legend_order


# def build_label_map(labels, colors, names = None, name_prefix = 'U'):
#     """Return (display_labels, palette_dict, legend_order).

#     * Clusters in `cluster_names` get that name as display label.
#     * Remaining clusters are renumbered 1, 2, 3, …
#     * If a display name exists in PALETTE_PRESET, that color is used automatically.
#     * Otherwise colors are drawn from PALETTE in order.
#     * `legend_order` controls legend ordering: named clusters first, then numbered.
#     """
#     unique = np.unique(labels)
#     if len(unique) != colors:
#         raise ValueError('colors length')   
    
#     if names is not None:
#         name_dict = {l:n for l,n in zip(range(len(names), names))}
#     else:
#         name_dict = {l:n for l,n in range(len(unique))}
#     print(name_dict)
#     # labels_renamed = []

#     # return display_labels, palette_dict, legend_order