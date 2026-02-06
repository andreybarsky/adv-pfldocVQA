import matplotlib.pyplot as plt
import scienceplots
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np 
import math 

plt.style.use(['science', 'ieee', 'no-latex'])

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'lines.linewidth': 3,
    'lines.markersize': 18,
    'lines.markerfacecolor': 'auto',
    'lines.markeredgecolor': 'white',
    'lines.markeredgewidth': 3,  
    'figure.dpi': 300
})

palette = {
    'blue':   '#376a9d', # ANLS-B
    'yellow': '#ffa600', # ANLS-GT
    'red':    '#d12c2c', # ASR
    'gray':   'gray',     # baseline
    'green':  '#4caf50',   # CDMG
    'purple':   '#6a1b9a'    # ANLS-C
}


all_plots_data = {
    

    ############# ASR-CDMG DENIAL OF ANSWER

    "pix2struct_full_doa": {
        "x_axis": [1, 2, 3, 4, 5],
        #"anls_baseline": 51.27,
        "title": "Full-document",
        "asr": [70.60, 51.80, 39.50, 28.20, 21.90], # fatta su file exp3 perché è soltanto doa che calcola quante label sbaglia
        "cdmg": [19.00, 19.23, 20.20, 20.70], # fatta su file exp4 perché è cdmg  
    },

    "pix2struct_patch_doa": {
        "x_axis": [1, 2, 3, 4, 5],
        #"anls_baseline": 51.27,
        "title": "Patch",
        "asr": [70.60, 51.80, 39.50, 28.20, 21.90],
        "cdmg": [11.05, 10.87, 11.75, 11.30], 
    },

    "donut_full_doa": {
        "x_axis": [1, 2, 3, 4, 5],
        #"anls_baseline": 41.14,
        "title": "Full-document",
        "asr": [82.00, 74.10, 69.10, 65.50, 61.80],
        "cdmg": [4.38, 6.00, 6.60, 9.30], 
    },

    "donut_patch_doa": {
        "x_axis": [1, 2, 3, 4, 5],
        #"anls_baseline": 41.14,
        "title": "Patch",
        "asr": [82.00, 74.10, 69.10, 65.50, 61.80],
        "cdmg": [2.33, 2.47, 3.00, 3.70], 
    },

    ############# ANLS DOA - manca ANLS_B che è misurata rispetto alla ground truth (?)

    "cdmg_pix2struct_full_doa": {
        "x_axis": [1, 2, 3, 4, 5],
        "anls_baseline": 51.27,
        "title": r"Full-document",
        #"cdmg": [19.00, 19.23, 20.20, 20.70],
        "anls_b": [27.20, 34.83, 36.88, 38.66, 38.99],  # questa è la vecchia ANLS_GT, che semplicemente diventa anls_b
        "anls_c": [89.72, 89.73, 88.89, 89.37],  
    },

    "cdmg_pix2struct_patch_doa": {
        "x_axis": [1, 2, 3, 4, 5],
        "anls_baseline": 51.27,
        "title": r"Patch",
        #"cdmg": [11.05, 10.87, 11.75, 11.30],
        "anls_b": [49.99, 49.40, 50.31, 50.30, 50.49], # old anls_gt
        "anls_c": [94.12, 94.24, 93.75, 94.05],  
    },

    "cdmg_donut_full_doa": {
        "x_axis": [1, 2, 3, 4, 5],
        "anls_baseline": 41.14,
        "title": r"Full-document",
        #"cdmg": [4.38, 6.00, 6.60, 9.30],
        "anls_b": [35.93, 29.04, 25.42, 22.08, 20.28],
        "anls_c": [53.57, 59.83, 69.62, 91.98],  
    },

    "cdmg_donut_patch_doa": {
        "x_axis": [1, 2, 3, 4, 5],
        "anls_baseline": 41.14,
        "title": r"Patch",
        #"cdmg": [2.33, 2.47, 3.00, 3.70],
        "anls_b": [38.39, 29.75, 26.60, 23.08, 21.34],
        "anls_c": [54.99, 62.53, 72.45, 95.74],  
    },


    ############# ASR-CDMG MULTI ANSWER

    "cdmg_pix2struct_full_multi": {
        "x_axis": [1, 2, 3, 4, 5],
        #"anls_baseline": 51.27,
        "title": "Full-document",
        "asr":  [98.40, 89.60, 79.30, 46.10, 16.70],
        "cdmg": [40.60, 41.43, 42.35, 42.80],  
    },

    "cdmg_pix2struct_patch_multi": {
        "x_axis": [1, 2, 3, 4, 5],
        #"anls_baseline": 51.27,
        "title": "Patch",
        "asr":     [93.70, 55, 32.10, 11.70, 4.80],
        "cdmg": [31.20, 36.60, 40.35, 40.90],
    },

    "cdmg_donut_full_multi": {
        "x_axis": [1, 2, 3, 4, 5],
        #"anls_baseline": 41.14,
        "title": "Full-document",
        "asr":     [74.5,4.4,0.2,0,0],
        "cdmg": [31.80, 37.43, 44.95, 59.00],
        
    },

    "cdmg_donut_patch_multi": {
        "x_axis": [1, 2, 3, 4, 5],
        #"anls_baseline": 41.14,
        "title": "Patch",
        "asr":     [15.3,0.1,0,0,0],
        "cdmg": [12.75, 16.50, 19.60, 24.70],
    },

    ##### ANLS MULTI ANSWER
    "pix2struct_full_multi": {
        "x_axis": [1, 2, 3, 4, 5],
        "anls_baseline": 51.27,
        #"title": r"$\epsilon = 8$",
        "title": "Full-document",
        #"anls_gt": [0.48, 1.35, 2.40, 5.33, 8.34], # non più utile misurarlo in questo caso
        "anls_b":  [98.40, 94.31, 90.46, 79.30, 68.58],
        "anls_c": [77.00, 75.07, 74.76, 74.11]
    },

    "pix2struct_patch_multi": {
        "x_axis": [1, 2, 3, 4, 5],
        "anls_baseline": 51.27,
        #"title": r"$\epsilon = 96$",
        "title": "Patch",
        #"anls_gt": [2.57, 9.14, 13.42, 17.15, 19.03],
        "anls_b":  [93.70, 73.41, 61.25, 48.60, 40.97],
        "anls_c": [76.06, 71.62, 68.14, 68.06],
    },

    "donut_full_multi": {
        "x_axis": [1, 2, 3, 4, 5],
        "anls_baseline": 41.14,
        #"title": r"$\epsilon = 32$",
        "title": "Full-document",
        #"anls_gt": [1.07,1.29,1.27,1.03,1.22], 
        "anls_b":  [75.07,42.53,28.78,23.03,17.92],
        "anls_c": [32.47, 34.48, 38.36, 51.62]
    },

    "donut_patch_multi": {
        "x_axis": [1, 2, 3, 4, 5],
        "anls_baseline": 41.14,
        "title": "Patch",
        #"anls_gt": [5.13,5.17,18.23,16.58,14.02],
        "anls_b":  [21.65,13.53,0.55,0.72,1],
        "anls_c": [46.84, 51.56, 59.24, 77.87],  
    },

}

def draw_plot_content(ax, data_dict, show_ylabel=True):
    
    x_full = data_dict["x_axis"]

    if "anls_b" in data_dict:
        y_data = data_dict["anls_b"]
        ax.plot(x_full[:len(y_data)], y_data, 
                marker='s', linestyle=':', color=palette['green'], label='ANLS-B')

    if "anls_gt" in data_dict:
        y_data = data_dict["anls_gt"]
        ax.plot(x_full[:len(y_data)], y_data, 
                marker='^', linestyle='--', color=palette['yellow'], label='ANLS-GT')

    if "cdmg" in data_dict:
        y_data = data_dict["cdmg"]
        ax.plot(x_full[:len(y_data)], y_data, 
                marker='d', linestyle='-.', color=palette['blue'], label="CDMG")
    
    if "anls_c" in data_dict:
        y_data = data_dict["anls_c"]
        ax.plot(x_full[:len(y_data)], y_data, 
                marker='v', linestyle='-', color=palette['purple'], label="ANLS-C")
        
    if "asr" in data_dict:
        y_data = data_dict["asr"]
        ax.plot(x_full[:len(y_data)], y_data, 
                marker='o', linestyle='-', color=palette['red'], label='ASR')

    if "anls_baseline" in data_dict:
        ax.axhline(y=data_dict["anls_baseline"], color=palette['gray'], 
                   linestyle=':', linewidth=4.0, label="ANLS-Baseline")

    if "title" in data_dict:
        ax.set_title(data_dict["title"], fontsize=24, pad=10, fontweight='bold')

    ax.set_ylim(-5, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis='y', linestyle='-', which='major', alpha=0.5)
    ax.set_xticks(x_full)
    
    if show_ylabel:
        ax.set_ylabel("Evaluation Metric (%)", fontsize=24)
    else:
        ax.set_ylabel("") 

    ax.set_xlabel('QA-pairs in objective (B)', fontsize=24)
    ax.set_box_aspect(1)
    
def create_single_plot(filename_suffix, data_dict):
    
    FIG_WIDTH = 5.0
    FIG_HEIGHT = 3.5
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    if "anls_b" in data_dict:
        ax.plot(data_dict["x_axis"], data_dict["anls_b"], 
                marker='o', linestyle='-', color=palette['blue'], label='ANLS-B')

    if "anls_gt" in data_dict:
        ax.plot(data_dict["x_axis"], data_dict["anls_gt"], 
                marker='o', linestyle='-', color=palette['yellow'], label='ANLS_GT')

    if "asr" in data_dict:
        ax.plot(data_dict["x_axis"], data_dict["asr"], 
                marker='o', linestyle='-', color=palette['red'], label='ASR')
        
    if "cdmg" in data_dict:
        ax.plot(data_dict["x_axis"], data_dict["cdmg"], 
                    marker='o', linestyle='-', color=palette['green'], label="CDMG")
    
    if "anls_c" in data_dict:
        ax.plot(data_dict["x_axis"], data_dict["anls_c"], 
                    marker='o', linestyle='-', color=palette['purple'], label="ANLS-C")

    if "anls_baseline" in data_dict:
        ax.axhline(y=data_dict["anls_baseline"], color=palette['gray'], linestyle=':', linewidth=2.0, label="ANLS-Baseline")

    if "title" in data_dict:
        ax.set_title(data_dict["title"])

    ax.set_ylim(-5, 105)
    ax.grid(axis='y', linestyle='-', which='major', alpha=0.5)
    ax.set_xticks(data_dict["x_axis"])
    
    ax.set_ylabel("Evaluation Metric (%)")
    #ax.set_xlabel('QA-pairs in objective (B)')
    ax.set_xlabel('QA-pairs not in objective (C=M-B)')
    
    plt.tight_layout()

    fig.savefig(f"plot_{filename_suffix}.svg", format="svg", bbox_inches='tight')
    
    print(f"Saved plot_{filename_suffix}")

    return fig, ax

def save_legend(palette, filename="plot_legenda.svg"):
    LEGEND = 2 # 0 = multinanswer, 1 = doa, 2 = cdmg

    common_style = {
        'marker': 'o',
        'linestyle': '-',
        'linewidth': plt.rcParams.get('lines.linewidth', 4),
        'markersize': plt.rcParams.get('lines.markersize', 10),
        'markeredgecolor': plt.rcParams.get('lines.markeredgecolor', 'white'),
        'markeredgewidth': plt.rcParams.get('lines.markeredgewidth', 1.5)
    }
    
    baseline_style = {
        'marker': None,
        'linestyle': ':',
        'linewidth': 2.0, 
    }
    
    handle_anls_gt = mlines.Line2D([], [], color=palette['yellow'], **common_style)
    handle_asr = mlines.Line2D([], [], color=palette['red'], **common_style)
    handle_anls_b = mlines.Line2D([], [], color=palette['blue'], **common_style)
    handle_anls_baseline = mlines.Line2D([], [], color=palette['gray'], **baseline_style)
    handle_cdmg = mlines.Line2D([], [], color=palette['green'], **common_style)
    handle_anls_c = mlines.Line2D([], [], color=palette['purple'], **common_style)

    legend_handles = [handle_anls_baseline]
    legend_labels = ["ANLS-baseline"]

    if LEGEND == 0:
        legend_handles.extend([handle_asr, handle_anls_b, handle_anls_gt])
        legend_labels.extend(["ASR","ANLS-B","ANLS-GT"])
    elif LEGEND == 1:
        legend_handles.append(handle_anls_gt)
        legend_labels.extend(["ANLS-GT"])
    elif LEGEND == 2:
        legend_handles.extend([handle_cdmg, handle_anls_c]) 
        legend_labels.extend(["CDMG","ANLS-C"])

    num_cols = len(legend_handles)
    fig_legend, ax_legend = plt.subplots(figsize=(num_cols * 3.0, 0.7)) 

    ax_legend.legend(legend_handles, legend_labels,
                     loc='center', 
                     ncol=num_cols,
                     frameon=True,
                     facecolor='white',
                     edgecolor='lightgray',
                     bbox_to_anchor=(0.5, 0.5),
                     columnspacing=1.5,
                     labelspacing=0.5,
                     fontsize=plt.rcParams.get('font.size', 10))

    ax_legend.axis('off')
    
    fig_legend.savefig(filename, 
                        format="svg", 
                        bbox_inches='tight',
                        transparent=True)
    plt.close(fig_legend)


TOTAL_WIDTH = 17
HEIGHT_SINGLE_ROW = 5.8
HEIGHT_LEGEND = 0.8 
TOTAL_HEIGHT = (HEIGHT_SINGLE_ROW * 2) + HEIGHT_LEGEND

TITLE_OFFSET_MAP = {
    0: 'Pix2Struct', 
    2: 'Donut'     
}

SCENARIO_LABELS = ["Denial of Answer Attack"]

all_keys = list(all_plots_data.keys())

chunk_size = 8 
num_chunks = math.ceil(len(all_keys) / chunk_size)

print(f"generating n.{num_chunks}")

for chunk_idx in range(num_chunks):
    
    start_index = chunk_idx * chunk_size
    plot_keys = all_keys[start_index : start_index + chunk_size]

    fig = plt.figure(figsize=(TOTAL_WIDTH, TOTAL_HEIGHT))
    gs = gridspec.GridSpec(3, 1, height_ratios=[HEIGHT_SINGLE_ROW, HEIGHT_SINGLE_ROW, HEIGHT_LEGEND])
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], wspace=0.19)
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], wspace=0.19)

    axs = []
    for i in range(4): axs.append(fig.add_subplot(gs_row1[i]))
    for i in range(4): axs.append(fig.add_subplot(gs_row2[i]))

    unique_legend_items = {}

    for i, ax in enumerate(axs):
        if i < len(plot_keys):
            key = plot_keys[i]
            data = all_plots_data[key]
            
            is_first_of_row = (i == 0) or (i == 4)
            draw_plot_content(ax, data, show_ylabel=is_first_of_row) 

            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l not in unique_legend_items:
                    unique_legend_items[l] = h
        else:
            ax.axis('off')

    ax_legend = fig.add_subplot(gs[2])
    ax_legend.axis('off')

    final_labels = list(unique_legend_items.keys())
    final_handles = list(unique_legend_items.values())

    ax_legend.legend(final_handles, final_labels, 
                     loc='center',
                     ncol=len(final_handles),
                     frameon=True,
                     facecolor='white',
                     edgecolor='lightgray',
                     fontsize=20,
                     columnspacing=1.5)

    plt.tight_layout(rect=[0.08, 0, 1.2, 0.92]) 
    
    if len(plot_keys) > 0:
        bbox_0 = axs[0].get_position()
        bbox_1 = axs[1].get_position()
        x_center_left = (bbox_0.x0 + bbox_1.x1) / 2
        
        if 0 in TITLE_OFFSET_MAP:
             fig.text(x_center_left, 0.89, TITLE_OFFSET_MAP[0], 
                      ha='center', va='bottom', fontsize=30, fontweight='bold')

        if len(plot_keys) > 2:
            bbox_2 = axs[2].get_position()
            bbox_3 = axs[3].get_position()
            x_center_right = (bbox_2.x0 + bbox_3.x1) / 2
            
            if 2 in TITLE_OFFSET_MAP:
                fig.text(x_center_right, 0.89, TITLE_OFFSET_MAP[2], 
                         ha='center', va='bottom', fontsize=30, fontweight='bold')

    offset = 0.06 

    if len(plot_keys) > 0:
        bbox_top = axs[0].get_position()
        bbox_bottom = axs[4].get_position()

        x_pos = bbox_top.x0 - offset
        y_center_global = (bbox_top.y1 + bbox_bottom.y0) / 2

        fig.text(x_pos, y_center_global, SCENARIO_LABELS[0], 
                 ha='right', va='center', fontsize=30, fontweight='bold', rotation=90)

    filename = f"plot_paper_{chunk_idx+1}.pdf"
    fig.savefig(filename, format="pdf", bbox_inches='tight')
    print(f"saved {filename}")