import os

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D

# Load secrets and config from optional .env file
load_dotenv()

dir = "./plots"
fm_data = pd.read_csv(
    os.path.join(dir, "w-europe_resnet50_baseline_run2-fm_acc_rej_vs_conf.csv"),
    index_col=0,
)
gs_data = pd.read_csv(
    os.path.join(dir, "w-europe_resnet50_baseline_run2-gs_acc_rej_vs_conf.csv"),
    index_col=0,
)
sp_data = pd.read_csv(
    os.path.join(dir, "w-europe_resnet50_baseline_run2-sp_acc_rej_vs_conf.csv"),
    index_col=0,
)
acc_data_fm = []
rej_data_fm = []
acc_data_gs = []
rej_data_gs = []
acc_data_sp = []
rej_data_sp = []
conf_thresh = [thresh for thresh in range(10, 100, 10)]

# Iterate over species data
for column in sp_data.columns:
    acc_data_sp.append(
        round(
            sp_data.loc["correct"][column] / sp_data.loc["conf-total"][column] * 100, 2
        )
    )
    rej_data_sp.append(
        round(
            sp_data.loc["reject"][column] / sp_data.loc["moths-total"][column] * 100, 2
        )
    )
    acc_data_gs.append(
        round(
            gs_data.loc["correct"][column] / gs_data.loc["conf-total"][column] * 100, 2
        )
    )
    rej_data_gs.append(
        round(
            gs_data.loc["reject"][column] / gs_data.loc["moths-total"][column] * 100, 2
        )
    )
    acc_data_fm.append(
        round(
            fm_data.loc["correct"][column] / fm_data.loc["conf-total"][column] * 100, 2
        )
    )
    rej_data_fm.append(
        round(
            fm_data.loc["reject"][column] / fm_data.loc["moths-total"][column] * 100, 2
        )
    )


fig, ax1 = plt.subplots(figsize=(6, 5))
ax2 = ax1.twinx()
plt.rcParams["figure.dpi"] = 1400

legend_elements = [
    Line2D([0], [0], color="black", label="Top-1 Accuracy"),
    Line2D([0], [0], color="black", linestyle="dashed", label="Rejection Rate"),
]

ax1.plot(conf_thresh, acc_data_sp, label="Species")
ax1.plot(conf_thresh, acc_data_gs, label="Genus")
ax1.plot(conf_thresh, acc_data_fm, label="Family")
ax2.plot(conf_thresh, rej_data_sp, linestyle="dashed")
ax2.plot(conf_thresh, rej_data_gs, linestyle="dashed")
ax2.plot(conf_thresh, rej_data_fm, linestyle="dashed")
ax1.set_xlabel("Confidence Threshold (%)", size="large")
ax1.set_ylabel("Top-1 Accuracy (%)", size="large")
ax2.set_ylabel("Rejection Rate (%)", size="large")
ax1.grid()
ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
ax2.legend(handles=legend_elements)
plt.savefig(
    os.path.join(dir, "w-europe_resnet50_conf_rej_acc.png"), bbox_inches="tight"
)
plt.savefig(
    os.path.join(dir, "w-europe_resnet50_conf_rej_acc.pdf"), bbox_inches="tight"
)
