import matplotlib.pyplot as plt
import json
import os
import seaborn as sns

dir = "/home/mila/a/aditya.jain/mothAI/cvpr2024/model_evaluation/plots"
sp_data = json.load(open(os.path.join(dir, "hist_species.json")))
gs_data = json.load(open(os.path.join(dir, "hist_genus.json")))
fm_data = json.load(open(os.path.join(dir, "hist_family.json")))

plt.rcParams['figure.dpi'] = 1400
plt.plot(sp_data.values(), label="Species")
plt.plot(gs_data.values(), label="Genus", alpha=0.6)
plt.plot(fm_data.values(), label="Family", alpha=0.6)
plt.yscale('log')
plt.xlabel("Sorted Classes", size="x-large")
plt.ylabel("Number of Images", size="x-large")
plt.title("AMI-Traps", size="x-large")
plt.grid()
plt.legend(fontsize="large") 
plt.savefig(os.path.join(dir, "ami-traps_distribution_log.png"), bbox_inches="tight")
# plt.savefig(os.path.join(dir, "ami-traps_distribution_log.pdf"), format="pdf")
