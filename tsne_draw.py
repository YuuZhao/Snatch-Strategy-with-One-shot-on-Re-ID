#%%
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Random state.
RS = 20150101

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

markers = ['o', 'v', 's', 'p', '*', 'h',  'D', 'd', 'P', 'X']
#%%
def scatter(fts_gp, lbs_gp,cams,lbs,labeled,labeled_lbs):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", np.unique(lbs).shape[0]))

    # We create a scatter plot.
    plt.figure(figsize=(40, 40))
    for i,(ft,lb) in enumerate(zip(fts_gp,lbs_gp)):
        sc = plt.scatter(ft[:,0], ft[:,1], lw=0, s=4,
                        c=palette[lb.astype(np.int)],marker=markers[i])
    plt.scatter(labeled[:,0], labeled[:,1], lw=0, s=2,
                        c='black',marker='.')
# %%
data_set = 'mars'
group_name = 'step_6'
fts1 = np.load('tsne/{}/{}/fts_1.npy'.format(data_set,group_name))
lbs1 = np.load('tsne/{}/{}/lbs_1.npy'.format(data_set,group_name))
cams1 = np.load('tsne/{}/{}/cams_1.npy'.format(data_set,group_name))
fts2 = np.load('tsne/{}/{}/fts_2.npy'.format(data_set,group_name))
lbs2 = np.load('tsne/{}/{}/lbs_2.npy'.format(data_set,group_name))
cams2 = np.load('tsne/{}/{}/cams_2.npy'.format(data_set,group_name))
fts = np.vstack((fts1,fts2))
lbs = np.hstack((lbs1,lbs2))
cams = np.hstack((cams1,cams2))

#%%
print(lbs.max(),np.unique(lbs).shape)
# %%
proj = TSNE(random_state=RS).fit_transform(fts)
#%%
fts_gp = []
lbs_gp = []
for cam in np.unique(cams):
    cam_idx = cam == cams
    fts_gp.append(proj[cam_idx])
    lbs_gp.append(lbs[cam_idx])
# %%
scatter(fts_gp, lbs_gp,np.unique(cams),lbs,proj[:625],lbs[:625])
plt.savefig('tsne/{}/{}/all_feas_tsne-generated.svg'.format(data_set,group_name), dpi=400,facecolor='white')
plt.show()

# %%
