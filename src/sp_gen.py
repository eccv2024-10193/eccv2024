import os
import numpy as np

from tf.ds.ds_weed_map import WeedMap
from ds.ds_def import DatasetSubset
from cfg.cfg_main import CFG
from skimage.segmentation import felzenszwalb, slic


def gen_sp_for_data(data, sp_param, sp_mode, stop_index):
    sp_total = 0
    for i in range(len(data))[:stop_index]:
        x, y = data.get_batch_sample(i, batch=1)
        if sp_mode == 0:
            segments = felzenszwalb(x[0], scale=sp_param)
        elif sp_mode == 1:
            segments = slic(x[0], start_label=0, n_segments=sp_param, compactness=10, sigma=0)
        subdir, image_name = data.get_image_name(i)
        mask_dir = os.path.join(data.subset_dir, "sp_mode_{}_{}".format(sp_mode, sp_param)) + "/" + image_name + ".txt"
        if segments.shape != x[0].shape[:-1]:
            print("WARNING, Wrong Shape: {}".format(segments.shape))
        sp_total += len(np.unique(segments))
        np.savetxt(mask_dir, segments)
    print(f"target {sp_param}, Ave sp count {sp_total / data.len}")


train = WeedMap(DatasetSubset.TRAIN)
val = WeedMap(DatasetSubset.VAL)

gen_sp_for_data(train, sp_param=CFG.activeLearn.sp_param, sp_mode=CFG.activeLearn.sp_mode, stop_index=train.len)
gen_sp_for_data(val, sp_param=CFG.activeLearn.sp_param, sp_mode=CFG.activeLearn.sp_mode, stop_index=val.len)
