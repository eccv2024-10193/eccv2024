import os
import numpy as np
from pathlib import Path
from skimage.segmentation import mark_boundaries
from skimage.io import imsave
from tf.ds.ds_cityscapes import CityScapes
from ds.ds_def import DatasetSubset
from cfg.cfg_main import CFG

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def vis_dataset(data):
    Path(f"sp_vis_cityscapes/sp_{CFG.activeLearn.sp_param}").mkdir(exist_ok=True)
    for i in range(len(data)):
        x,y = data.get_batch_sample(i, batch=1, aug=False)
        dd, name = data.get_image_name(i)
        # print(name)
        name = name.split('/')[-1]
        sp_map = data.get_sample_sp_map(i)
        # print(x[0].shape, sp_map.shape, sp_map.dtype)
        sp_bound_image = mark_boundaries(x[0], sp_map.astype(np.uint))
        imsave(f"sp_vis_cityscapes/sp_{CFG.activeLearn.sp_param}/"+name+".png", sp_bound_image)
        # input("STOP")

data = CityScapes(DatasetSubset.TRAIN, "train")
vis_dataset(data)
data = CityScapes(DatasetSubset.TRAIN, "val")
vis_dataset(data)
