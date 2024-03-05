import os
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from cfg.cfg_main import CFG


class WeedMap(object):
    CLASS = 3
    IM_SHAPE = (256, 256, 3)

    def __init__(self, subset):
        self.data_dir = CFG.datasets.nassar.dir
        self.subset_dir = os.path.join(self.data_dir, "train")

        data_list_fn = self.subset_dir + f"/data_list/{subset.value}.txt"
        print(f"Loading {subset.value} set, file list {data_list_fn}")
        self.f = open(data_list_fn)
        count = 0
        self.data_list = []
        for line in self.f:
            self.data_list.append(line[:-1])
            count += 1
        self.len = count

        self.image_dir = os.path.join(self.subset_dir, "data")
        self.mask_dir = os.path.join(self.subset_dir, "mask")
        self.mask_dir_pixel_al = os.path.join(self.subset_dir, "lbl_mode_super{}".format(CFG.activeLearn.pixel_mode))
        self.mask_dir_sp = os.path.join(self.subset_dir,
                                        "sp_mode_{}_{}".format(CFG.activeLearn.sp_mode, CFG.activeLearn.sp_param))
        Path(self.mask_dir_pixel_al).mkdir(exist_ok=True)
        Path(self.mask_dir_sp).mkdir(exist_ok=True)

    def __del__(self):
        self.f.close()

    def __len__(self):
        return self.len

    @staticmethod
    def get_data_from_image(path, is_mask, raw=False, grayscale=False):
        img = img_to_array(load_img(path, color_mode="grayscale" if (grayscale or is_mask) else "rgb"))
        if raw:
            return img
        else:
            return img if is_mask else img / 255.0

    @staticmethod
    def get_mask(raw_mask):
        return raw_mask

    def get_image_name(self, index, use_generated=False):
        """
        get image file name of index with dataset directory
        """
        dir = self.mask_dir_pixel_al if use_generated else self.mask_dir
        return dir, self.data_list[index]

    def __get_datapoint(self, index, use_generated=False):
        image_name = self.data_list[index]
        img_dir = self.image_dir + "/" + image_name + ".png"
        mask_parent = self.mask_dir_pixel_al if use_generated else self.mask_dir
        mask_dir = mask_parent + "/" + image_name + ".png"
        img_array = self.get_data_from_image(img_dir, False)
        mask_array = self.get_data_from_image(mask_dir, True)
        mask = self.get_mask(mask_array)
        return img_array, mask

    def generator(self, use_generated=False):
        X, y, batch_size = self.init_tensors()
        while True:
            loc = 0
            while loc < len(self):
                if (loc + batch_size) < len(self):
                    data_indices = list(range(loc, loc + batch_size))
                else:
                    data_indices = list(
                        range(len(self) - batch_size, len(self))
                    )
                loc += batch_size
                for i, d in enumerate(data_indices):
                    x_i, y_i = self.__get_datapoint(d, use_generated=use_generated)
                    X[i] = x_i
                    y[i] = y_i
                yield X, y

    def generator_indices(self, indices, use_generated=False):
        X, y, batch_size = self.init_tensors()
        while True:
            loc = 0
            while loc < len(indices):
                if (loc + batch_size) < len(indices):
                    data_indices = list(range(loc, loc + batch_size))
                else:
                    data_indices = list(range(len(indices) - batch_size, len(indices)))
                loc += batch_size
                for i, d in enumerate(data_indices):
                    x_i, y_i = self.__get_datapoint(indices[d], use_generated=use_generated)
                    X[i] = x_i
                    y[i] = y_i
                yield X, y

    def get_batch_sample(self, index, batch=None, use_generated=None):
        X, y, batch_size = self.init_tensors(batch)
        data_indices = list(range(index, index + batch_size))
        for i, d in enumerate(data_indices):
            x_i, y_i = self.__get_datapoint(d, use_generated=use_generated)
            X[i] = x_i
            y[i] = y_i
        return X, y

    def init_tensors(self, batch=None):
        batch_size = CFG.batchSize if batch is None else batch
        X = np.zeros(
            (batch_size, self.IM_SHAPE[0], self.IM_SHAPE[1], 3), dtype=np.float32
        )
        y = np.zeros(
            (batch_size, self.IM_SHAPE[0], self.IM_SHAPE[1], 1), dtype=np.int32
        )
        return X, y, batch_size

    def get_sample_sp_map(self, index, sp_param):
        subdir, image_name = self.get_image_name(index)
        mask_dir = self.mask_dir_sp + "/" + image_name + ".txt"
        return np.loadtxt(mask_dir)
