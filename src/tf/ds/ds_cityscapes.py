import os
import numpy as np
import albumentations as A
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from cfg.cfg_main import CFG


class CityScapes(object):
    IM_SHAPE = (512, 1024, 3)
    GT_SHAPE = (512, 1024, 1)

    NUM_CLASS = 19

    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    # Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    # Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    # Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    # Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    # Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    # Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    # Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    # Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    # Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    # Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    # Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    # Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    # Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    # Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    # Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    # Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    # Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    # Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    # Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    # Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    # Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    # Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    # Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    # Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    # Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    # Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    # Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    # Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    # Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    # Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    # Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    # Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    # Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    # Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    # Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ID_MAP = {
        0: (255, (0, 0, 0)),
        1: (255, (0, 0, 0)),
        2: (255, (0, 0, 0)),
        3: (255, (0, 0, 0)),
        4: (255, (0, 0, 0)),
        5: (255, (111, 74, 0)),
        6: (255, (81, 0, 81)),
        7: (0, (128, 64, 128)),
        8: (1, (244, 35, 232)),
        9: (255, (250, 170, 160)),
        10: (255, (230, 150, 140)),
        11: (2, (70, 70, 70)),
        12: (3, (102, 102, 156)),
        13: (4, (190, 153, 153)),
        14: (255, (180, 165, 180)),
        15: (255, (150, 100, 100)),
        16: (255, (150, 120, 90)),
        17: (5, (153, 153, 153)),
        18: (255, (153, 153, 153)),
        19: (6, (250, 170, 30)),
        20: (7, (220, 220, 0)),
        21: (8, (107, 142, 35)),
        22: (9, (152, 251, 152)),
        23: (10, (70, 130, 180)),
        24: (11, (220, 20, 60)),
        25: (12, (255, 0, 0)),
        26: (13, (0, 0, 142)),
        27: (14, (0, 0, 70)),
        28: (15, (0, 60, 100)),
        29: (255, (0, 0, 90)),
        30: (255, (0, 0, 110)),
        31: (16, (0, 80, 100)),
        32: (17, (0, 0, 230)),
        33: (18, (119, 11, 32)),
        -1: (255, (0, 0, 142)),
    }

    label_color_dict = {
        255: (0, 0, 0),
        0: (128, 64, 128),
        1: (244, 35, 232),
        2: (70, 70, 70),
        3: (102, 102, 156),
        4: (190, 153, 153),
        5: (153, 153, 153),
        6: (250, 170, 30),
        7: (220, 220, 0),
        8: (107, 142, 35),
        9: (152, 251, 152),
        10: (70, 130, 180),
        11: (220, 20, 60),
        12: (255, 0, 0),
        13: (0, 0, 142),
        14: (0, 0, 70),
        15: (0, 60, 100),
        16: (0, 80, 100),
        17: (0, 0, 230),
        18: (119, 11, 32),
    }

    label_name_dict = {
        255: 'other',
        0: 'road',
        1: 'sidewalk',
        2: 'building',
        3: 'wall',
        4: 'fence',
        5: 'pole',
        6: 'traffic light',
        7: 'traffic sign',
        8: 'vegetation',
        9: 'terrain',
        10: 'sky',
        11: 'person',
        12: 'rider',
        13: 'car',
        14: 'truck',
        15: 'bus',
        16: 'train',
        17: 'motorcycle',
        18: 'bicycle',
    }

    pixel_dist = [37.11, 6.15, 22.73, 0.69, 0.91, 1.19, 0.2, 0.55, 15.94, 1.14, 4.03, 1.2, 0.13, 6.79, 0.29, 0.22, 0.24,
                  0.09, 0.4]
    image_dist = [8.47, 8.13, 8.48, 2.9, 3.79, 8.51, 4.73, 8.09, 8.34, 4.69, 7.66, 6.67, 2.96, 8.15, 1.1, 0.77, 0.41,
                  1.48, 4.66]
    class_dist = np.array(pixel_dist) + np.array(image_dist)
    class_sum = np.sum(class_dist)
    class_weights = np.zeros(len(class_dist))
    for i in range(len(class_dist)):
        class_weights[i] = (1 / class_dist[i]) * (class_sum / 2.0)
    class_weights = class_weights / np.min(class_weights)

    def __init__(self, subset, op_type):
        self.data_dir = CFG.datasets.cityscapes.dir
        self.image_dir = "leftImg8bit_trainvaltest/leftImg8bit/"
        self.label_dir = "gtFine_trainvaltest/gtFine/"
        self.image_name = "leftImg8bit.png"
        self.label_name = "gtFine_labelIds.png"

        self.subset_image_dir = self.data_dir + self.image_dir + subset.value
        self.subset_label_dir = self.data_dir + self.label_dir + subset.value
        self.subset_label_dir_pixel_al = os.path.join(self.subset_label_dir,
                                                      "lbl_mode_super{}".format(CFG.activeLearn.pixel_mode))
        self.subset_label_dir_sp = os.path.join(self.subset_label_dir, "sp_mode_{}_{}".format(CFG.activeLearn.sp_mode,
                                                                                              CFG.activeLearn.sp_param))

        Path(self.subset_label_dir_pixel_al).mkdir(exist_ok=True)
        Path(self.subset_label_dir_sp).mkdir(exist_ok=True)

        data_list_fn = self.subset_image_dir + f"/{subset.value}.txt"
        print(f"Loading {subset.value} set, file list {data_list_fn}")
        self.op_type = op_type
        self.f = open(data_list_fn)
        count = 0
        self.data_list = []
        for line in self.f:
            self.data_list.append(line[:-1])
            count += 1
        if op_type == 'train':
            self.data_list = self.data_list[:int(0.9 * count)]  # use first 90% to train and rest as val
        elif op_type == 'val':
            self.data_list = self.data_list[int(0.9 * count):]
        elif op_type == 'test':
            pass
        else:
            pass
        self.len = len(self.data_list)
        print(f"Loaded dataset file, Count: {self.len}")

        if op_type == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.Sequential([
                        A.RandomCrop(int(self.IM_SHAPE[0] * 0.6), int(self.IM_SHAPE[1] * 0.6)),
                        A.PadIfNeeded(min_height=self.IM_SHAPE[0], min_width=self.IM_SHAPE[1], )
                    ]),
                    A.Sequential([
                        A.RandomCrop(int(self.IM_SHAPE[0] * 0.8), int(self.IM_SHAPE[1] * 0.8)),
                        A.PadIfNeeded(min_height=self.IM_SHAPE[0], min_width=self.IM_SHAPE[1], )
                    ])
                ], p=0.6),
                A.RandomBrightnessContrast(p=0.8),
                A.RandomGamma(p=0.8)
            ])
        else:
            self.transform = None

    def __del__(self):
        self.f.close()

    def __len__(self):
        return self.len

    def id2trainid(self, id):
        return self.ID_MAP[id][0]

    def id2rgb(self, id):
        return self.ID_MAP[id][1]

    @staticmethod
    def get_data_from_image(path, is_mask, raw=False, grayscale=False):
        img = img_to_array(load_img(path, color_mode="grayscale" if (grayscale or is_mask) else "rgb"))
        if raw:
            return img
        else:
            return img if is_mask else img / 255.0

    def get_image_name(self, index, use_generated=False):
        """
        get image file name of index with dataset directory
        """
        dir = self.subset_label_dir_pixel_al if use_generated else self.subset_label_dir
        return dir, self.data_list[index]

    def __get_datapoint(self, index, aug, use_generated=False):
        image_name = self.data_list[index]
        image_path = self.subset_image_dir + "/" + image_name + "_" + self.image_name
        label_parent = self.subset_label_dir_pixel_al if use_generated else self.subset_label_dir
        lbl_fn_ext = ".png" if use_generated else ("_" + self.label_name)
        label_path = label_parent + "/" + image_name + lbl_fn_ext
        # print(image_path)
        # print(label_path)
        img_array = self.get_data_from_image(image_path, False)
        mask_array = self.get_data_from_image(label_path, True)
        # mask = self.get_mask(mask_array)

        img_array = tf.image.resize(img_array, (self.IM_SHAPE[0], self.IM_SHAPE[1]))
        mask_array = mask_array if use_generated else self.get_mask(mask_array)
        mask = tf.image.resize(mask_array, (self.GT_SHAPE[0], self.GT_SHAPE[1]), method='nearest')

        img_array = img_array.numpy()
        mask = mask.numpy()
        # print(type(img_array), type(mask))
        # input("HERE")
        # print(img_array.shape, mask_array.shape)
        # print(mask.shape)
        # print(np.unique(mask_array), np.unique(mask))
        if self.op_type == 'train' and aug:
            transformed = self.transform(image=img_array, mask=mask)
            img_array = transformed['image']
            mask = transformed['mask']
        return img_array, mask

    def get_mask(self, raw_mask):
        mask = np.zeros(raw_mask.shape)
        # print(np.unique(raw_mask, return_counts=True))
        for i in range(-1, 34):
            mask[raw_mask == i] = self.id2trainid(i)
        return mask

    def generator(self, use_generated=None, aug=True):
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
                    x_i, y_i = self.__get_datapoint(d, aug=aug, use_generated=use_generated)
                    X[i] = x_i
                    y[i] = y_i
                yield X, y

    def generator_flat(self, weights=False, use_generated=None, aug=True):
        X, y, w, batch_size = self.init_tensors(weights=True)
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
                    x_i, y_i = self.__get_datapoint(d, aug=aug, use_generated=use_generated)
                    X[i] = x_i
                    y_flat = np.reshape(y_i, (y_i.shape[0] * y_i.shape[1], 1))
                    # print(y_i.shape)
                    # print(y_flat.shape)
                    y[i] = y_flat
                    for lid in range(self.NUM_CLASS):
                        # print((y_flat==lid).shape)
                        w[i][y_flat == lid] = self.class_weights[lid]
                if weights:
                    yield X, y, w
                else:
                    yield X, y

    def generator_indices(self, indices, use_generated=None, aug=True):
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
                    x_i, y_i = self.__get_datapoint(indices[d], aug=aug, use_generated=use_generated)
                    X[i] = x_i
                    y[i] = y_i
                yield X, y

    def get_batch_sample(self, index, batch=None, use_generated=None, aug=True):
        X, y, batch_size = self.init_tensors(batch)
        data_indices = list(range(index, index + batch_size))
        for i, d in enumerate(data_indices):
            x_i, y_i = self.__get_datapoint(d, aug=aug, use_generated=use_generated)
            X[i] = x_i
            y[i] = y_i
        return X, y

    def init_tensors(self, batch=None, y_chan=1, weights=False):
        batch_size = CFG.batchSize if batch is None else batch
        X = np.zeros(
            (batch_size, self.IM_SHAPE[0], self.IM_SHAPE[1], 3), dtype=np.float32
        )
        if weights:
            y = np.zeros(
                (batch_size, self.GT_SHAPE[0] * self.GT_SHAPE[1], y_chan), dtype=np.int32
            )
            w = np.zeros(
                (batch_size, self.GT_SHAPE[0] * self.GT_SHAPE[1], y_chan), dtype=np.int32
            )
            return X, y, w, batch_size
        y = np.zeros(
            (batch_size, self.GT_SHAPE[0], self.GT_SHAPE[1], y_chan), dtype=np.int32
        )
        return X, y, batch_size

    def label2color(self, label_int):
        return self.label_color_dict[label_int]

    def lab2name(self, label_int):
        return self.label_name_dict[label_int]

    def color_int_mask(self, mask):
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
        for k in self.label_color_dict:
            # print((mask == k).shape, (np.array(self.label2color(k))).shape)
            color_mask[np.squeeze(mask == k, axis=-1)] = np.array(self.label2color(k))
            # if k in mask:
            #     print(f'Class {k}, Color {self.label2color(k)}, Value {self.lab2name(k)}')
        return color_mask

    def get_sample_sp_map(self, index):
        subdir, image_name = self.get_image_name(index)
        mask_dir = self.subset_label_dir_sp + "/" + image_name + ".txt"
        return np.loadtxt(mask_dir)

    def create_sp_map_folder(self):
        sp_dir = self.subset_label_dir_sp
        for i in self.data_list:
            image_name = i
            mask_dir_split = (sp_dir + "/" + image_name).split("/")
            mask_dir = "/".join(mask_dir_split[:-1])
            print(f"Creating folder {mask_dir}")
            Path(mask_dir).mkdir(exist_ok=True)
