import gc
import pathlib
from pathlib import Path
import numpy as np
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import save_img
from bunet.keras_bcnn.models import MCSampler
from bunet.keras_bcnn.prediction_entropy import image_entropy, samples_entropy
from tf.runner.runner_bnn_cityscape import RunnerBnnCityscape
from tf.func.untils import get_mean_var, create_mask, display
from tf.func.sp_utils import get_non_bg_image_uncertain, merge_sp_mask

from cfg.cfg_main import CFG
from ds.ds_def import DatasetSubset
from neptunecontrib.monitoring.keras import NeptuneMonitor


class RunnerBnnCityscapeSpActiveLearn(RunnerBnnCityscape):
    def __init__(self, use_generator=False):
        super().__init__(use_generator=use_generator)
        self.train_full = self.create_input_data(DatasetSubset.TRAIN, op_type='train')
        self.val_full = self.create_input_data(DatasetSubset.TRAIN, op_type='val')
        self.train_full_len = len(self.train_full)
        self.val_full_len = len(self.val_full)
        self.train_remain_list = list(range(self.train_full_len))
        self.val_remain_list = list(range(self.val_full_len))
        self.train_list = []
        self.val_list = []
        self.rank_list_train = []
        self.rank_list_val = []
        self.mc_iteration = 5
        self.init_size = 0.1
        self.step_size = 0.1
        self.gen_percent = {'gen_sp': [], 'total_sp': []}
        self.class_label_pixel_count = np.zeros(self.num_class)
        self.class_label_sp_count = np.zeros(self.num_class)
        self.init_train_len = int(self.train_full_len * self.init_size)
        self.init_val_len = int(self.val_full_len * self.init_size)

    def train_init_batch(self):
        """
        move initial batch from unlabeled pool to training set, train initial model
        Returns:

        """
        print("Training init data! Train {}, Val {}".format(self.init_train_len, self.init_val_len))
        train_indices = list(range(self.init_train_len))
        val_indices = list(range(self.init_val_len))
        self.train_init = train_indices
        self.val_init = val_indices
        self.create_gen_data_folder(train=True)
        self.create_gen_data_folder(train=False)
        self.copy_init_to_generated(train_indices, val_indices)
        self.update_lists(train_indices, val_indices)
        self.model = self.createModel()
        callbacks, initial_epoch = self.init_model()
        self.train_on_list(callbacks, initial_epoch, True)

    def copy_init_to_generated(self, train_list, val_list):
        """copy for train and val"""
        self.copy_list_to_generated(train_list, train=True)
        self.copy_list_to_generated(val_list, train=False)

    def copy_list_to_generated(self, ll, train=False):
        """
        copy original mask to generated folder and save as grayscale
        Args:
            ll: list of index to copy
            train: true if is training

        Returns:

        """
        dd = self.train_full if train else self.val_full
        for i in ll:
            x, y = dd.get_batch_sample(i, batch=1)
            y = y.squeeze(axis=0)
            subdir, image_name = dd.get_image_name(i, use_generated=True)
            mask_dir = subdir + "/" + image_name + ".png"
            save_img(mask_dir, y, scale=False)

    def create_gen_data_folder(self, train=False):
        dd = self.train_full if train else self.val_full
        for i in range(len(dd)):
            subdir, image_name = dd.get_image_name(i, use_generated=True)
            mask_dir_split = (subdir + "/" + image_name).split("/")
            mask_dir = "/".join(mask_dir_split[:-1])
            Path(mask_dir).mkdir(exist_ok=True)

    def train_on_list(self, callbacks, initial_epoch, init_train=False):
        """
        train the model using samples in the training set
        Args:
            callbacks: keras training callbacks
            initial_epoch: start epoch
            init_train: true for initial training, false for active learning step training

        Returns:

        """
        self.exp.log_metric("training size", len(self.train_list) + len(self.val_list))
        train_step = len(self.train_list) / CFG.batchSize
        val_step = len(self.val_list) / CFG.batchSize
        use_generated = not init_train
        self.model.fit(
            x=self.train_full.generator_indices(self.train_list, use_generated=use_generated),
            batch_size=CFG.batchSize,
            epochs=CFG.numEpochs,
            steps_per_epoch=train_step,
            callbacks=callbacks,
            validation_data=self.val_full.generator_indices(self.val_list, use_generated=use_generated),
            validation_steps=val_step,
            initial_epoch=initial_epoch,
        )
        model_name = "/model_cityscape_init" if init_train else "/model_cityscape_save"
        self.model.save(CFG.checkpointDir() + model_name)
        del self.model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

    def init_model(self):
        """
        create callbacks, compile model
        Returns:

        """
        pathlib.Path(CFG.checkpointDir()).mkdir(parents=True, exist_ok=True)
        callbacks = [
            ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-8, verbose=1),
            ModelCheckpoint(
                CFG.checkpointDir() + "/epoch_{epoch:02d}.ckpt",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
            ),
            EarlyStopping(patience=40),
            NeptuneMonitor(),
        ]
        self.model.compile(
            optimizer=self.optimizer, loss=self.lossFnc,
            metrics=["accuracy", MeanIoU(self.num_class, ignore_class=255, sparse_y_pred=False)]
        )
        initial_epoch = 0
        return callbacks, initial_epoch

    def rank_remain_lists(self, step):
        """
        Make prediction of the remaining data in pool, measure entropy and then return the sorted index to add to
        Returns:

        """
        model_name = "/model_cityscape_init" if step == 0 else "/model_cityscape_save"
        model = tf.keras.models.load_model(CFG.checkpointDir() + model_name)
        sampler = MCSampler(model, self.mc_iteration, activation=None, reduce_var="none",
                            reduce_mean="none").build_sample_only()
        print("Getting Uncertainty Train!!")
        e_train = self.get_uncertain_list(sampler, train=True)
        print("Getting Uncertainty Val!!")
        e_val = self.get_uncertain_list(sampler, train=False)
        del sampler
        del model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()
        return e_train, e_val

    def generate_step_mask(self, step, train_list, val_list):
        """
        generate masks for an active learning step
        Args:
            step: step number
            train_list: training batch to generate mask
            val_list: val batch to generate mask

        Returns:

        """
        model_name = "/model_cityscape_init" if step == 0 else "/model_cityscape_save"
        model = tf.keras.models.load_model(CFG.checkpointDir() + model_name)
        sampler = MCSampler(model, self.mc_iteration, activation=None, reduce_var="none",
                            reduce_mean="none").build_sample_only()
        print("Generate Mask Train {} Samples!!".format(len(train_list)))
        self.generate_mask_list(train_list, sampler, step, train=True)
        print("Generate Mask Val {} Samples!!".format(len(val_list)))
        self.generate_mask_list(val_list, sampler, step, train=False)
        del sampler
        del model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

    def update_lists(self, new_train, new_val):
        """
        pop new_train and new_val from unlabeled pool and add them to train/val lists
        Args:
            new_train:
            new_val:

        Returns:

        """
        self.pop_list(self.train_remain_list, self.train_list, new_train)
        self.pop_list(self.val_remain_list, self.val_list, new_val)

    @staticmethod
    def pop_list(pop_from, append_to, i_pop):
        """
        pop list i_pop from pop_from and append it to append_to
        Args:
            pop_from: list to pop from
            append_to: list to append to
            i_pop: list of indices to be popped

        Returns:

        """
        l_pop = deepcopy(i_pop)
        for i, v in enumerate(l_pop):
            t = pop_from.pop(v)
            append_to.append(t)
            for ii, k in enumerate(l_pop[i + 1:]):
                if k > v:
                    l_pop[i + 1 + ii] -= 1

    def get_uncertain_list(self, sampler, train=True):
        """
        calculate the uncertainty value for each entry in the unlabeled pool
        Args:
            sampler: mc sample instance
            train: true for training set, false for val set

        Returns: list containing uncertainty value for each entry in unlabeled pool

        """
        e_list = []
        ll = self.train_remain_list if train else self.val_remain_list
        dd = self.train_full if train else self.val_full
        for i in ll:
            self.get_sample_uncertainty(dd, e_list, i, sampler)
        return e_list

    def generate_mask_list(self, ll, sampler, step, train=True):
        """
        ll is corresponding indecis in train/val_remain_list
        """
        pathlib.Path("{}/step{}".format(CFG.logDir(), step)).mkdir(exist_ok=True)
        dd = self.train_full if train else self.val_full
        r_list = self.train_remain_list if train else self.val_remain_list
        gen_list = [r_list[x] for x in ll]
        if CFG.activeLearn.regen_mask:
            re_list = self.train_list[self.init_train_len:] if train else self.val_list[self.init_val_len:]
            gen_list += re_list
        for i in gen_list:
            pred, y, uncertain_matrix, x = self.get_sample_uncertainty(dd, [], i, sampler)
            self.generate_sample_mask(dd, i, pred, y, uncertain_matrix, step, x)

    def generate_sample_mask(self, data_generator, index, pred, y, uncertain_matrix, step, x):
        true_mask = y[0]
        pred_mask = create_mask(np.squeeze(pred, axis=0))
        sp_map = data_generator.get_sample_sp_map(index)
        run_in_cluster_mode = CFG.activeLearn.pixel_mode == 1  # 1 in cluster mode, 0 in percentage mode
        gen_mask, query_mask = merge_sp_mask(true_mask, pred_mask, uncertain_matrix, sp_map,
                                             gen_percent=self.gen_percent, use_ignore=CFG.activeLearn.use_ignore,
                                             use_clustering=run_in_cluster_mode,
                                             sp_keep_percent=CFG.activeLearn.pixel_percent,
                                             number_cluster=CFG.activeLearn.num_cluster,
                                             class_pixel_count=self.class_label_pixel_count,
                                             class_sp_count=self.class_label_sp_count, )
        subdir, image_name = data_generator.get_image_name(index, use_generated=True)
        mask_dir = subdir + "/" + image_name + ".png"
        save_img(mask_dir, gen_mask, scale=False)

        sample_sp_uncertain = round(get_non_bg_image_uncertain(uncertain_matrix, pred_mask, include_first_class=True),
                                    ndigits=4)
        display([x[0], data_generator.color_int_mask(true_mask), data_generator.color_int_mask(gen_mask), query_mask],
                "{}/step{}/{}_{}.png".format(CFG.logDir(), step, image_name.split('/')[-1], sample_sp_uncertain),
                [None, None, None, ['w', 'k']],
                use_title=["RGB", "Manual Label", "Active Learning Label", "Query Mask"],
                entropy=np.expand_dims(uncertain_matrix, axis=-1))

    def get_sample_uncertainty(self, data_generator, e_list, index, sampler):
        x, y = data_generator.get_batch_sample(index, batch=1, aug=False)
        samples = sampler.predict(x)[0]
        pred, var = get_mean_var(samples)
        if CFG.activeLearn.mode == 3:
            uncertain_matrix = var
        else:
            entropy = image_entropy(np.squeeze(pred, axis=0))
            if CFG.activeLearn.mode == 2:
                entropy += samples_entropy(samples)
            uncertain_matrix = entropy
        sample_sp_uncertain = get_non_bg_image_uncertain(uncertain_matrix, create_mask(np.squeeze(pred, axis=0)),
                                                         include_first_class=True)
        e_list.append(sample_sp_uncertain)
        return pred, y, uncertain_matrix, x

    def run_active_learn_step(self, step):
        """
        rank, update, train again
        Returns:

        """
        train_al_step = int(self.train_full_len * self.step_size)
        val_al_step = int(self.val_full_len * self.step_size)
        self.class_label_pixel_count = np.zeros(self.num_class)
        self.class_label_sp_count = np.zeros(self.num_class)

        if step == 0 or CFG.activeLearn.rerank_step:
            train_remain_rank, val_remain_rank = self.rank_remain_lists(step=step)
            self.rank_list_train = train_remain_rank
            self.rank_list_val = val_remain_rank

        rank_arg_train = np.argsort(self.rank_list_train)[-train_al_step:]
        rank_arg_val = np.argsort(self.rank_list_val)[-val_al_step:]
        self.generate_step_mask(step, rank_arg_train, rank_arg_val)
        self.update_lists(rank_arg_train, rank_arg_val)
        self.pop_list(self.rank_list_train, [], rank_arg_train)
        self.pop_list(self.rank_list_val, [], rank_arg_val)

        model_name = "/model_cityscape_init" if step == 0 else "/model_cityscape_save"
        self.model = tf.keras.models.load_model(CFG.checkpointDir() + model_name)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=CFG.lr)
        callbacks, initial_epoch = self.init_model()
        self.train_on_list(callbacks, initial_epoch)
        self.exp.log_metric("percent_sp_gen", np.sum(self.gen_percent['gen_sp']) / np.sum(self.gen_percent['total_sp']))
        for i in range(self.num_class):
            self.exp.log_metric(f"pixel_count_{self.train_full.lab2name(i)}",
                                self.class_label_pixel_count[i])
            self.exp.log_metric(f"sp_count_{self.train_full.lab2name(i)}", self.class_label_sp_count[i])
        np.savetxt(f"{CFG.logDir()}/step{step}/step_{step}_gen_count.csv",
                   np.array([self.gen_percent['gen_sp'], self.gen_percent['total_sp']]), delimiter=",", fmt='%s')

    def train(self):
        """
        active learning loop
        Returns:

        """
        self.train_init_batch()
        for count in range(6):
            print("Start active learn step {}!!".format(count))
            self.run_active_learn_step(count)
            self.test()
