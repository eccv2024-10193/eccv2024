import gc
import random
import pathlib
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tf.runner.runner_bnn_cityscape import RunnerBnnCityscape

from cfg.cfg_main import CFG
from ds.ds_def import DatasetSubset
from neptunecontrib.monitoring.keras import NeptuneMonitor


class RunnerBnnCityscapeRandom(RunnerBnnCityscape):
    def __init__(self):
        super().__init__()
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
        self.mc_iteration = 20
        self.mc_step = 5
        self.init_size = 0.1
        self.step_size = 0.1

    def train_init_batch(self):
        """
        move initial batch from unlabeled pool to training set, train initial model
        Returns:

        """
        init_train_len = int(self.train_full_len * self.init_size)
        init_val_len = int(self.val_full_len * self.init_size)
        print("Training init data! Train {}, Val {}".format(init_train_len, init_val_len))
        train_indices = list(range(init_train_len))
        val_indices = list(range(init_val_len))
        print(init_train_len, init_val_len)
        self.update_lists(train_indices, val_indices)
        self.model = self.createModel()
        callbacks, initial_epoch = self.init_model()
        self.train_on_list(callbacks, initial_epoch, True)

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
        self.model.fit(
            x=self.train_full.generator_indices(self.train_list),
            batch_size=CFG.batchSize,
            epochs=CFG.numEpochs,
            steps_per_epoch=train_step,
            callbacks=callbacks,
            validation_data=self.val_full.generator_indices(self.val_list),
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

    def run_active_learn_step(self, step):
        """
        rank, update, train again
        Returns:

        """
        train_al_step = int(self.train_full_len * self.step_size)
        val_al_step = int(self.val_full_len * self.step_size)

        # random baseline
        self.update_lists(random.sample(range(len(self.train_remain_list)), train_al_step),
                          random.sample(range(len(self.val_remain_list)), val_al_step))

        model_name = "/model_cityscape_init" if step == 0 else "/model_cityscape_save"
        print(f"Loading model: {model_name}")
        self.model = tf.keras.models.load_model(CFG.checkpointDir() + model_name)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=CFG.lr)
        callbacks, initial_epoch = self.init_model()
        self.train_on_list(callbacks, initial_epoch)

    def train(self):
        """
        active learning loop
        Returns:

        """
        self.train_init_batch()
        for count in range(9):
            print("Start active learn step {}!!".format(count))
            self.run_active_learn_step(count)
            self.test()
