import pathlib
import re

import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras.losses as loss
import tensorflow.keras.optimizers as op
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model

from cfg.cfg_main import CFG
from ds.ds_def import DatasetSubset
from neptunecontrib.monitoring.keras import NeptuneMonitor
from tc.runner_base import RunnerBase
from tf.func.untils import create_mask, display


class RunnerBaseTf(RunnerBase):
    def __init__(self, use_generator=False):
        super().__init__()

        pathlib.Path(CFG.logDir()).mkdir(parents=True, exist_ok=True)
        self.num_class = 3
        self.optimizer = op.Adam(learning_rate=CFG.lr)
        self.lossFnc = loss.CategoricalCrossentropy()
        self.use_generator = use_generator
        self.model = None

    def create_input_data(self, subset=DatasetSubset.TRAIN):
        raise NotImplementedError("Not implemented.")

    def train(self):
        ckpt_path = CFG.checkpointPath()
        if ckpt_path is not None:
            print(f"Using checkpoint: {ckpt_path}")
            self.model.load_weights(ckpt_path)

        pathlib.Path(CFG.checkpointDir()).mkdir(parents=True, exist_ok=True)

        callbacks = [
            ReduceLROnPlateau(factor=0.1, patience=20, min_lr=1e-8, verbose=1),
            ModelCheckpoint(
                CFG.checkpointDir() + "/epoch={epoch:02d}-val_loss={val_loss:.2f}.hdf5",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
            ),
            NeptuneMonitor(),
        ]
        self.model.compile(
            optimizer=self.optimizer, loss=self.lossFnc, metrics=["accuracy"]
        )
        self.model.summary()

        initial_epoch = 0
        if ckpt_path is not None:
            # If resuming from a checkpoint, get the starting epoch from its filename.
            m = re.search("/(.+)-.+\.hdf5", ckpt_path)
            epoch_str = m.group()[1]
            initial_epoch = int(epoch_str)

        if self.use_generator:
            data = self.create_input_data(DatasetSubset.TRAIN)
            val_data = self.create_input_data(DatasetSubset.VAL)
            train_step = len(data) / CFG.batchSize
            val_step = len(val_data) / CFG.batchSize

            result = self.model.fit(
                x=data.generator(),
                batch_size=CFG.batchSize,
                shuffle=False,
                epochs=CFG.numEpochs,
                steps_per_epoch=train_step,
                callbacks=callbacks,
                validation_data=val_data.generator(),
                validation_steps=val_step,
                initial_epoch=initial_epoch,
            )
        else:
            xs, ys = self.create_input_data(DatasetSubset.TRAIN)
            result = self.model.fit(
                x=xs,
                y=ys,
                batch_size=CFG.batchSize,
                shuffle=False,
                epochs=CFG.numEpochs,
                steps_per_epoch=None,
                callbacks=callbacks,
                # Reserve 10% of training dataset for post-epoch validation.
                validation_split=0.1,
                initial_epoch=initial_epoch,
            )
        self.model.save(CFG.checkpointDir() + "/model_save")
        self.exp.log_artifact(CFG.checkpointDir() + "/model_save")
        self.save_train_plot(result)

    def test(self):
        self.model.compile(
            optimizer=self.optimizer, loss=self.lossFnc, metrics=["accuracy"]
        )
        self.model.summary()

        if CFG.checkpointPath() is None:
            raise AssertionError("test() requires a checkpoint.")
        self.model.load_weights(CFG.checkpointPath())

        print(f"Loaded model state that complete {CFG.numEpochs} epochs.")

        if self.use_generator:
            data = self.create_input_data(DatasetSubset.TEST)
            test_size = len(data) / CFG.batchSize
            eva_loss = self.model.evaluate(
                x=data.generator(),
                steps=test_size,
                callbacks=[NeptuneMonitor()],
            )
            if CFG.test_vis:
                x, y = data.get_batch_sample(0)
                for i, v in enumerate(x):
                    vis = self.model.predict(np.expand_dims(v, axis=0))
                    pred_mask = create_mask(vis)
                    mask = create_mask(y[i])
                    cmap_list = ['k', 'g', 'r']
                    display([v, mask, np.squeeze(pred_mask, axis=0)],
                            "{}/{}_vis.png".format(CFG.logDir(), i),
                            [None, cmap_list, cmap_list])
        else:
            xs, ys = self.create_input_data(DatasetSubset.TEST)
            eva_loss = self.model.evaluate(
                x=xs,
                y=ys,
                steps=None,
                callbacks=[NeptuneMonitor()],
            )

        print("Evaluation Loss, Accuracy: {}".format(eva_loss))

    def createModel(self) -> Model:
        raise NotImplementedError("Not implemented.")

    def log(self, b: int, e: int, data, prediction, target, loss: float):
        print(f"Image #{b}: Loss: {loss}")
        self.exp.log_metric("Loss", loss)

    def save_train_plot(self, result):
        plot = plt.figure()
        plt.title("Learning curve")
        plt.plot(result.history["loss"], label="loss")
        plt.plot(result.history["val_loss"], label="val_loss")
        plt.plot(
            np.argmin(result.history["val_loss"]),
            np.min(result.history["val_loss"]),
            marker="x",
            color="r",
            label="best model",
        )
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.legend()
        plt.savefig(f"{CFG.logDir()}/train_log.png")
        self.exp.log_image("train_log", plot, "train_log")
