import gc
import time
import pathlib
import numpy as np
import tensorflow as tf
from tf.func.untils import get_mean_var, class_iou, create_mask, display
import tensorflow.keras.losses as loss
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tf.ds.ds_cityscapes import CityScapes
from tf.runner.runner_base_tf import RunnerBaseTf
from bunet.keras_bcnn.models import MCSampler
from deeplab.model import Deeplabv3

from cfg.cfg_main import CFG
from ds.ds_def import DatasetSubset
from neptunecontrib.monitoring.keras import NeptuneMonitor


class RunnerBnnCityscape(RunnerBaseTf):
    def __init__(self, use_generator=False):
        super().__init__(use_generator=use_generator)
        self.lossFnc = loss.SparseCategoricalCrossentropy(ignore_class=255)
        self.num_class = 19

    def create_input_data(self, subset=DatasetSubset.TRAIN, op_type=None):
        data = CityScapes(subset, op_type=op_type)
        return data

    def createModel(self):
        if CFG.numWorkers > 1:
            print(f"Using Mirrored Strategy, Workers: {CFG.numWorkers}")
            mirrored_strategy = tf.distribute.MirroredStrategy()

            with mirrored_strategy.scope():
                model = self.get_model()
                model.compile(
                    optimizer=self.optimizer, loss=self.lossFnc,
                    metrics=["accuracy", MeanIoU(self.num_class, ignore_class=255, sparse_y_pred=False)]
                )
        else:
            model = self.get_model()
            model.compile(
                optimizer=self.optimizer, loss=self.lossFnc,
                metrics=["accuracy", MeanIoU(self.num_class, ignore_class=255, sparse_y_pred=False)]
            )
        return model

    def get_model(self):
        model = Deeplabv3(input_shape=CityScapes.IM_SHAPE,
                          classes=CityScapes.NUM_CLASS,
                          backbone='xception',
                          activation='softmax', dropout=CFG.models.Deeplab.dropout)
        return model

    def log(self, b: int, e: int, data, prediction, target, loss: float):
        super().log(b, e, data, prediction, target, loss)

        img = array_to_img(data[0])
        pred = tf.math.argmax(prediction[0])  # Predicted number in 1st image.

        self.exp.log_image(
            "Image Visualization", img, description=f"Loss: {loss:.3f}. Target: {pred}."
        )

    def train(self):
        data = self.create_input_data(DatasetSubset.TRAIN, op_type='train')
        val_data = self.create_input_data(DatasetSubset.TRAIN, op_type='val')

        self.model = self.createModel()
        pathlib.Path(CFG.checkpointDir()).mkdir(parents=True, exist_ok=True)

        self.model.summary()

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

        initial_epoch = 0

        train_step = len(data) / CFG.batchSize
        val_step = len(val_data) / CFG.batchSize
        result = self.model.fit(
            x=data.generator(),
            batch_size=CFG.batchSize,
            shuffle=True,
            epochs=CFG.numEpochs,
            steps_per_epoch=train_step,
            callbacks=callbacks,
            validation_data=val_data.generator(),
            validation_steps=val_step,
            initial_epoch=initial_epoch,
        )
        self.model.save(CFG.checkpointDir() + "/model_cityscape_save")
        self.save_train_plot(result)

    def test(self):
        model = tf.keras.models.load_model(CFG.checkpointDir() + "/model_cityscape_save")
        mc_iteration = 5
        sampler = MCSampler(model, mc_iteration, activation=None, reduce_var="none",
                            reduce_mean="none").build_sample_only()
        sampler.summary()
        m = MeanIoU(self.num_class, ignore_class=255)
        data = self.create_input_data(DatasetSubset.VAL, op_type='test')
        acc_time = 0
        for i in range(len(data)):
            start = time.time()
            x, y = data.get_batch_sample(i, batch=1)
            samples = sampler.predict(x)[0]
            pred, var = get_mean_var(samples)
            acc_time += time.time() - start
            true_mask = y[0]
            pred_mask = create_mask(np.squeeze(pred, axis=0))
            m.update_state(true_mask, pred_mask)
            if CFG.test_vis:
                display([x[0], data.color_int_mask(true_mask), data.color_int_mask(pred_mask)],
                        "{}/{}_vis.png".format(CFG.logDir(), i),
                        [None, None, None])
        c_iou = class_iou(m.total_cm)
        ave_inference = acc_time / len(data)
        for i in range(self.num_class):
            print(f"Class: {data.lab2name(i)}, IOU: {c_iou[i]}")
            self.exp.log_metric(f"test_iou_{data.lab2name(i)}", c_iou[i])
        print("Average Inference Time: {} seconds".format(round(ave_inference, 4)))
        self.exp.log_metric("average_inference", round(ave_inference, 4))
        self.exp.log_metric("test_miou", np.mean(c_iou))
        del sampler
        del model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()
