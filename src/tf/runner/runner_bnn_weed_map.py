import re
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
from tf.ds.ds_weed_map import WeedMap
from tf.runner.runner_base_tf import RunnerBaseTf
from bunet.keras_bcnn.models import BayesianUNet2D, MCSampler

from cfg.cfg_main import CFG
from ds.ds_def import DatasetSubset
from neptunecontrib.monitoring.keras import NeptuneMonitor


class RunnerBnnWeedMap(RunnerBaseTf):
    def __init__(self, use_generator=False):
        super().__init__(use_generator=use_generator)
        self.lossFnc = loss.SparseCategoricalCrossentropy()

    def create_input_data(self, subset=DatasetSubset.TRAIN):
        data = WeedMap(subset)
        return data

    def createModel(self):
        if CFG.numWorkers > 1:
            mirrored_strategy = tf.distribute.MirroredStrategy()

            with mirrored_strategy.scope():
                model = self.get_model()
                model.compile(
                    optimizer=self.optimizer, loss=self.lossFnc,
                    metrics=["accuracy", MeanIoU(self.num_class, ignore_class=255, sparse_y_pred=False)])
        else:
            model = self.get_model()
            model.compile(
                optimizer=self.optimizer, loss=self.lossFnc,
                metrics=["accuracy", MeanIoU(self.num_class, ignore_class=255, sparse_y_pred=False)])
        return model

    def get_model(self):
        model = BayesianUNet2D(WeedMap.IM_SHAPE, self.num_class, nfilter=CFG.models.Unet.numFilters,
                               drop_prob=CFG.models.Unet.dropout, batch_norm=True, nlayer=5,
                               drop_center=CFG.models.Unet.dropcenter).build()
        return model

    def log(self, b: int, e: int, data, prediction, target, loss: float):
        super().log(b, e, data, prediction, target, loss)

        img = array_to_img(data[0])
        pred = tf.math.argmax(prediction[0])  # Predicted number in 1st image.

        self.exp.log_image(
            "Image Visualization", img, description=f"Loss: {loss:.3f}. Target: {pred}."
        )

    def train(self):
        data = self.create_input_data(DatasetSubset.TRAIN)
        val_data = self.create_input_data(DatasetSubset.VAL)

        self.model = self.createModel()
        ckpt_path = CFG.checkpointPath()
        if ckpt_path is not None:
            print(f"Using checkpoint: {ckpt_path}")
            self.model.load_weights(ckpt_path)

        pathlib.Path(CFG.checkpointDir()).mkdir(parents=True, exist_ok=True)

        callbacks = [
            ReduceLROnPlateau(factor=0.1, patience=50, min_lr=1e-8, verbose=1),
            ModelCheckpoint(
                CFG.checkpointDir() + "/epoch_{epoch:02d}.ckpt",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
            ),
            EarlyStopping(patience=100),
            NeptuneMonitor(),
        ]
        self.model.compile(
            optimizer=self.optimizer, loss=self.lossFnc, metrics=["accuracy"]
        )
        self.model.summary()

        initial_epoch = 0
        if ckpt_path is not None:
            # If resuming from a checkpoint, get the starting epoch from its filename.
            m = re.search("/(.+)-.+\.ckpt", ckpt_path)
            epoch_str = m.group()[1]
            initial_epoch = int(epoch_str)

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
        self.model.save(CFG.checkpointDir() + "/model_weedmap_save")
        self.save_train_plot(result)

    def test(self):
        model = tf.keras.models.load_model(CFG.checkpointDir() + "/model_weedmap_save")
        mc_iteration = 20
        mc_step = 5
        sampler = MCSampler(model, mc_step, activation=None, reduce_var="none", reduce_mean="none").build_sample_only()
        sampler.summary()
        m = MeanIoU(self.num_class)
        data = self.create_input_data(DatasetSubset.TEST)
        acc_time = 0
        for i in range(len(data)):
            start = time.time()
            x, y = data.get_batch_sample(i, batch=1)
            samples = np.zeros((mc_iteration, *x.shape[1:]))
            for ii in range(mc_iteration // mc_step):
                samples[ii * mc_step: (ii + 1) * mc_step] = sampler.predict(x)[0]
            pred, var = get_mean_var(samples)
            acc_time += time.time() - start
            true_mask = y[0]
            pred_mask = create_mask(np.squeeze(pred, axis=0))
            m.update_state(true_mask, pred_mask)
            if CFG.test_vis:
                cmap_list = ['k', 'g', 'r']
                display([x[0], true_mask, pred_mask],
                        "{}/{}_vis.png".format(CFG.logDir(), i),
                        [None, cmap_list, cmap_list])
        c_iou = class_iou(m.total_cm)
        ave_inference = acc_time / len(data)
        print("Evaluation MIoU: BG {}, Crop {}, Weed {}".format(c_iou[0], c_iou[1], c_iou[2]))
        print("Average Inference Time: {} seconds".format(round(ave_inference, 4)))
        self.exp.log_metric("average_inference", round(ave_inference, 4))
        self.exp.log_metric("test_miou_bg", c_iou[0])
        self.exp.log_metric("test_miou_crop", c_iou[1])
        self.exp.log_metric("test_miou_weed", c_iou[2])
        self.exp.log_metric("test_miou", m.result())
        del sampler
        del model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()
