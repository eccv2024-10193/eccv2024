import tensorflow as tf

devices = tf.config.experimental.list_physical_devices('GPU')

tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

import tf.runner.runner_bnn_weed_map
import tf.runner.runner_bnn_weed_map_random
import tf.runner.runner_bnn_weed_map_sp_active_learn
import tf.runner.runner_bnn_cityscape
import tf.runner.runner_bnn_cityscape_random
import tf.runner.runner_bnn_cityscape_sp_active_learn
from cfg.cfg_main import CFG
from ds.ds_def import DatasetSubset

if __name__ == "__main__":
    # Select a runner.

    if CFG.runner == "tf_bnn_weed_map":
        runner = tf.runner.runner_bnn_weed_map.RunnerBnnWeedMap()
    elif CFG.runner == "tf_bnn_weed_map_random":
        runner = tf.runner.runner_bnn_weed_map_random.RunnerBnnWeedMapRandom()
    elif CFG.runner == "tf_bnn_weed_map_sp_active_learn":
        runner = tf.runner.runner_bnn_weed_map_sp_active_learn.RunnerBnnWeedMapSpActiveLearn()
    elif CFG.runner == "tf_bnn_cityscape":
        runner = tf.runner.runner_bnn_cityscape.RunnerBnnCityscape()
    elif CFG.runner == "tf_bnn_cityscape_random":
        runner = tf.runner.runner_bnn_cityscape_random.RunnerBnnCityscapeRandom()
    elif CFG.runner == "tf_bnn_cityscape_sp_active_learn":
        runner = tf.runner.runner_bnn_cityscape_sp_active_learn.RunnerBnnCityscapeSpActiveLearn()
    else:
        raise AssertionError(f"Runner '{CFG.runner}' is unrecognized.")

    # Execute runner.

    if CFG.mode == DatasetSubset.TRAIN:
        runner.train()
    elif CFG.mode == DatasetSubset.TEST:
        runner.test()
    else:
        raise AssertionError(f"Unsupported runner mode '{CFG.mode}'")
