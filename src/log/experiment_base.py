import PIL.Image


class ExperimentBase:
    def __init__(self):
        """
        Initialize a experiment (new or existing)
        """
        raise NotImplementedError("Not implemented.")

    def log_metric(self, log_name: str, value):
        """
        Record a metric to this experiment.

        :param log_name:
          Name of metric (e.g. "loss").

        :param value:
          Datapoint to add. (e.g. 0.2455)

        Implementations like Neptune will automatically generate a graph using all the
        recorded values.
        """
        raise NotImplementedError("Not implemented.")

    def log_image(self, log_name: str, img_1: PIL.Image, description=None):
        """
        Add an image to this experiment.

        :param log_name:
          Name of the image type (e.g. "Training Examples").

        :param img_1:
          Image to upload.

        :param description:
          Description to associated with the image.
        """
        raise NotImplementedError("Not implemented.")
