from functools import partial

from ..postprocess.detection import (
    prepare_input,
    process_output,
    validate_model_config,
)
from ..postprocess.pose import process_pose_output
from ..session import OnnxSessionMixin, load_onnx_model_config


class Model(OnnxSessionMixin):
    def __init__(self, model_uri, providers=None, accelerator=None):
        model_path, self.config = load_onnx_model_config(model_uri)
        self.task, self.input_shape, self.classes = validate_model_config(
            self.config
        )
        if self.task == "classification":
            raise ValueError(
                "Classification ONNX models must be loaded with ResNetClassifier"
            )
        if self.task not in ("detection", "segmentation", "pose"):
            raise ValueError(
                "Detection ONNX models support detection, segmentation, or pose"
            )
        decoder = process_pose_output if self.task == "pose" else process_output
        self.decoder = partial(decoder, classes=list(self.classes))
        self._init_onnx_session(
            model_path,
            providers=providers,
            accelerator=accelerator,
        )
        self.input_name = self.session.get_inputs()[0].name

    def run(
        self,
        img,
        conf_threshold=0.35,
        iou_threshold=0.7,
        approx=0.001,
        labels=None,
        top_k=1,
        score_threshold=None,
    ):
        self._ensure_open()
        img_size = img.shape[1], img.shape[0]
        input_tensor, scale, padding = prepare_input(
            img, self.input_shape, return_transform=True
        )
        inputs = {self.input_name: input_tensor}
        outputs = self.session.run(None, inputs)
        return self.decoder(
            outputs,
            img_size,
            self.input_shape,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            approx=approx,
            labels=labels,
            transform=(scale, padding),
        )
