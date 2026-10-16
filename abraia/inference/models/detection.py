from functools import partial

from ..postprocess.detection import (
    prepare_input,
    prepare_rtmdet_input,
    process_output,
    process_rtmdet_output,
    validate_model_config,
)
from ..postprocess.pose import process_pose_output
from ..session import OnnxSessionMixin, load_onnx_model_config


class Model(OnnxSessionMixin):
    def __init__(self, model_uri, providers=None, accelerator=None):
        model_path, config = load_onnx_model_config(model_uri)
        self.task, self.input_shape, self.classes = validate_model_config(
            config
        )
        self.backend = str(config.get("backend", "ultralytics")).lower()
        if self.task == "classification":
            raise ValueError(
                "Classification ONNX models must be loaded with ResNetClassifier"
            )
        if self.task not in ("detection", "segmentation", "pose"):
            raise ValueError(
                "Detection ONNX models support detection, segmentation, or pose"
            )
        self._init_onnx_session(
            model_path,
            providers=providers,
            accelerator=accelerator,
        )
        self.input_name = self.session.get_inputs()[0].name
        output_names = [item.name for item in self.session.get_outputs()]
        if self.backend == "rtmdet":
            self.decoder = partial(
                process_rtmdet_output,
                output_names=output_names,
                classes=list(self.classes),
                task=self.task,
            )
        else:
            decoder = process_pose_output if self.task == "pose" else process_output
            self.decoder = partial(decoder, classes=list(self.classes))

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
        if self.backend == "rtmdet":
            input_tensor, scale, padding = prepare_rtmdet_input(
                img,
                self.input_shape,
                return_transform=True,
            )
        else:
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
