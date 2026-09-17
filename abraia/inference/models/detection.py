import os

from ..postprocess.decoders import (
    create_decoder,
    prepare_input,
    validate_model_config,
)
from ..session import OnnxSessionMixin
from ...utils import download_file, load_json


class Model(OnnxSessionMixin):
    def __init__(self, model_uri):
        model_uri = os.fspath(model_uri)
        if os.path.isabs(model_uri) and not os.path.isfile(model_uri):
            raise FileNotFoundError(f"Model file not found: {model_uri}")
        config_uri = f"{os.path.splitext(model_uri)[0]}.json"
        model_path = model_uri if os.path.isfile(model_uri) else download_file(model_uri)
        if os.path.isfile(config_uri):
            config_path = config_uri
        else:
            config_path = download_file(config_uri)
        self.config = load_json(config_path)
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
        self.decoder = create_decoder(self.task, self.classes)
        self._init_onnx_session(model_path)
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
