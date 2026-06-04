from argparse import Namespace

from rfdetr.deploy.export import trtexec

args = Namespace(
    verbose=True,
    profile=False,
    dry_run=False,
)

trtexec("object_detection/inference_model.onnx", args)