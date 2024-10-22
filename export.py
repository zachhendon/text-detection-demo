from fast import FAST
import modal
import os

pytorch_image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.09-py3")
mounts = modal.Mount.from_local_file("model.pth", remote_path="/root/model.pth")
volume = modal.Volume.from_name("model-export", create_if_missing=True)
app = modal.App("model-export")

@app.function(image=pytorch_image, mounts=[mounts], volumes={"/models": volume}, gpu="L4")
def export():
    print(os.listdir("."))
    import torch
    import torch_tensorrt

    model = FAST().eval().cuda()
    checkpoint = torch.load("model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    inputs = [torch.randn(1, 3, 640, 1140).cuda()]

    trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
    torch_tensorrt.save(trt_gm, "/models/model.ep", inputs=inputs)
    print("exported")
    torch.export.load("/models/model.ep").module()
    print("successful model import")
    

@app.local_entrypoint()
def main():
    export.remote()
