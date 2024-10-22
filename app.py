import modal
import io
import base64
import time
import sys


app = modal.App("text-detection-demo")
model_volume = modal.Volume.from_name("model-export")

pytorch_image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.09-py3")
pytorch_image.pip_install("pillow")
with pytorch_image.imports():
    import torch
    import torchvision
    import torch_tensorrt
    import numpy as np
websocket_image = modal.Image.debian_slim().pip_install("pillow")
with websocket_image.imports():
    from PIL import Image


@app.function(image=pytorch_image, volumes={"/models": model_volume}, gpu="A100")
@modal.asgi_app()
def endpoint():
    from fastapi import FastAPI, WebSocket
    import numpy as np

    ws_app = FastAPI()

    model = torch.export.load("/models/model.ep").module()
    threshold = 0.55
    blur_kernel = (
        torch.tensor(
            [[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]], device="cuda"
        )
        / 16
    )

    @ws_app.websocket("/ws")
    async def websocket_handler(websocket: WebSocket) -> None:
        await websocket.accept()
        while True:
            data = await websocket.receive_text()
            out = forward(data)
            await websocket.send_text(f"{out}")

    def forward(x):
        # convert from base64 to pytorch tensor
        torch_bytes = torch.frombuffer(base64.b64decode(x), dtype=torch.uint8)
        torch_img = (torchvision.io.decode_jpeg(torch_bytes) / 255).cuda()
        torch_img = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((640, 1140)),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )(torch_img).unsqueeze(0)

        # forward passs
        out = model(torch_img)

        # process and serialize
        blurred = torch.nn.functional.conv2d(
            out.unsqueeze(0),
            blur_kernel,
            stride=1,
            padding=1,
        ).squeeze(0)
        binary = (blurred > threshold).to(dtype=torch.uint8) * 255
        binary = torchvision.transforms.Resize((640, 480))(binary).permute(0, 2, 1)
        binary_image = torchvision.transforms.ToPILImage()(binary)
        buffer = io.BytesIO()
        binary_image.save(buffer, format="JPEG")
        binary_base64 = base64.b64encode(buffer.getvalue())
        return binary_base64

    return ws_app
