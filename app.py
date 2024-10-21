import modal
import io
import base64
import time


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


@app.cls(image=pytorch_image, volumes={"/models": model_volume}, gpu="L4")
class Model:
    @modal.enter()
    def load_model(self):
        import torch

        self.model = torch.export.load("/models/model.ep").module()
        self.threshold = 0.55
        self.blur_kernel = (
            torch.tensor(
                [[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]], device="cuda"
            )
            / 16
        )

    @modal.method()
    def forward(self, x):
        input_image = Image.open(io.BytesIO(base64.b64decode(x)))
        input = torchvision.transforms.PILToTensor()(input_image)
        input = (input / 255).to(dtype=torch.float32, device="cuda")
        print(input.shape)
        input = torchvision.transforms.Resize((640, 1140))(input)
        print(input.shape)
        input = input.unsqueeze(0)
        out = self.model(input)

        # process and serialize
        blurred = torch.nn.functional.conv2d(
            out.unsqueeze(0),
            self.blur_kernel,
            stride=1,
            padding=1,
        ).squeeze(0)
        binary = (blurred > self.threshold).to(dtype=torch.uint8) * 255
        binary = torchvision.transforms.Resize((640, 480))(binary).permute(0, 2, 1)
        binary_image = torchvision.transforms.ToPILImage()(binary)
        buffer = io.BytesIO()
        binary_image.save(buffer, format="JPEG")
        binary_base64 = base64.b64encode(buffer.getvalue())
        return binary_base64


@app.function(image=websocket_image)
@modal.asgi_app()
def endpoint():
    from fastapi import FastAPI, WebSocket
    import numpy as np

    app = FastAPI()

    @app.websocket("/ws")
    async def websocket_handler(websocket: WebSocket) -> None:
        await websocket.accept()
        while True:
            data = await websocket.receive_text()
            s = time.time()
            out = Model().forward.remote(data)
            print(time.time() - s)
            await websocket.send_text(f"{out}")

    return app
