import modal
import json
import io
import pickle
import base64
import time
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi import FastAPI, WebSocket


app = modal.App("text-detection-demo")
# fast_app = FastAPI()
model_volume = modal.Volume.from_name("model-export")

pytorch_image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.09-py3")
pytorch_image.pip_install("fastapi", "websockets")
with pytorch_image.imports():
    import torch
    import torchvision
    import torch_tensorrt


@app.cls(image=pytorch_image, volumes={"/models": model_volume}, gpu="L4")
class Model:
    @modal.enter()
    def load_model(self):
        import torch

        self.model = torch.export.load("/models/model.ep").module()

    @modal.method()
    def forward(self, x):
        s = time.time()
        inp = torch.from_numpy(x).to(dtype=torch.float32, device="cuda")
        # inp = torch.randn(1, 3, 640, 1140).cuda()
        print(time.time() - s)
        out = self.model(inp)
        print(time.time() - s)
        print()
        return out.sum().item()

    # @modal.method()
    # def inference(self):
    #     out = self.forward.remote(None)
        # image = torchvision.transforms.ToPILImage()(out)
        # return_image = io.BytesIO()
        # image.save(return_image, "JPEG")
        # return_image.seek(0)
        # return StreamingResponse(content=return_image, media_type="image/jpeg")

@app.function()
@modal.asgi_app()
def endpoint():
    from fastapi import FastAPI, WebSocket
    import numpy as np
    
    app = FastAPI()

    @app.websocket("/ws")
    async def websocket_handler(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("WORKING")
        while True:
            data = await websocket.receive_text()

            x = np.random.randn(1, 3, 640, 1140)
            out = Model().forward.remote(x)
            await websocket.send_text(f"{data}: {out}")
    return app
