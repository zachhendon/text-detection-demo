# Text Detection Demo

Detect arbitrarily-shaped text from your webcam feed in real-time. Developed in PyTorch, the model takes in images and outputs a binary mask that locates text instances.

**[Live Demo](https://text-detection-demo.netlify.app/)**

## Client

The React client connects to the server via a websocket. It extracts JPEG frames from the client's webcam feed and sends them to the server in base64 format. When it receives a response from the server, it displays it to the page.

## Server

The server is developed using [Modal](https://modal.com/) for fast inference with GPUs and containers that automatically scale. The app will automatically scale to 0 instances when not in use with around 20-30s cold start times and scale up to many GPUs when under heavy load. When the server receives image data from a client, it decodes and preprocessed before passing it through the text detection model. It compresses the binary output mask to JPEG format and converts it back to base64 before sending the result to the client.

## Model

The [model](https://github.com/zachhendon/image-translator/tree/main/text-detection) was heavily influenced by [FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation](https://arxiv.org/abs/2111.02394) and [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947). The backbone is comprised of a small CNN with an FPN. The data is preprocessed by applying image augmentations and a morphological erosion operation to the ground truth text instances to avoid overlap. In the postprocessing phase, it utilizes morphological dilation, which is GPU parallelizable, instead of typical clipping algorithms to bring the text instances back to their original size. The small backbone and efficient postprocessing are the major reasons this model has such high throughput. It is pretrained on synthtext and fine-tuned on ICDAR2015. For both datasets, hard example mining and a combination of BCE loss and dice loss is used. 

The model achieves and F-measure of 81.27%. The model is exported and optimized with TensorRT and has ~1.03ms latency on an L4 GPU.