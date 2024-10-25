import { useRef, useEffect, useState } from "react";
import useWebSocket, { ReadyState } from "react-use-websocket";
import "./App.css";

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const imageRef = useRef<HTMLImageElement>(new Image(640, 480));
  const maskRef = useRef<HTMLCanvasElement | null>(null);
  const maskCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const frameIdRef = useRef<number | null>(null);
  const previousTimeRef = useRef<number>(0);
  const intervalRef = useRef<number>(1000 / 30);

  const socketUrl =
    "wss://zachhendon--text-detection-demo-endpoint.modal.run/ws";

  const animate = (currentTime: number): void => {
    const deltaTime = currentTime - previousTimeRef.current;

    if (deltaTime >= intervalRef.current) {
      captureFrame();
      previousTimeRef.current = currentTime;
    }

    frameIdRef.current = requestAnimationFrame(animate);
  };

  const updateMaskCanvas = (event: WebSocketEventMap["message"]): void => {
    if (!maskRef.current ||!imageRef.current) return;

    imageRef.current.src = `data:image/jpg;base64,${event.data.slice(
      2,
      event.data.length - 1
    )}`;
    imageRef.current.onload = () => {
      if (!maskCtxRef.current) return;
      maskCtxRef.current.drawImage(imageRef.current, 0, 0);
    };
  };

  const captureFrame = (): void => {
    if (!canvasRef.current || !videoRef.current || !canvasCtxRef.current)
      return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvasCtxRef.current;

    const width: number = video.videoWidth;
    const height: number = video.videoHeight;
    canvas.width = width;
    canvas.height = height;

    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob: Blob | null) => {
      if (!blob) return;

      const reader: FileReader = new FileReader();

      reader.readAsDataURL(blob);
      reader.onload = () => {
        if (reader.result && typeof reader.result == "string") {
          const base64Data: string = reader.result.split(",")[1];
          sendMessage(base64Data);
        }
      };
    }, "image/jpeg");
  };

  const { sendMessage, readyState } = useWebSocket(socketUrl, {
    share: true,
    onOpen: (): void => {
      if (frameIdRef.current) {
        cancelAnimationFrame(frameIdRef.current);``
      }
      frameIdRef.current = requestAnimationFrame(animate);
    },
    onMessage: updateMaskCanvas,
  });

  useEffect(() => {
    const getCamera = async (): Promise<void> => {
      try {
        const stream: MediaStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        });
        stream.getVideoTracks();
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        if (canvasRef.current && videoRef.current) {
          canvasCtxRef.current = canvasRef.current.getContext("2d");
        }
        if (maskRef.current) {
          maskRef.current.width = 640;
          maskRef.current.height = 480;
          maskCtxRef.current = maskRef.current.getContext("2d");
        }
      } catch (vent) {
        console.log("Error: no camera found on this device");
      }
    };
    getCamera();
  }, []);

  const connectionStatus = {
    [ReadyState.CONNECTING]: "Connecting",
    [ReadyState.OPEN]: "Open",
    [ReadyState.CLOSING]: "Closing",
    [ReadyState.CLOSED]: "Closed",
    [ReadyState.UNINSTANTIATED]: "Uninstantiated",
  }[readyState];

  return (
    <div>
      <p>The WebSocket is currently {connectionStatus}</p>
      <video
        ref={videoRef}
        autoPlay
        style={{ width: "640px", height: "480px" }}
      ></video>
      <button onClick={captureFrame}>CLICK</button>
      <canvas ref={canvasRef} style={{ width: "640px", height: "480px", display: "none" }} />
      <canvas ref={maskRef} style={{ width: "640px", height: "480px" }} />
    </div>
  );
}

export default App;
