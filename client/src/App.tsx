import { useRef, useEffect, useState } from "react";
import useWebSocket from "react-use-websocket";
import "./App.css";

interface Dimensions {
  width: number;
  height: number;
}

function App() {
  const [dimensions] = useState<Dimensions>({
    width: 640,
    height: 480,
  });
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const maskRef = useRef<HTMLCanvasElement | null>(null);
  const maskCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const imageRef = useRef<HTMLImageElement>(
    new Image(dimensions.width, dimensions.height)
  );
  const frameIdRef = useRef<number | null>(null);
  const previousTimeRef = useRef<number>(0);
  const intervalRef = useRef<number>(1000 / 30);
  const [isConnected, setIsConnected] = useState<boolean>(false);

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
    setIsConnected(true);

    if (!maskRef.current || !imageRef.current) return;

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

  const { sendMessage } = useWebSocket(socketUrl, {
    share: true,
    onOpen: (): void => {
      if (frameIdRef.current) {
        cancelAnimationFrame(frameIdRef.current);
      }
      frameIdRef.current = requestAnimationFrame(animate);
    },
    onClose: (): void => {
      if (frameIdRef.current) {
        cancelAnimationFrame(frameIdRef.current);
      }
      setIsConnected(false);
    },
    onMessage: updateMaskCanvas,
  });

  useEffect(() => {
    const getCamera = async (): Promise<void> => {
      try {
        const stream: MediaStream = await navigator.mediaDevices.getUserMedia({
          video: { width: dimensions.width, height: dimensions.height },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        if (canvasRef.current) {
          canvasRef.current.width = dimensions.width;
          canvasRef.current.height = dimensions.height;
          canvasCtxRef.current = canvasRef.current.getContext("2d");
        }
        if (maskRef.current) {
          maskRef.current.width = dimensions.width;
          maskRef.current.height = dimensions.height;
          maskCtxRef.current = maskRef.current.getContext("2d");
        }
      } catch {
        alert("Error: no camera found on this device");
        return;
      }
    };
    getCamera();
  }, []);

  return (
    <div>
      <p>
        {!isConnected
          ? "Currently connecting to server. Estimated startup time: 20-30 seconds."
          : `Detecting text at ${Math.round(1000 / intervalRef.current)} fps`}
      </p>
      <video
        ref={videoRef}
        autoPlay
        style={{
          width: `${dimensions.width}px`,
          height: `${dimensions.height}px`,
        }}
      ></video>
      <canvas
        ref={canvasRef}
        style={{
          width: `${dimensions.width}px`,
          height: `${dimensions.height}px`,
          display: "none",
        }}
      />
      <canvas
        ref={maskRef}
        style={{
          width: `${dimensions.width}px`,
          height: `${dimensions.height}px`,
        }}
      />
    </div>
  );
}

export default App;
