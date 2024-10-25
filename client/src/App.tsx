import { useRef, useEffect } from "react";
import useWebSocket, { ReadyState } from "react-use-websocket";
import "./App.css";

function App() {
  const openWebsocket = (event: WebSocketEventMap["open"]): void => {
    console.log("Started")
    console.log(event);
  }
  const updateMaskCanvas = (event: WebSocketEventMap["message"]): void => {
    if (!canvasCtxRef.current || !image) return;
    const context = canvasCtxRef.current;

    image.src = `data:image/jpg;base64,${event.data.slice(
      2,
      event.data.length - 1
    )}`;
    image.onload = () => {
      context.drawImage(image, 0, 0);
    };
  };

  const captureFrame = (): void => {
    if (!canvasRef.current || !myRef.current || !canvasCtxRef.current) return;

    const canvas = canvasRef.current;
    const video = myRef.current;
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

  const socketUrl =
    "wss://zachhendon--text-detection-demo-endpoint.modal.run/ws";
  const { sendMessage, readyState } = useWebSocket(socketUrl, {
    onOpen: openWebsocket,
    onMessage: updateMaskCanvas,
  });

  const myRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const maskRef = useRef<HTMLCanvasElement | null>(null);
  const image = new Image(640, 480);

  useEffect(() => {
    const getCamera = async (): Promise<void> => {
      try {
        const stream: MediaStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        });
        stream.getVideoTracks();
        if (myRef.current) {
          myRef.current.srcObject = stream;
        }
        if (canvasRef.current && myRef.current) {
          canvasCtxRef.current = canvasRef.current.getContext("2d");
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
        ref={myRef}
        autoPlay
        style={{ width: "640px", height: "480px" }}
      ></video>
      <button onClick={captureFrame}>CLICK</button>
      <canvas ref={canvasRef} style={{ width: "640px", height: "480px" }} />
      <canvas ref={maskRef} style={{ width: "640px", height: "480px" }} />
    </div>
  );
}

export default App;
