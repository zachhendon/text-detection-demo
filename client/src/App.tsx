import useWebSocket, { ReadyState } from 'react-use-websocket';
import './App.css'


function App() {
  const socketUrl = 'wss://zachhendon--text-detection-demo-endpoint-dev.modal.run/ws';
  const { sendMessage, lastMessage, readyState } = useWebSocket(socketUrl);

  const connectionStatus = {
    [ReadyState.CONNECTING]: 'Connecting',
    [ReadyState.OPEN]: 'Open',
    [ReadyState.CLOSING]: 'Closing',
    [ReadyState.CLOSED]: 'Closed',
    [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
  }[readyState];

  return (
    <div>
      <button
        onClick={() => sendMessage("Hello")}
        disabled={readyState !== ReadyState.OPEN}
      >
        Click Me to send 'Hello'
      </button>
      <span>The WebSocket is currently {connectionStatus}. </span>
      {lastMessage ? <span>Last message: {lastMessage.data}</span> : null}
    </div>
  );
}

export default App
