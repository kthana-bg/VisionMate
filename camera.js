import React, { useEffect, useRef, useState } from "react";

export default function Camera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);

  const [data, setData] = useState({});

  // ----------------------------
  // CONNECT WEBSOCKET
  // ----------------------------
  useEffect(() => {
    wsRef.current = new WebSocket("ws://localhost:8000/ws");

    wsRef.current.onmessage = (event) => {
      setData(JSON.parse(event.data));
    };

    return () => wsRef.current.close();
  }, []);

  // ----------------------------
  // START CAMERA
  // ----------------------------
  useEffect(() => {
    async function startCamera() {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });

      videoRef.current.srcObject = stream;
    }

    startCamera();
  }, []);

  // ----------------------------
  // SEND FRAMES
  // ----------------------------
  const sendFrame = () => {
    if (!videoRef.current || !wsRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = 640;
    canvas.height = 480;

    ctx.drawImage(videoRef.current, 0, 0, 640, 480);

    const base64 = canvas.toDataURL("image/jpeg", 0.6);

    if (wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(base64);
    }
  };

  // send frame every 200ms (~5 FPS)
  useEffect(() => {
    const interval = setInterval(sendFrame, 200);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ display: "flex", gap: "20px" }}>
      <div>
        <video ref={videoRef} autoPlay playsInline />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>

      <div>
        <h2>Live Metrics</h2>

        <p>Health: {data.health_score}</p>
        <p>Eye: {data.eye_status}</p>
        <p>Posture: {data.posture_status}</p>
        <p>EAR: {data.ear_value}</p>
        <p>Angle: {data.posture_angle}</p>
      </div>
    </div>
  );
}
