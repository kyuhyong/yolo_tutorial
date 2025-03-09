#!/usr/bin/env python
# server.py
import cv2
import asyncio
import websockets
import numpy as np
import base64

async def stream(websocket, path):
    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))
            if not ret:
                break

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # Send frame over WebSocket
            await websocket.send(jpg_as_text)

            # Limit frame rate
            await asyncio.sleep(0.03)  # ~30 FPS
    
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected. Waiting for new connection...")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()

async def main():
    async with websockets.serve(stream, "0.0.0.0", 9999):
        print("Server started on port 9999...")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
        except Exception as e:
            print(f"Server error: {e}")
            print("Restarting server...")
            continue
