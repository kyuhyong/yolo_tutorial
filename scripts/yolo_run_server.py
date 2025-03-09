#!/usr/bin/env python
import torch
from ultralytics import YOLO
import cv2

import asyncio
import websockets
import numpy as np
import base64
import os
import argparse

def check_file_extension(filename):
    """
    Check if the file has a valid extension (.pt or .engine)
    """
    _, ext = os.path.splitext(filename)
    if ext.lower() not in ['.pt', '.engine']:
        raise ValueError("Invalid file extension. Use a .pt or .engine file.")
    return filename

def load_model(model_path):
    """
    Load the YOLO model based on the file extension
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    return model

async def stream(websocket, path, model):
    cap = cv2.VideoCapture(0)
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return

    try:
        while True:
            # Capture a frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Convert the frame from BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = model.predict(image_rgb)
            # Get the image with predictions drawn
            annotated_image = results[0].plot()  # This method draws the boxes and labels on the image
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # Send frame over WebSocket
            await websocket.send(jpg_as_text)

            # Limit frame rate
            #await asyncio.sleep(0.03)  # ~30 FPS

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected. Waiting for new connection...")
    
    except KeyboardInterrupt:
        # Stop the video stream when the user interrupts (Ctrl+C)
        print("Video stream stopped.")

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Release the camera and close all OpenCV windows
        cap.release()

async def main(model):
    async with websockets.serve(lambda ws, path: stream(ws, path, model), "0.0.0.0", 9999):
        print("Server started on port 9999...")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Model WebSocket Streamer")
    parser.add_argument("filename", type=check_file_extension, help="Path to the YOLO model file (.pt or .engine)")
    args = parser.parse_args()
    
    # Load the YOLO model using the provided filename
    model = load_model(args.filename)

    while True:
        try:
            asyncio.run(main(model))
        except Exception as e:
            print(f"Server error: {e}")
            print("Restarting server...")
            continue
