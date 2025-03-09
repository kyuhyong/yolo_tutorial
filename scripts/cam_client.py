#!/usr/bin/env python
# client.py
import cv2
import asyncio
import websockets
import numpy as np
import base64
import argparse

def parse_ip_port(ip_port):
    """
    Parse the IP:Port argument and return them as a tuple
    """
    try:
        ip, port = ip_port.split(":")
        port = int(port)  # Convert port to integer
        return ip, port
    except ValueError:
        raise argparse.ArgumentTypeError("IP:Port must be in the format xxx.xxx.xxx.xxx:yyyy")

async def receive_stream(ip, port):
    uri = "ws://"+ip+":"+str(port)
    print("Trying to connec to: "+uri)

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected to the server...")

                while True:
                    try:
                        # Receive frame data
                        frame_data = await websocket.recv()
                        jpg_original = base64.b64decode(frame_data)

                        # Convert to numpy array and decode
                        np_arr = np.frombuffer(jpg_original, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        
                        # Fix: Convert RGB to BGR for proper color display
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        # Display the frame
                        cv2.imshow("Webcam Stream", frame_bgr)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed by server.")
                        break
        
        except (OSError, 
                asyncio.TimeoutError, 
                websockets.exceptions.InvalidURI) as e:
            print(f"Failed to connect to server: {e}")
            print("Retrying in 5 seconds...")
            await asyncio.sleep(5)
            continue

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Camera Client with IP and Port")
    parser.add_argument("ip_port", type=parse_ip_port, help="Server IP and Port (e.g., ###.###.###.###:PORT)")
    
    # Parse the arguments
    args = parser.parse_args()
    ip, port = args.ip_port

    asyncio.run(receive_stream(ip, port))
