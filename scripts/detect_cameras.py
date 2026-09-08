import cv2

def detect_cameras():
    print("Detecting cameras...")
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"Camera {i}: {w}x{h} @ {fps}fps")
            cap.release()

if __name__ == "__main__":
    detect_cameras()
