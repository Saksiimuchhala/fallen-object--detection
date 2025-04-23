# using yolov11 
import cv2
from pathlib import Path
import os
from ultralytics import YOLO

# Load YOLOv5 model (replace with your model path if needed)
model = YOLO(r"D:\Sakshi muchhala\object detection\yolo11n.pt")

# Classes to exclude
exclude_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'stop sign', 'fire hydrant', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
]

# Class name to ID mapping
all_class_names = model.names
exclude_ids = [cls_id for cls_id, name in all_class_names.items() if name in exclude_classes]

# Input path (file or folder)
input_path = Path(r"D:\Sakshi muchhala\object detection\test2.mkv")  # Replace with your actual input path
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

# Get list of video files of any supported format
video_files = []
if input_path.is_dir():
    for ext in video_extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
elif input_path.is_file() and input_path.suffix.lower() in video_extensions:
    video_files = [input_path]
else:
    print("❌ Invalid input path or unsupported file type.")
    exit()

codec_map = {
    '.mp4': 'mp4v',  # Common for mp4 files
    '.avi': 'XVID',  # MPEG-4 for .avi
    '.mov': 'avc1',  # H.264 for .mov
    '.mkv': 'X264',  # H.264 for .mkv
    '.flv': 'FLV1',  # Flash Video
    '.wmv': 'WMV1',  # Windows Media Video
    '.webm': 'VP80', # VP8 for .webm
}

for video_path in video_files:
    cap = cv2.VideoCapture(str(video_path))

    # Prepare output video path
    out_video_path = output_dir / f"{video_path.stem}_filtered.mp4"
    ext = video_path.suffix.lower()  # Get file extension of the current video
    fourcc = cv2.VideoWriter_fourcc(*codec_map.get(ext, 'mp4v'))  
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))

    print(f"Processing: {video_path.name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)[0]  # Get first result from list
        detections = results.boxes

        
        for box in detections:
            cls_id = int(box.cls.item())
            if cls_id not in exclude_ids:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item()
                label = all_class_names[cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

        # Show preview (optional)
        cv2.imshow("Filtered Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

cv2.destroyAllWindows()
print("All videos processed and saved to the output folder.")
