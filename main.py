from ultralytics import YOLO
import cv2
import tkinter as tk
import os
def main():
    myroot = tk.Tk()
    
    model = YOLO("YOLOv8Checkpoint/YOLOv8Checkpoint/train4/weights/best.pt")
    video_path = 0
    cap = cv2.VideoCapture(video_path)

    ret = True
    frame_count = 0
    save_dir = "save_crop"
    #os.makedirs(save_dir, exist_ok=True)
    while ret:
        ret, frame = cap.read()
        frame_count += 1
        if ret:
            results = model.track(frame, persist = True)
            names = model.names
            for i, det in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, det[:4])
                crop = frame[y1:y2, x1:x2]
                #crop_name = f"frame_{frame_count}_ID_{results[0].boxes.id[i]}.jpg"
                #crop_path = os.path.join(save_dir, crop_name)
                #cv2.imwrite(crop_path, crop)
                for r in results:
                    for c in r.boxes.cls:
                        print("Bounding Box Coordinates of class", names[int(c)],"is : (", x1,",", y1 ,")","(",x2, y2,  ")")
        _frame = results[0].plot()
        cv2.imshow('frame', _frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()