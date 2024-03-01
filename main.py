import cv2
import math
import numpy as np
import os
import torch


from ultralytics import YOLO
# from utils.sort import Sort


def checkDevice():
    # Test cuda availability
    try:
        torch.cuda.is_available()
    except:
        device = 'cpu'
    else:
        device = 'cuda:0'
    finally:
        print('Running on %s' % device)
        return device
    
def checkVideo(videoPath):
    if not os.path.exists(videoPath):
        print('Video not found')
        exit()
    else:
        video = cv2.VideoCapture(videoPath)
        return video



def draw_boxes(img, className, pred, color=(255, 0, 255)):
    for result in pred:
        for box in result.boxes:
            # Get the coordinates of the box
            x1, y1, x2, y2 = box.xyxy[0]
            _id = box.id
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convert to int
            w, h = x2 - x1, y2 - y1
            # Get the confidence score
            conf = math.ceil(box.conf[0] * 100) / 100
            # Get the predicted class label
            cls = className[int(box.cls[0])]
            if (cls == 'car' or cls == 'truck' or cls == 'bus') and conf > 0.3 and _id!=None:
                # Draw the box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(img, '%s' % (cls), (x1+20, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,255,255),2)
                # Draw the label
                cv2.putText(img, '%s' % (str(int(_id[0].item()))), (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img
    
def counter(now_y,pre_y, count):
    if now_y > 350 and pre_y < 350:
        count += 1
    return count




def main(videoPath, modelName):
    device = checkDevice()  # Check device for running the model
    model = YOLO(modelName).to(device)  # Load model
    video = checkVideo(videoPath)  # Load video
    mask = cv2.imread(maskPath)
    graphic = cv2.imread(graphic_path)
    once = True
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20, (1280,720))
    count = 0
    pass_list = []
    classes = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]  # class list for COCO dataset
    
    # Loop
    while True:
        success, frame = video.read()  # Read frame
        if not success:
            break

        if once:
            list_graphic =[]
            for i in range(len(graphic)):
                for j in range(len(graphic[0])):
                    if  graphic[i][j].tolist() != [76,112,71]:
                        list_graphic.append((i,j))
            once = False
        newframe = cv2.bitwise_and(frame, mask)


        # Detect
        results = model.track(newframe,conf=0.2,persist=True,verbose=False) # result list of detections
        for i, j in list_graphic:
            frame[i][j] = graphic[i][j]


        # Draw
        newframe = draw_boxes(frame, classes, results)

        for box in results[0].boxes:
            if box.xyxy[0][1] > 280 and box.id!=None:
                if box.id.tolist()[0] not in pass_list:
                    pass_list.append(box.id.tolist()[0])
                    count +=1

        newframe = cv2.line(newframe,(350,280),(720,280),color=(0,0,255),thickness=2)
        newframe = cv2.putText(newframe, str(count), (graphic.shape[0] + 100, graphic.shape[1] - 360),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
        # Show
        cv2.imshow('frame', newframe)
        out.write(newframe)

        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # close all windows
    cv2.destroyAllWindows()


if __name__ == '__main__': 
    videoPath = 'cars.mp4'
    modelName = 'yolov8n.pt'
    maskPath = 'mask_.png'
    graphic_path = 'graphics.png'
    main(videoPath, modelName)