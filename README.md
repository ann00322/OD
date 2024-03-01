# Object Detection

## Introduction

- 把影片和graphic以[76,112,71]這個陣列為基準，判斷是否為去被區域進行合併
- 把影片跟mask進行bitwise_and，得到影片遮罩位置給yolov8進行判定
- 透過model.track()對車子ID進行追蹤，並畫出其ID
- 以(350,280),(720,280)為起點和終點畫出分隔線，並以此為基點判斷是否有車經過
- 最後把經過的量進行cv2.putText()
