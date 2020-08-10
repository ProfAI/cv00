import cv2
import numpy as np

TARGET_CLASSES = None
MIN_CONFIDENCE = 0.6

SIZE = (320, 320)


def load_yolo(yolo_path=""):
	net = cv2.dnn.readNet(yolo_path+"yolov3.weights", yolo_path+"yolov3.cfg")
	classes = []
	with open(yolo_path+"coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):			
    img = cv2.resize(img, SIZE)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=SIZE)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_boxes(outputs, height, width):

    boxes = []
    class_ids = []
    confs = []

    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if(TARGET_CLASSES!=None and classes[class_id] not in TARGET_CLASSES):
                continue
            
            if(conf > MIN_CONFIDENCE):
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                class_ids.append(class_id)
                confs.append(conf)
    
    return boxes, class_ids, confs


def draw_results(img, boxes, class_ids, confs=None, show_conf=True):

    if(show_conf and confs==None):
        print("Missing confidence array")
        return

    for i in range(len(boxes)):

        box = boxes[i]
        class_id = class_ids[i]

        x, y, w, h = box
        color = colors[class_id]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        
        label = classes[class_id]

        if(show_conf):
            label+=" (%.1f%%)" % (confs[i]*100)

        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1)

    return img
    

net, classes, colors, output_layers = load_yolo()

#input_video_path = input("Insert input video path: ")
#output_video_path = input("Insert output video path: ")

input_video_path = "video_small.mp4"
output_video_path = "video_detected.avi"

cap = cv2.VideoCapture(input_video_path)
ret, frame = cap.read()

if(not ret):
    print("Webcam non disponibile")
    exit(0)

codec = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_video_path, codec, 20.0, (frame.shape[0], frame.shape[1]))

frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 1

while(cap.isOpened()):

    ret, frame = cap.read()

    if(not ret):
        break

    print("Processing frame %d of %d..." % (current_frame, frames_count), end="")

    blob, outputs = detect_objects(frame, net, output_layers)
    boxes, class_ids, confs = get_boxes(outputs, frame.shape[0], frame.shape[1])
    frame = draw_results(frame, boxes, class_ids, confs=confs, show_conf=True)

    print("DONE")

    out.write(frame)
    current_frame+=1

cap.release()
out.release()
cv2.destroyAllWindows()