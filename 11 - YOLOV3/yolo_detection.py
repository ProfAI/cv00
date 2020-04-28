import cv2
import numpy as np

IMG_PATH = "pic.jpg"
SIZE = (320, 320)


def load_yolo():
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("coco.names", "r") as f:
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


def get_bboxes(outputs, height, width):

    boxes = []
    class_ids = []
    confs = []

    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            #if(conf > 0.3 and classes[class_id]=="bicycle"):
            if(conf > 0.3):
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


def show_labels(img, boxes, class_ids, confs=None, show_conf=True):

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
            #print(confs)
            label+=" (%.1f%%)" % (confs[i]*100)

        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1)


    cv2.imshow("img", img)
    cv2.waitKey(0)
    

net, classes, colors, output_layers = load_yolo()
img = cv2.imread(IMG_PATH)
img_small = cv2.resize(img, SIZE)
blob, outputs = detect_objects(img, net, output_layers)
boxes, class_ids, confs = get_bboxes(outputs, img.shape[0], img.shape[1])
show_labels(img, boxes, class_ids, confs=confs, show_conf=True)