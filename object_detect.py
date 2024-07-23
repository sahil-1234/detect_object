import cv2
from classification import classify_objects

def perform_object_detection(cap):
    config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = 'frozen_inference_graph.pb'

    model = cv2.dnn_DetectionModel(frozen_model, config_file)

    class_labels = []
    file_name = 'labels.txt'
    with open(file_name, 'rt') as fpt:
        class_labels = fpt.read().rstrip('\n').split('\n')

    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127, 5, 127.5))

    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN

    while True:
        ret, frame = cap.read()
        ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

        object_count = len(ClassIndex)
        print(f"Number of objects detected: {object_count}")

        if object_count != 0:
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                if ClassInd <= 80:
                    cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                    cv2.putText(frame, class_labels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                                fontScale=font_scale, color=(0, 255, 0), thickness=3)

                  
                    classification = classify_objects(ClassInd, class_labels)
                    print(f"Object: {class_labels[ClassInd - 1]}, Classification: {classification}")

        cv2.putText(frame, f"Objects: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Obj detection', frame)

        if cv2.waitKey(2) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()
