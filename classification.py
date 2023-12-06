import cv2

def classify_objects(class_index, class_labels):
    recyclable_classes = ['bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl']
    non_recyclable_classes = ['person', 'car', 'dog', 'chair', 'laptop', 'teddy bear']

    class_name = class_labels[class_index - 1]

    if class_name in recyclable_classes:
        return 'Recyclable'
    elif class_name in non_recyclable_classes:
        return 'Non-Recyclable'
    else:
        return 'Unclassified'
