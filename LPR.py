import darknet
import cv2
import easyocr

configPath  = "./lp-detection-layout-classification.cfg"
dataPath    = "./lp-detection-layout-classification.data"
namePath    = "./lp-detection-layout-classification.names"
weightPath  = "./lp-detection-layout-classification.weights"

reader = easyocr.Reader(['ko'])

'''
def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def draw_boxes(detections, image, colors):
    import cv2
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image
'''

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def LP_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect

    origin_height, origin_width, _ = image.shape

    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    # image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)

    box = []
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        box.append(int(left * (origin_width / width)))
        box.append(int(top * (origin_height / height)))
        box.append(int(right * (origin_width / width)))
        box.append(int(bottom * (origin_height / height)))
    
    return image, box

def OCR(image):
    ret = reader.readtext(image, detail = 0)
    return ret

def LPR(image):
    network, class_names, class_colors = darknet.load_network(
        configPath,
        dataPath,
        weightPath
    )

    image, box = LP_detection(
        image, network, class_names, class_colors, .5
    )

    LP_img = image[box[1]:box[3], box[0]:box[2]]
    LP_text_list = OCR(LP_img)
    if LP_text_list:
        LP_text = LP_text_list[0]
    else:
        LP_text = ""

    return LP_img, LP_text

def main():
    imagePath = "/home/joohyuk99/Desktop/workspace/LPR/test/Project1-master/darknet/1.jpg"
    img = cv2.imread(imagePath)

    image, text = LPR(img)
    
    '''
    cv2.imshow("detections", detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
