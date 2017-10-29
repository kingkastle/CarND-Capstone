
import tensorflow as tf
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

PRINT_IMAGES = True
USE_DNN_CLASSIFIER = False

def load_detection_graph(frozen_graph_filename):
    # https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="detection")
    return graph

def load_classification_graph(frozen_graph_filename):
    # https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="classification")
    return graph

def get_traffic_light_b_box(image):
    """Determines the traffic light bounding box within an image"""

    pred_boxes, pred_scores, pred_classes = detection_session.run([detection_boxes, detection_scores, detection_classes], 
        feed_dict={detection_image: np.expand_dims(image, axis=0)})

    pred_boxes = pred_boxes.squeeze()
    pred_classes = pred_classes.squeeze()
    pred_scores = pred_scores.squeeze()

    confidence_thresh = 0.1
    max_confidence = 0
    box_idx = -1
    img_h, img_w = image.shape[:2]

    # Select bounding box with highest level of confidence in traffic light class (10)
    for i in range(pred_boxes.shape[0]):
        box_i = pred_boxes[i]
        score_i = pred_scores[i]
        class_i = pred_classes[i]
        if score_i > confidence_thresh and class_i == 10:
            if score_i > max_confidence:
                box_idx = i
                max_confidence = score_i

    if box_idx > -1:
        box = pred_boxes[box_idx]

        # Get box coordinates
        x0 = int(box[1] * img_w) #top left horizontal
        x1 = int(box[3] * img_w) #bottom right horizontal
        y0 = int(box[0] * img_h) #top left vertical
        y1 = int(box[2] * img_h) #bottom right vertical

        # Expand boxes
        x0 = max(x0-5,0)
        x1 = min(x1+5, img_w)
        y0 = max(y0-10, 0)
        y1 = min(y1+10,img_h)

        return x0, x1, y0, y1
    else:
        return None

def draw_bounding_box(image, box, label, decision):
    color = (255, 0, 0)
    if decision == 0:
        color = (0, 0, 255)
    elif decision == 1:
        color = (0, 255, 255)
    elif decision == 2:
        color = (0, 255, 0)

    font = cv2.FONT_HERSHEY_DUPLEX
    x0, x1, y0, y1 = box
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)    
    cv2.putText(image, label, (x0 + int((x1 - x0)/2) , y1 + 40), font, 1, color, 1, cv2.LINE_AA)
    return image

# Detect traffic light bounding boxes
def get_classification(image, iterat):
    """Determines the color of the traffic light in the image

    Args:
        image (cv::Mat): image containing the traffic light

    Returns:
        int: ID of traffic light color (specified in styx_msgs/TrafficLight)

    """

    tl_b_box = get_traffic_light_b_box(image)

    if tl_b_box == None:
        return 4, image

    x0, x1, y0, y1 = tl_b_box
    tl_b_box_image = image[y0:y1,x0:x1]

    path = output_images_path+"/box_"+str(iterat)+".jpg"
    cv2.imwrite(path, tl_b_box_image)

    return cv_classifier(tl_b_box_image, image, tl_b_box, iterat)


def cv_classifier(image, original_image, box, iterat):
    """
    Classifier that uses the brightness (V in HSV) of an image, and determines the proportion of the image height where the maximum brightness occur.
    If this proportion (normalized mean brightness row index), is on the top of the image, hence it is a red light. 
    If it is in the middle, hence it is a yellow light.
    If it is the bottom, then it is a green light.

    Tested with very high accuracy both in simulator and real images.

    Args:
        Image: cv2.image in BGR

    """

    decision = 4

    img_h, img_w = image.shape[:2]

    bright_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,2] 
    max_bright = bright_img.max()
    thresh_bright_idx_row, thresh_bright_idx_col = np.where(bright_img >= (max_bright-20))
    mean_bright_idx_row = thresh_bright_idx_row.mean()
    norm_mean_bright = mean_bright_idx_row / img_h

    print("Normalized mean index: ", norm_mean_bright)

    if norm_mean_bright < 0.425:
        decision = 0
    elif norm_mean_bright >= 0.425 and norm_mean_bright < 0.55:
        decision = 1
    else:
        decision = 2

    if PRINT_IMAGES == True:
        cv2.imwrite(output_images_path+"/"+'bright_img_'+str(iterat)+'.jpg', bright_img)

    label = light2str[decision]
    cat_image = np.copy(original_image)
    cat_image = draw_bounding_box(cat_image, box, label, decision)

    return decision, cat_image

def cv_classifier_2(image, original_image, box):
    """
    Suggested by @bostonbio, autobots team member

    Args:
        Image: cv2.image in BGR
    """

    decision = 4
    outim = image.copy();
    testim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lowBound1 = np.array([0,51,50])
    highBound1 = np.array([10,255,255])
    testim1 = cv2.inRange(testim, lowBound1 , highBound1)
    
    lowBound2 = np.array([170,51,50])
    highBound2 = np.array([180,255,255])
    testim2 = cv2.inRange(testim, lowBound2 , highBound2)
    
    # testimcombined = cv2.addWeighted(testim1, 1.0, testim2, 1.0, 0.0)
    testimcombined = testim1
    testimblur = cv2.GaussianBlur(testimcombined,(15,15),0)
    c = cv2.HoughCircles(testimblur,cv2.HOUGH_GRADIENT,0.5, 41, param1=70, param2=30,minRadius=7,maxRadius=150)

    if c is not None:
        decision = 0
    
    label = light2str[decision]
    cat_image = np.copy(original_image)
    cat_image = draw_bounding_box(cat_image, box, label, decision)


    return decision, cat_image

def dnn_classifier(image, original_image, box):
    """
    Source: https://github.com/tokyo-drift/traffic_light_classifier

    Args:
        Image: cv2.image in BGR
    """

    # Resize image for DNN squeezenet graph
    resized_image = cv2.resize(image, (32, 32))
    with classification_session.as_default(), classification_graph.as_default():
        softmax_prob = list(classification_session.run(tf.nn.softmax(output_graph.eval(feed_dict={input_graph: [resized_image]}))))
        softmax_ind = softmax_prob.index(max(softmax_prob))


    decision = index2msg[softmax_ind]

    label = light2str[decision]
    cat_image = np.copy(original_image)
    cat_image = draw_bounding_box(cat_image, box, label, decision)


    return decision, cat_image

# Path to models & test images
root_dir = '..'
detection_model_path = os.path.join(root_dir,'ros', 'src', 'tl_detector', 'light_classification', 'models', 'detection_models', 'ssd_mobilenet', 'frozen_inference_graph.pb')
classification_model_path = os.path.join(root_dir, 'ros', 'src', 'tl_detector', 'light_classification', 'models', 'classification_models', 'tokyo_drift', 'model_classification.pb') 

# Real images
green_image_path = [os.path.join('test_images', 'green_{}.jpg'.format(i)) for i in range(1, 12)]
red_image_path = [os.path.join('test_images', 'red_{}.jpg'.format(i)) for i in range(1, 7)]
yellow_image_path = [os.path.join('test_images', 'yellow_{}.jpg'.format(i)) for i in range(1, 5)]

light2str = {0: 'red', 1: 'yellow', 2: 'green', 3: 'none', 4: 'unknown'}
output_images_path = 'pred_images'

### Detection model ####
# Load graph 
detection_graph = load_detection_graph(detection_model_path)

# Start tensor flow session
detection_session = tf.Session(graph=detection_graph)

# (Input) Image: input tensor
detection_image = detection_graph.get_tensor_by_name('detection/image_tensor:0')

# (Output) Boxes: part of the image where an object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection/detection_boxes:0')

# (Output) Scores: level of confidence for each object detected
detection_scores = detection_graph.get_tensor_by_name('detection/detection_scores:0')

# (Output) Classes: the class of the object detected (10 = traffic light)
detection_classes = detection_graph.get_tensor_by_name('detection/detection_classes:0')

### Classification model ###
# Source: https://github.com/tokyo-drift/traffic_light_classifier    
if USE_DNN_CLASSIFIER == True:
    classification_graph = load_classification_graph(classification_model_path)
    classification_graph = load_classification_graph(classification_model_path)
    classification_session = tf.Session(graph=classification_graph)
    input_graph = classification_graph.get_tensor_by_name('classification/input_1_1:0')
    output_graph = classification_graph.get_tensor_by_name('classification/output_0:0')
    index2str = {0: 'red', 1: 'green', 2: 'yellow'}
    index2msg = {0: 0, 1: 2, 2: 1}

# Iterate over images
cat_images = []
correct = 0
total = 0
iteration = 0

for img_path in red_image_path:
    img = cv2.imread(img_path)
    cat, cat_img = get_classification(img, iteration)
    cat_images.append(cat_img)
    if cat == 0:
        correct = correct + 1
    total = total + 1
    iteration = iteration + 1


for img_path in yellow_image_path:
    img = cv2.imread(img_path)
    cat, cat_img = get_classification(img, iteration)
    cat_images.append(cat_img)
    if cat == 1:
        correct = correct + 1
    total = total + 1
    iteration = iteration + 1

for img_path in green_image_path:
    img = cv2.imread(img_path)
    cat, cat_img = get_classification(img, iteration)
    cat_images.append(cat_img)
    if cat == 2:
        correct = correct + 1
    total = total + 1
    iteration = iteration + 1


# Print results
print( 'accuracy: ' + str(correct) + "/" + str(total))

iteration = 0
for i in range(0, len(cat_images)): 
    image = cat_images[i]
    #plt.figure(figsize=(12, 8))
    #plt.imshow(image)
    path = output_images_path+"/pred_"+str(iteration)+".jpg"
    cv2.imwrite(path, image)
    iteration = iteration+1
