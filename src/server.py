# import the necessary packages
import pickle
import sys
import time
from os import listdir
from os.path import isfile
from os.path import join

import imutils
import numpy as np
from flask import Flask
from flask_restful import Api
from flask_restful import Resource
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from nms import nms
from scipy import ndimage as ndi
from waitress import serve

import colorgram
import cv2
from opencv_text_detection import utils
from opencv_text_detection.decode import decode
from skimage import color
from skimage import measure

app = Flask(__name__)
api = Api(app)


def image_colorfulness(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


def text_detect(image, min_confidence, width, height):
    # load the input image and grab the image dimensions
    image = cv2.imread(image)
    orig = image.copy()
    (origHeight, origWidth) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width, height)
    ratioWidth = origWidth / float(newW)
    ratioHeight = origHeight / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (imageHeight, imageWidth) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    # load the pre-trained EAST text detector
    #     print("[INFO] loading EAST text detector...")
    net = text_model

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (imageWidth, imageHeight), (123.68, 116.78, 103.94), swapRB=True,
                                 crop=False)

    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # NMS on the the unrotated rects
    confidenceThreshold = min_confidence
    nmsThreshold = 0.4

    # decode the blob info
    (rects, confidences, baggage) = decode(scores, geometry, confidenceThreshold)

    offsets = []
    thetas = []
    for b in baggage:
        offsets.append(b['offset'])
        thetas.append(b['angle'])

    ##########################################################

    functions = [nms.felzenszwalb.nms, nms.fast.nms, nms.malisiewicz.nms]

    drawpolys_count, drawrects_count = 0, 0

    for i, function in enumerate(functions):

        indicies = nms.boxes(rects, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
                             nsm_threshold=nmsThreshold)

        indicies = np.array(indicies).reshape(-1)

        if len(indicies) != 0:
            drawrects = np.array(rects)[indicies]
            drawpolys_count = len(drawrects)

    # convert rects to polys
    polygons = utils.rects2polys(rects, thetas, offsets, ratioWidth, ratioHeight)

    for i, function in enumerate(functions):

        start = time.time()
        indicies = nms.polygons(polygons, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
                                nsm_threshold=nmsThreshold)
        end = time.time()

        indicies = np.array(indicies).reshape(-1)

        if len(indicies) != 0:
            drawpolys = np.array(polygons)[indicies]
            drawrects_count = len(drawpolys)

    return [drawpolys_count, drawrects_count]

ie_model = pickle.load(open('../models/I_E.model', 'rb'))
sn_model = pickle.load(open('../models/S_N.model', 'rb'))
tf_model = pickle.load(open('../models/T_F.model', 'rb'))
jp_model = pickle.load(open('../models/J_P.model', 'rb'))
facecascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../models/haarcascade_eye.xml')
text_model = cv2.dnn.readNet('../models/frozen_east_text_detection.pb')


class ImagePath(Resource):
    def get_mbti(self, imagename):

        imagepath = f'/tmp/image-preprocessing/{imagename}'

        vectors = extractor.extract_folder(imagepath)
        results = []
        for vector in vectors:
            try:
                x = np.array(vector).reshape(1, -1)

                ie_pred = ie_model.predict_proba(x)
                ie_pred = ie_pred[0][0]
                IE = 'I' if ie_pred <= 0.5 else 'E'

                sn_pred = sn_model.predict_proba(x)
                sn_pred = sn_pred[0][0]
                SN = 'S' if sn_pred <= 0.5 else 'N'


                tf_pred = tf_model.predict_proba(x)
                tf_pred = tf_pred[0][0]
                TF = 'T' if tf_pred <= 0.5 else 'F'


                jp_pred = jp_model.predict_proba(x)
                jp_pred = jp_pred[0][0]
                JP = 'J' if jp_pred <= 0.5 else 'P'

                mbti = {
                    'psy_type': '{}{}{}{}'.format(IE, SN, TF, JP),
                     'I_E': ie_pred,
                     'S_N': sn_pred,
                     'T_F': tf_pred,
                     'J_P': jp_pred
                }
                results.append(mbti)
            except Exception as e:
                print(f'Skip {vector} because error {e}')
        return results

    def get(self, user_id):
        mbti = self.get_mbti(user_id)

        if mbti != None:
            return mbti
        else:
            return {'status': 'ERROR'}


class Extractor:
    BATCH_SIZE = 10
    model = ResNet50(weights='imagenet')

    def extract_folder(self, path):
        images = [f for f in listdir(path) if isfile(join(path, f))]
        print('Find images in folder: {}'.format(len(images)))

        vectors = []
        for image in images:
            try:
                vector = self.extract_image(path, image)
            except Exception as e:
                print('GOT ERROR', e)
            else:
                vectors.append(vector)
        return vectors

    def extract_image(self, folder, img_path):
        image_feature_vector = []
        if folder:
            path = '{}/{}'.format(folder, img_path)
        else:
            path = img_path

        print('Process {}'.format(path))
        # colors features
        colors = colorgram.extract(path, 6)
        colors_features = []
        for image_color in colors:
            colors_features.extend([image_color.rgb.r,image_color.rgb.g, image_color.rgb.b])

        image_feature_vector.extend(colors_features)

        # colorfulness
        img = cv2.imread(path)
        img = imutils.resize(img, width=250)
        C = image_colorfulness(img)

        image_feature_vector.extend([C])

        # face detection
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        imgtest1 = img.copy()
        imgtest = cv2.cvtColor(imgtest1, cv2.COLOR_BGR2GRAY)

        faces = facecascade.detectMultiScale(imgtest, scaleFactor=1.2, minNeighbors=5)

        image_feature_vector.extend([C])

        texts = text_detect(path, 0.5, 320, 320)
        image_feature_vector.extend([texts[0], texts[1]])
        return image_feature_vector


extractor = Extractor()

if len(sys.argv) == 1 or sys.argv[1] not in ['prefetch', 'run', 'dev']:
    print("Run {} prefetch|run".format(sys.argv[0]))
    sys.exit(1)
elif sys.argv[1] == 'run':
    extractor = Extractor()
    api.add_resource(ImagePath, "/image/<int:user_id>")
    serve(app, host='0.0.0.0', port=5000)
elif sys.argv[1] == 'dev':
    api.add_resource(ImagePath, "/image/<int:user_id>")
    app.run(host='0.0.0.0', port=5000, debug=True)
