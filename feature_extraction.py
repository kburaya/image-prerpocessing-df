import argparse
from os import listdir
from os.path import isfile, join

import colorgram
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from scipy import ndimage as ndi
from skimage import measure
from skimage import color

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=False, help="path to the input folder")
args = vars(ap.parse_args())


class Extractor:
    BATCH_SIZE = 10
    model = ResNet50(weights='imagenet')

    def extract_folder(self, path):
        images = [f for f in listdir(path) if isfile(join(path, f))]
        print ('Find images in folder: {}'.format(len(images)))

        try:
            with open('processed_files.csv', 'r') as f:
                processed_images = f.readlines()
        except:
            processed_images = []

        print (processed_images)
        print (images)
        for image in images:
            if image not in processed_images:
                self.extract_image(path, image)

        return

    def extract_image(self, folder, img_path):
        path = '{}/{}'.format(folder, img_path)
        print ('Process {}'.format(path))
        # colors features
        colors = colorgram.extract(path, 6)
        colors_features = []
        for image_color in colors:
            colors_features.extend([image_color.rgb.r,image_color.rgb.g, image_color.rgb.b])

        with open('color_features.csv', 'a') as f:
            feature_str = ', '.join(str(x) for x in colors_features)
            f.writelines(feature_str)
            f.writelines('\n')

        # shape features
        im = ndi.imread(path)
        gimg = color.colorconv.rgb2grey(im)
        contours = measure.find_contours(gimg, 0.8)

        with open('shape_feature.csv', 'a') as f:
            f.writelines(str(len(contours)))
            f.writelines('\n')

        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = self.model.predict(x)
        preds = preds[0]

        with open('object_features.csv', 'a') as f:
            feature_str = ','.join(str(x) for x in preds)
            f.writelines(feature_str)
            f.writelines('\n')

        with open('processed_files.csv', 'a') as f:
            f.writelines(str(img_path))
            f.writelines('\n')

        return


if __name__ == "__main__":
    extractor = Extractor()
    extractor.extract_folder(args['folder'])
