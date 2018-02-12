# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import keras
import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.preprocessing.coco import CocoGenerator

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import argparse
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

	
		
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--model", required=True, help="path to output model")
# ap.add_argument("-p", "--plot", type=str, default="plot.png",
	# help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
	
	
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
keras.backend.tensorflow_backend.set_session(get_session())

#model = keras.models.load_model('/home/fiuna/Documentos/notebook/snapshots/resnet50_pascal_50.h5', custom_objects=custom_objects)
model = keras.models.load_model(args["model"], custom_objects=custom_objects)


# image = cv2.imread("D:\\Luis\\codigos\\deep learning\\basketball\\JPEGImages\\n02802426_9097.jpeg")
image = cv2.imread(args["image"])

#dst = cv2.resize(src, (2*width, 2*height), interpolation = cv2.INTER_CUBIC)


#draw = cv2.resize(image, (300,300), interpolation = cv2.INTER_CUBIC)
draw = image.copy()
#draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

_, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))

detections2 = detections
#detections = detections[0,1:,4:]

print("Detections shape ", detections.shape)
#print(detections[:, :3, 4:])
print(detections)


predicted_labels = np.argmax(detections[0, :, 4:], axis=1)#orig
#predicted_labels = detections[0, 1, 4:]
#print("Predicted Labels > ", predicted_labels)

scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]
print("Scores shape > ", scores.shape)
idx=0
while scores[idx]>0.5: 
#for idx in range(0, 10):
#idx=3	
# if score < 0.5:
#     continue
	b = detections[0, idx, :4].astype(int)
#print("b values> ", b)

	cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 3)
# caption = "{} {:.3f}".format(val_generator.label_to_name(label), score)
	caption = "Tomate"
	cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
	cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

	#draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
	print("Scores > ", scores[idx])
	idx = idx+1

#draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
cv2.imwrite("xxpruebas.jpg",draw)
cv2.imshow("prueba", draw)
cv2.waitKey()
cv2.destroyAllWindows()
  
