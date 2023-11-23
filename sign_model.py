# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from PIL import Image
# from matplotlib import cm
# import cv2
# import math

# BASE_DIR = '/content/Indian/'

# train_datagen = ImageDataGenerator(rescale = 1/255.,
#                                    validation_split = 0.2)

# train_data = train_datagen.flow_from_directory(BASE_DIR,
#                                                target_size = (224,224),
#                                                batch_size = 64,
#                                                subset = 'training')

# test_data = train_datagen.flow_from_directory(BASE_DIR,
#                                                target_size = (224,224),
#                                                batch_size = 64,
#                                                subset = 'validation')

# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# model_0 = tf.keras.Sequential([
#     Conv2D(128 , 2 , activation = 'relu', input_shape = (224,224,3)),
#     MaxPooling2D(2),
#     Conv2D(128 , 2 , activation = 'relu'),
#     MaxPooling2D(2),
#     Conv2D(128 , 2 , activation = 'relu'),
#  #   MaxPool2D(2),
#     Flatten(),
#     Dense(35 , activation = 'sigmoid')
# ])

# model_0.compile(loss = 'categorical_crossentropy',
#                 optimizer = 'adam',
#                 metrics = ['accuracy'])

# history_0 = model_0.fit(train_data,
#                         epochs=1,
#                         batch_size = 64)

# classnames = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U']



    
# def load_image(filename , img_shape=224):

#   img = tf.io.read_file(filename)

#   img = tf.image.decode_image(img , channels = 3)

#   img = tf.image.resize(img , size = [img_shape , img_shape])

#   img = img/255.

#   return img


# def make_predictions(model , filename , classnames):

#   image = load_image(filename)

#   pred = model.predict(tf.expand_dims(image , axis=0))

#   pred_class = classnames[tf.argmax(tf.round(pred)[0])]

#   plt.imshow(image)
#   plt.title(f'Predicted class: {pred_class}')

#   plt.axis(False)

text_file = 'wlasl_class_list.txt'
labels = []

with open(text_file, 'r') as file:
    # labels = [line.strip() for line in f]
    for line in file:
        values = line.strip().split('\t')
        if len(values) > 1:
            label = values[1].strip()
            labels.append(label)

print(len(labels))