import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# model = tf.keras.applications.EfficientNetB1(weights='imagenet')
# tf.keras.models.save_model(model,'../../deep_models_full/efficientnetb1_model.h5',save_format='h5')

# model = tf.keras.applications.DenseNet121(weights='imagenet')
# tf.keras.models.save_model(model,'../../deep_models_full/dense_model.h5',save_format='h5')

model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
tf.keras.models.save_model(model,'../../deep_models_full/resnet_model.h5',save_format='h5')
