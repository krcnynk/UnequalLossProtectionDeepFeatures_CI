import tensorflow as tf

model = tf.keras.applications.DenseNet121(weights='imagenet')
tf.keras.models.save_model(model,'./eff.h5',save_format='h5')
