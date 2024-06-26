import numpy as np
import tensorflow as tf
import os
from multiprocessing import Pool, cpu_count
import sys

tf.config.set_visible_devices([], 'GPU')

def __make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # print(tf.argmax(preds[0]),pred_index)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # grads = grads[0]
    last_conv_layer_output = last_conv_layer_output[0]
    # last_conv_layer_output = last_conv_layer_output + tf.math.reduce_min(last_conv_layer_output)
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

    # Method2
    heatmapTensor = tf.math.multiply(
        (last_conv_layer_output),
        grads[0],
    )

    # # If you want to remove the negative values which contribute to other labels use this.
    heatmapTensor = tf.nn.relu(heatmapTensor)
    heatmap = tf.nn.relu(heatmap)
    heatmap = tf.squeeze(heatmap)
    # print(np.array(heatmapTensor))
    return [np.array(heatmap), np.array(heatmapTensor)]
    # return [np.array(heatmap), 1*(np.array(heatmapTensor)-np.amin(heatmapTensor))/(np.amax(heatmapTensor)-np.amin(heatmapTensor))]


def processDirectory(trainDir,name,HMtrainDir,listOfFilenameLabel,modelPath,gradientRespectToLayer):

    loaded_model = tf.keras.models.load_model(os.path.join(modelPath))
    fileNames = [fname for fname in os.listdir(os.path.join(trainDir, name))]
    for fname in fileNames:
        I = tf.keras.preprocessing.image.load_img(os.path.join(trainDir,name,fname))
        I = I.resize([224, 224])
        im_array = tf.keras.preprocessing.image.img_to_array(I)
        im_array = tf.keras.applications.resnet50.preprocess_input(im_array)
        # im_array = tf.keras.applications.densenet.preprocess_input(im_array)
        # im_array = tf.keras.applications.efficientnet.preprocess_input(im_array)
        _ , heatmapTensor = __make_gradcam_heatmap(
            np.expand_dims(im_array, axis=0),
            loaded_model,
            gradientRespectToLayer,
            int(listOfFilenameLabel.index(name)),
        )
        with open(os.path.join(HMtrainDir,name,fname[:-5])+".npy", 'wb') as fil:
            np.save(fil, heatmapTensor)
    return

def findHeatmaps(gradientRespectToLayer,modelName,argumentName,typeProcess):

    #Changeable values
    labelFile = "/local-scratch2/korcan/caffe.txt"
    valDir = "/local-scratch2/korcan/ILSVRC2012_img_val"
    trainDir = "/local-scratch2/korcan/ILSVRC2012_img_trainSubset100"
    # labelFile = "/home/foniks/scratch/caffe.txt"
    # valDir = "/home/foniks/scratch/ILSVRC2012_img_val"
    # trainDir = "/home/foniks/scratch/ILSVRC2012_img_train"

    modelPath = "deep_models_full/" + modelName + "_model.h5"
    mobile_model_path = (
        "deep_models_split/" + modelName + "_" + splitLayer + "_mobile_model.h5"
    )
    cloud_model_path = (
        "deep_models_split/" + modelName + "_" + splitLayer + "_cloud_model.h5"
    )


    if typeProcess == 1:
        #Processing training dataset
        with open(labelFile) as file:
            listOfFilenameLabel = [line.split(" ")[0] for line in file]
        HMtrainDir = trainDir+"_HM_"+modelName+"_"+gradientRespectToLayer
        if not os.path.exists(HMtrainDir):
            os.makedirs(HMtrainDir)
        if not os.path.exists(os.path.join(HMtrainDir,argumentName)):
            os.makedirs(os.path.join(HMtrainDir,argumentName))
        processDirectory(trainDir,argumentName,HMtrainDir,listOfFilenameLabel,modelPath,gradientRespectToLayer)

    if typeProcess == 2:
        # Procesing validation dataset
        loaded_model = tf.keras.models.load_model(os.path.join(modelPath))
        HMvalDIR = valDir+"_HM_"+modelName+"_"+gradientRespectToLayer
        if not os.path.exists(HMvalDIR):
            os.makedirs(HMvalDIR)
        label = argumentName
        I = tf.keras.preprocessing.image.load_img(os.path.join(valDir,argumentName+ ".JPEG"))
        I = I.resize([224, 224])
        im_array = tf.keras.preprocessing.image.img_to_array(I)
        im_array = tf.keras.applications.resnet50.preprocess_input(im_array)
        # im_array = tf.keras.applications.densenet.preprocess_input(im_array)
        # im_array = tf.keras.applications.efficientnet.preprocess_input(im_array)
        _ , heatmapTensor = __make_gradcam_heatmap(
            np.expand_dims(im_array, axis=0),
            loaded_model,
            gradientRespectToLayer,
            int(label),
        )
        with open(os.path.join(HMvalDIR,label)+".npy", 'wb') as fil:
            np.save(fil, heatmapTensor)

if __name__ == "__main__":
    # modelName = "efficientnetb0"
    # splitLayer = "block2b_add"
    # modelName = "dense"
    # splitLayer = "pool2_conv"
    modelName = "resnet"
    splitLayer = "conv2_block1_add"
    argumentName = sys.argv[1]
    typeProcess = sys.argv[2]
    findHeatmaps(splitLayer,modelName,str(argumentName),int(typeProcess))
