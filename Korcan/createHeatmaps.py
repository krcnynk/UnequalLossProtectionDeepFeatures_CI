import numpy as np
import tensorflow as tf
import os
import asyncio

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

def __make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        print(tf.argmax(preds[0]),pred_index)
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
    #print(1e10*np.array(heatmapTensor))
    return [np.array(heatmap), 1*(np.array(heatmapTensor)-np.amin(heatmapTensor))/(np.amax(heatmapTensor)-np.amin(heatmapTensor))]

def findHeatmaps(gradientRespectToLayer,modelName):
    @background
    def parallelizedFunction(trainDir,name,fname,HMtrainDir):
        I = tf.keras.preprocessing.image.load_img(os.path.join(trainDir,name,fname))
        I = I.resize([224, 224])
        im_array = tf.keras.preprocessing.image.img_to_array(I)
        im_array = tf.keras.applications.densenet.preprocess_input(im_array)
        _ , heatmapTensor = __make_gradcam_heatmap(
            np.expand_dims(im_array, axis=0),
            loaded_model,
            gradientRespectToLayer,
            int(listOfFilenameLabel.index(name)),
        )
        with open(os.path.join(HMtrainDir,name,fname[:-5])+".npy", 'wb') as fil:
            np.save(fil, heatmapTensor)

    modelPath = "deep_models_full/" + modelName + "_model.h5"
    mobile_model_path = (
        "deep_models_split/" + modelName + "_" + splitLayer + "_mobile_model.h5"
    )
    cloud_model_path = (
        "deep_models_split/" + modelName + "_" + splitLayer + "_cloud_model.h5"
    )

    loaded_model = tf.keras.models.load_model(os.path.join(modelPath))


    # labelFile = "/media/sf_CondaEnv/UnequalLossProtectionDeepFeatures_CI/datasets/caffe.txt"
    labelFile = "/home/foniks/UnequalLossProtectionDeepFeatures_CI/datasets/caffe.txt"
    with open(labelFile) as file:
        listOfFilenameLabel = [line.split(" ")[0] for line in file]
    # trainDir = "/media/sf_Downloads/ILSVRC2012_img_train"
    trainDir = "/home/foniks/scratch/ILSVRC2012_img_train"
    HMtrainDir = trainDir+"_HM_"+modelName+"_"+gradientRespectToLayer
    if not os.path.exists(HMtrainDir):
        os.makedirs(HMtrainDir)
    folderNames = [name for name in os.listdir(trainDir)]
    for name in folderNames:
        if not os.path.exists(os.path.join(HMtrainDir,name)):
            os.makedirs(os.path.join(HMtrainDir,name))
        fileNames = [fname for fname in os.listdir(os.path.join(trainDir, name))]
        for fname in fileNames:
           parallelizedFunction(trainDir,name,fname,HMtrainDir)

    # valDir = "/media/sf_Downloads/ILSVRC2012_img_val"
    valDir = "/home/foniks/scratch/ILSVRC2012_img_val"
    HMvalDIR = valDir+"_HM_"+modelName+"_"+gradientRespectToLayer
    if not os.path.exists(HMvalDIR):
        os.makedirs(HMvalDIR)
    fileNames = [name for name in os.listdir(valDir) if os.path.isfile(os.path.join(valDir,name))]
    for f in fileNames:
        label = f[:-5]
        I = tf.keras.preprocessing.image.load_img(os.path.join(valDir,f))
        I = I.resize([224, 224])
        im_array = tf.keras.preprocessing.image.img_to_array(I)
        im_array = tf.keras.applications.densenet.preprocess_input(im_array)
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
    modelName = "dense"
    splitLayer = "pool2_conv"
    findHeatmaps(splitLayer,modelName)
