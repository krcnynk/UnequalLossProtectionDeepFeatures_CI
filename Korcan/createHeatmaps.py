import numpy as np
import tensorflow as tf
import os

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
    return [np.array(heatmap), np.array(heatmapTensor)]

def findHeatmaps(gradientRespectToLayer,modelName):

    modelPath = "deep_models_full/" + modelName + "_model.h5"
    mobile_model_path = (
        "deep_models_split/" + modelName + "_" + splitLayer + "_mobile_model.h5"
    )
    cloud_model_path = (
        "deep_models_split/" + modelName + "_" + splitLayer + "_cloud_model.h5"
    )

    loaded_model = tf.keras.models.load_model(os.path.join(modelPath))


    labelFile = "/media/sf_Downloads/ILSVRCdatabase/caffe.txt"
    with open(labelFile) as file:
        listOfFilenameLabel = [line.split(" ")[0] for line in file]

    trainDir = "/media/sf_Downloads/datasetILSVRC/ILSVRC2012_img_trainNew"
    HMtrainDir = trainDir+"_HM_"+modelName+"_"+gradientRespectToLayer
    if not os.path.exists(HMtrainDir):
        os.makedirs(HMtrainDir)
    folderNames = [name for name in os.listdir(trainDir)]

    for name in folderNames:
        if not os.path.exists(os.path.join(HMtrainDir,name)):
            os.makedirs(os.path.join(HMtrainDir,name))
        fileNames = [fname for fname in os.listdir(os.path.join(trainDir, name))]
        for fname in fileNames:
            I = tf.keras.preprocessing.image.load_img(os.path.join(trainDir,name,fname))
            I = I.resize([224, 224])
            im_array = tf.keras.preprocessing.image.img_to_array(I)
            _ , heatmapTensor = __make_gradcam_heatmap(
                np.expand_dims(im_array, axis=0),
                loaded_model,
                gradientRespectToLayer,
                int(listOfFilenameLabel.index(name)),
            )
            with open(os.path.join(HMtrainDir,name,fname[:-5])+".npy", 'wb') as fil:
                np.save(fil, heatmapTensor)







    # valDir = "/media/sf_Downloads/ILSVRCdatabase/ILSVRC2012_img_val"
    # HMvalDIR = valDir+"_HM_"+modelName+"_"+gradientRespectToLayer


    # if not os.path.exists(HMvalDIR):
    #     os.makedirs(HMvalDIR)

    # fileNames = [name for name in os.listdir(valDir) if os.path.isfile(os.path.join(valDir,name))]

    # for f in fileNames:
    #     label = f[:-5]
    #     I = tf.keras.preprocessing.image.load_img(os.path.join(valDir,f))
    #     I = I.resize([224, 224])
    #     im_array = tf.keras.preprocessing.image.img_to_array(I)
    #     _ , heatmapTensor = __make_gradcam_heatmap(
    #         np.expand_dims(im_array, axis=0),
    #         loaded_model,
    #         gradientRespectToLayer,
    #         int(label),
    #     )
    #     with open(os.path.join(HMvalDIR,label)+".npy", 'wb') as fil:
    #         np.save(fil, heatmapTensor)

    # for fo in folderNames:
    #     if not os.path.exists(os.path.join(HMtrainDIR,fo)):
    #         os.makedirs(os.path.join(HMtrainDIR,fo))
    #     fileNames = [name for name in os.listdir(os.path.join(trainDir,fo)) if os.path.isfile(os.path.join(trainDir,fo,name))]
    #     for f in fileNames:
    #         I = tf.keras.preprocessing.image.load_img(os.path.join(trainDir,fo,f))
    #         I = I.resize([224, 224])
    #         im_array = tf.keras.preprocessing.image.img_to_array(I)
    #         _ , heatmapTensor = __make_gradcam_heatmap(
    #             np.expand_dims(im_array, axis=0),
    #             loaded_model,
    #             gradientRespectToLayer,
    #             int(fo),
    #         )
    #         with open(os.path.join(HMtrainDIR,fo,f[:-5])+".npy", 'wb') as fil:
    #             np.save(fil, heatmapTensor)

if __name__ == "__main__":
    modelName = "efficientnetb0"
    splitLayer = "block2b_add"
    findHeatmaps(splitLayer,modelName)
