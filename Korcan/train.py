import tensorflow as tf
import numpy as np
import sys, os
import datetime
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.BrokenModel import BrokenModel
from runExpt.simmods import *
from models.quantizer import QLayer

np.seterr(over="raise")

def generate_arrays_from_file(HMtrainDIR,trainDir):
    while 1:
        # xTrainData = []
        # yTrainData = []
        folderNamesTrain = [name for name in os.listdir(HMtrainDIR)]
        for fo in folderNamesTrain:
            TrainFileNames = [name for name in os.listdir(os.path.join(trainDir,fo)) if os.path.isfile(os.path.join(trainDir,fo,name))]
            HMTrainFileNames = [name for name in os.listdir(os.path.join(HMtrainDIR,fo)) if os.path.isfile(os.path.join(HMtrainDIR,fo,name))]
            for i in range(len(TrainFileNames)):
                I = tf.keras.preprocessing.image.load_img(os.path.join(trainDir,fo,TrainFileNames[i]))
                I = I.resize([224, 224])
                im_array = tf.keras.preprocessing.image.img_to_array(I)
                targetTensor = np.load(os.path.join(HMtrainDIR,fo,HMTrainFileNames[i]))
                if im_array is None:
                    print("KOKOKANAKAOAAOANAPAKANAOAJAN")
                yield(np.expand_dims(im_array, axis=0),targetTensor)
                # xTrainData.append(im_array)
                # yTrainData.append(targetTensor)
                # i = i + 1
                # if(i == batchsize):
                #     i = 0
                #     xTrainData = []
                #     yTrainData = []
                #     xTrainData = np.array(xTrainData)
                #     yTrainData = np.array(yTrainData)
                #     yield(xTrainData,yTrainData)

        # f = open(path)
        # for line in f:
        #     # create numpy arrays of input data
        #     # and labels, from each line in the file
        #     x, y = process_line(line)
        #     img = load_images(x)
        #     yield (img, y)
        # f.close()

def loadModel(modelName, splitLayer):
    modelPath = "deep_models_full/" + modelName + "_model.h5"
    mobile_model_path = (
        "deep_models_split/" + modelName + "_" + splitLayer + "_mobile_model.h5"
    )
    cloud_model_path = (
        "deep_models_split/" + modelName + "_" + splitLayer + "_cloud_model.h5"
    )

    loaded_model = tf.keras.models.load_model(os.path.join(modelPath))
    # loaded_model.summary()
    loaded_model_config = loaded_model.get_config()
    loaded_model_name = loaded_model_config["name"]
    if os.path.isfile(mobile_model_path) and os.path.isfile(cloud_model_path):
        print(
            f"Sub-models of {loaded_model_name} split at {splitLayer} are available."
        )
        mobile_model = tf.keras.models.load_model(
            os.path.join(mobile_model_path)
        )
        cloud_model = tf.keras.models.load_model(
            os.path.join(cloud_model_path)
        )
    else:
        testModel = BrokenModel(loaded_model, splitLayer, None)
        testModel.splitModel()
        mobile_model = testModel.deviceModel
        cloud_model = testModel.remoteModel
        # Save the mobile and cloud sub-model
        mobile_model.save(mobile_model_path)
        cloud_model.save(cloud_model_path)
    return mobile_model

if __name__ == "__main__":
    modelName = "efficientnetb0"
    splitLayer = "block2b_add"
    # modelName = "resnet18"
    # splitLayer = "add_1"
    # modelName = "dense"
    # splitLayer = "pool2_conv"
    valDir = "/media/sf_Downloads/ILSVRCdatabase/ILSVRC2012_img_val"
    trainDir = "/media/sf_Downloads/ILSVRC2012_img_trainNew"
    HMvalDIR = valDir+"_HM_"+modelName+"_"+splitLayer
    HMtrainDIR = trainDir+"_HM_"+modelName+"_"+splitLayer


    mobileModel = loadModel(modelName, splitLayer)
    mobileModel.trainable = True
    mobileModel.compile(optimizer=tf.keras.optimizers.Adam(1e-1),  # Very low learning rate
              loss=tf.keras.losses.MeanSquaredError(),)
            #   metrics=[tf.keras.metrics.MeanSquaredError()])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
    esCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5')
    xValidationData = []
    yValidationData = []

    validationFileNames = [name for name in os.listdir(valDir) if os.path.isfile(os.path.join(valDir,name))]
    HMvalidationFilenames = [name for name in os.listdir(HMvalDIR) if os.path.isfile(os.path.join(HMvalDIR,name))]
    for i in range(len(validationFileNames)):
        I = tf.keras.preprocessing.image.load_img(os.path.join(valDir,validationFileNames[i]))
        I = I.resize([224, 224])
        im_array = tf.keras.preprocessing.image.img_to_array(I)
        targetTensor = np.load(os.path.join(HMvalDIR,HMvalidationFilenames[i]))
        xValidationData.append(im_array)
        yValidationData.append(targetTensor)
    xValidationData = np.array(xValidationData)
    yValidationData = np.array(yValidationData)

    datasetCount = sum([len(files) for r, d, files in os.walk(trainDir)])
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    mobileModel.fit(generate_arrays_from_file(HMtrainDIR,trainDir),steps_per_epoch = datasetCount,epochs=100, 
    validation_data=(xValidationData,yValidationData),callbacks=[tensorboard_callback,reduce_lr,esCallback,checkpoint])