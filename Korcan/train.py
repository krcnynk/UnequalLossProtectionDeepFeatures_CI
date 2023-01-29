import tensorflow as tf
import numpy as np
import sys, os
import datetime
import random
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.BrokenModel import BrokenModel
from runExpt.simmods import *
from models.quantizer import QLayer

np.seterr(over="raise")

def generate_arrays_from_file(folderFilePath,trainBaseDir,HMbaseDIR,batchSize):
    while 1:
        xTrainData = []
        yTrainData = []
        count = 0
        for i in range(len(folderFilePath)):
            count = count +1
            I = tf.keras.preprocessing.image.load_img(os.path.join(trainBaseDir,folderFilePath[i]))
            I = I.resize([224, 224])
            im_array = tf.keras.preprocessing.image.img_to_array(I)
            im_array = tf.keras.applications.densenet.preprocess_input(im_array)
            targetTensor = np.load(os.path.join(HMbaseDIR,folderFilePath[i][:-4]+"npy"))
            xTrainData.append(im_array)
            yTrainData.append(targetTensor)
            if im_array is None:
                print("im_array is None")
            if(count == batchSize):
                count = 0
                yield(np.array(xTrainData),np.array(yTrainData))
                xTrainData = []
                yTrainData = []
                #yield(np.expand_dims(im_array, axis=0),targetTensor)

# def generate_arrays_from_file_Validation(valDir,HMvalDIR,batchSize):
#     while 1:
#         xValidationData = []
#         yValidationData = []
#         count = 0
#         validationFileNames = [name for name in os.listdir(valDir) if os.path.isfile(os.path.join(valDir,name))]
#         HMvalidationFilenames = [name for name in os.listdir(HMvalDIR) if os.path.isfile(os.path.join(HMvalDIR,name))]
#         for i in range(len(validationFileNames)):
#             count = count +1
#             I = tf.keras.preprocessing.image.load_img(os.path.join(valDir,validationFileNames[i]))
#             I = I.resize([224, 224])
#             im_array = tf.keras.preprocessing.image.img_to_array(I)
#             im_array = tf.keras.applications.densenet.preprocess_input(im_array)
#             targetTensor = np.load(os.path.join(HMvalDIR,HMvalidationFilenames[i]))
#             xValidationData.append(im_array)
#             yValidationData.append(targetTensor)
#             if(count == batchSize):
#                 count = 0
#                 yield(np.array(xValidationData),np.array(yValidationData))
#                 xValidationData = []
#                 yValidationData = []

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
    # return tf.keras.models.clone_model(mobile_model)


if __name__ == "__main__":
    # modelName = "efficientnetb0"
    # splitLayer = "block2b_add"
    # modelName = "resnet18"
    # splitLayer = "add_1"

    modelName = "dense"
    splitLayer = "pool2_conv"
    # valDir = "/home/foniks/scratch/ILSVRC2012_img_val"
    # trainDir = "/home/foniks/scratch/ILSVRC2012_img_train"
    valDir = "/local-scratch2/korcan/ILSVRC2012_img_val"
    trainDir = "/local-scratch2/korcan/ILSVRC2012_img_trainSubset5"

    HMvalDIR = valDir+"_HM_"+modelName+"_"+splitLayer
    HMtrainDIR = trainDir+"_HM_"+modelName+"_"+splitLayer


    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    mobileModel = loadModel(modelName, splitLayer)
    #mobileModel = tf.keras.models.load_model("checkpoints1/model.44-0.00.h5")
    mobileModel.trainable = True
    mobileModel.compile(optimizer=tf.keras.optimizers.Adam(1e-1),
                loss=tf.keras.losses.MeanAbsolutePercentageError(),)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=4, min_lr=1e-7,min_delta=1e-4,verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5')
    

    folderFilePath = []
    folderNamesTrain = [name for name in os.listdir(trainDir)]
    for fo in folderNamesTrain:
        TrainFileNames = [name for name in os.listdir(os.path.join(trainDir,fo)) if os.path.isfile(os.path.join(trainDir,fo,name))]
        for i in range(len(TrainFileNames)):
            folderFilePath.append(os.path.join(fo,TrainFileNames[i]))
    random.shuffle(folderFilePath)

    datasetCount = np.ceil(sum([len(files) for r, d, files in os.walk(trainDir)]))

    xValidationData = []
    yValidationData = []
    validationFileNames = [name for name in os.listdir(valDir) if os.path.isfile(os.path.join(valDir,name))]
    HMvalidationFilenames = [name for name in os.listdir(HMvalDIR) if os.path.isfile(os.path.join(HMvalDIR,name))]
    for i in range(len(validationFileNames)):
        I = tf.keras.preprocessing.image.load_img(os.path.join(valDir,validationFileNames[i]))
        I = I.resize([224, 224])
        im_array = tf.keras.preprocessing.image.img_to_array(I)
        im_array = tf.keras.applications.densenet.preprocess_input(im_array)
        targetTensor = np.load(os.path.join(HMvalDIR,HMvalidationFilenames[i]))
        xValidationData.append(im_array)
        yValidationData.append(targetTensor)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    batchSize = 1
    print("MAX MIN",np.amax(np.array(yValidationData)),np.amin(np.array(yValidationData)))
    mobileModel.evaluate(np.array(xValidationData),np.array(yValidationData))
    # mobileModel.fit(generate_arrays_from_file(folderFilePath,trainDir,HMtrainDIR,batchSize),steps_per_epoch=datasetCount/batchSize,validation_steps=1000,epochs=1000,
    # # validation_data=generate_arrays_from_file_Validation(valDir,HMvalDIR,batchSize),
    # validation_data=(np.array(xValidationData),np.array(yValidationData)),
    # callbacks=[tensorboard_callback,reduce_lr,checkpoint],verbose=1)
    # #,max_queue_size=100,workers=4,use_multiprocessing=True)
