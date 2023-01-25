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
    #return mobile_model
    return tf.keras.models.clone_model(mobile_model)


if __name__ == "__main__":
    modelName = "efficientnetb0"
    splitLayer = "block2b_add"
    # modelName = "resnet18"
    # splitLayer = "add_1"
    # modelName = "dense"
    # splitLayer = "pool2_conv"
    valDir = "/localhome/kuyanik/dataset/ILSVRC2012_img_val"
    trainDir = "/localhome/kuyanik/dataset/ILSVRC2012_img_trainNew"
    # valDir = "/media/sf_Downloads/datasetILSVRC/ILSVRC2012_img_val"
    # trainDir = "/media/sf_Downloads/datasetILSVRC/ILSVRC2012_img_trainNew"
    HMvalDIR = valDir+"_HM_"+modelName+"_"+splitLayer
    HMtrainDIR = trainDir+"_HM_"+modelName+"_"+splitLayer

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    mobileModel = loadModel(modelName, splitLayer)
    #mobileModel = tf.keras.models.load_model("checkpoints1/model.44-0.00.h5")
    mobileModel.trainable = True
    mobileModel.compile(optimizer=tf.keras.optimizers.Adam(1e-1),
                loss=tf.keras.losses.MeanSquaredError(),)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, min_lr=1e-7,min_delta=1e-3,verbose=1)
    #esCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
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

#    xTrainData = []
#    yTrainData = []
#    folderNamesTrain = [name for name in os.listdir(HMtrainDIR)]
#    for fo in folderNamesTrain:
#        TrainFileNames = [name for name in os.listdir(os.path.join(trainDir,fo)) if os.path.isfile(os.path.join(trainDir,fo,name))]
#        HMTrainFileNames = [name for name in os.listdir(os.path.join(HMtrainDIR,fo)) if os.path.isfile(os.path.join(HMtrainDIR,fo,name))]
#        for i in range(len(TrainFileNames)):
#            I = tf.keras.preprocessing.image.load_img(os.path.join(trainDir,fo,TrainFileNames[i]))
#            I = I.resize([224, 224])
#            im_array = tf.keras.preprocessing.image.img_to_array(I)
#            targetTensor = np.load(os.path.join(HMtrainDIR,fo,HMTrainFileNames[i]))
#            xTrainData.append(im_array)
#            yTrainData.append(targetTensor)
#            if im_array is None:
#                print("im_array is None")
#    xTrainData=np.array(xTrainData)
#    yTrainData=np.array(yTrainData)

    folderFilePath = []
    folderNamesTrain = [name for name in os.listdir(trainDir)]
    for fo in folderNamesTrain:
        TrainFileNames = [name for name in os.listdir(os.path.join(trainDir,fo)) if os.path.isfile(os.path.join(trainDir,fo,name))]
        for i in range(len(TrainFileNames)):
            folderFilePath.append(os.path.join(fo,TrainFileNames[i]))
    random.shuffle(folderFilePath)

    datasetCount = np.ceil(sum([len(files) for r, d, files in os.walk(trainDir)]))

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    mobileModel.fit(generate_arrays_from_file(folderFilePath,trainDir,HMtrainDIR,30),steps_per_epoch=datasetCount/30,validation_steps=1000,epochs=1000,
    #mobileModel.fit(x=xTrainData,y=yTrainData,steps_per_epoch = datasetCount,epochs=1000,
    validation_data=(xValidationData,yValidationData),callbacks=[tensorboard_callback,reduce_lr,checkpoint],verbose=1)
    #,max_queue_size=100,workers=4,use_multiprocessing=True)
