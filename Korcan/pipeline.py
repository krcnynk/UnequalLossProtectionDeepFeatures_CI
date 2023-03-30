import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib as mpl
import pickle

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys, os
import math
import random
import glob
import gbChannel

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.BrokenModel import BrokenModel
from runExpt.simmods import *
from models.quantizer import QLayer

np.seterr(over="raise")


class pipeline:
    pdict = {}
    heatmapsBatch = []
    heatMapsChannelsBatch = []

    def __normalizeToUnit(self, arr):
        if (np.ptp(arr)) == 0:
            return arr
        arr = (arr - arr.min()) / (np.ptp(arr))
        return arr

    def __superimposeHeatmap(self, img, heatmap, alpha=0.3):
        if len(img.shape) == 3:
            img = self.__rgb2gray(img)
        # heatmap = normz(heatmap)
        img = np.stack((img, img, img), axis=2)
        # img = tf.keras.preprocessing.image.array_to_img(img)
        # img = tf.keras.preprocessing.image.img_to_array(img)
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap[heatmap == 0] = 0

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        return superimposed_img

    def saveSuperImposedChannels(self, modelName):
        mainPath = os.path.abspath(
            "Korcan/Plots/" + modelName + "/tensorHeatmapOverlay/"
        )
        if not os.path.exists(mainPath):
            os.makedirs(mainPath)
        for label in self.dataset_y_labels:
            if not os.path.exists(os.path.join(mainPath, label)):
                os.makedirs(os.path.join(mainPath, label))

        featureMapBatch = np.array(self.latentOutputBatch)
        heatMapBatch = np.array(self.heatMapsChannelsBatch)
        iS = featureMapBatch.shape
        for i_b in range(len(featureMapBatch)):  # 9
            matrixFeature = np.empty((iS[1] * 6, iS[2] * 4))
            matrixHeat = np.empty((iS[1] * 6, iS[2] * 4))
            ind = 0
            heatMap = self.__normalizeToUnit(heatMapBatch[i_b])
            for i_cx in range(6):
                for i_cy in range(4):
                    # print(featureMap.shape)
                    matrixFeature[
                        i_cx * iS[1] : i_cx * iS[1] + iS[1],
                        i_cy * iS[1] : i_cy * iS[1] + iS[1],
                    ] = featureMapBatch[i_b, :, :, ind]
                    matrixHeat[
                        i_cx * iS[1] : i_cx * iS[1] + iS[1],
                        i_cy * iS[1] : i_cy * iS[1] + iS[1],
                    ] = heatMap[:, :, ind]
                    ind = ind + 1

            jet = cm.get_cmap("jet", 256)
            r = np.linspace(0, 1, 256)
            colors = jet(r)
            for i in range(len(colors)):
                colors[i][3] = r[i]
            newcmp = matplotlib.colors.ListedColormap(colors)
            plt.imshow(matrixFeature, interpolation="bilinear", cmap="gray")
            matrixHeat = matrixHeat * 1  # HERE COMMENT
            matrixHeat[matrixHeat > 1] = 1
            # matrixHeat[matrixHeat < 0.8] = 0.6
            # matrixHeat[matrixHeat < 0.4] = 0.3
            plt.imshow(matrixHeat, interpolation="bilinear", cmap=newcmp)
            plt.colorbar()
            plt.axis("off")
            plt.savefig(
                os.path.join(
                    mainPath, self.dataset_y_labels[i_b], self.file_names[i_b]
                ),
                bbox_inches="tight",
                dpi=400,
            )
            plt.close()

    def __make_gradcam_heatmap(
        self, img_array, model, last_conv_layer_name, pred_index=None
    ):
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        # grads = grads[0]
        last_conv_layer_output = last_conv_layer_output[0]
        # last_conv_layer_output = last_conv_layer_output + tf.math.reduce_min(last_conv_layer_output)
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        # #Method1
        # heatmapTensor = tf.math.multiply(
        #     (last_conv_layer_output + tf.math.reduce_min(last_conv_layer_output)),
        #     grads[0],
        # )

        # Method2
        heatmapTensor = tf.math.multiply(
            (last_conv_layer_output),
            grads[0],
        )

        # Method3
        # heatmapTensor = np.empty(last_conv_layer_output.shape)
        # for i in range(len(pooled_grads)):
        #     heatmapTensor[..., i] = last_conv_layer_output[..., i] * pooled_grads[i]

        # # If you want to remove the negative values which contribute to other labels use this.
        heatmapTensor = tf.nn.relu(heatmapTensor)
        heatmap = tf.nn.relu(heatmap)

        heatmap = tf.squeeze(heatmap)
        # heatmapTensor = np.stack(
        #     [np.array(heatmap) for i in range(len(pooled_grads))], axis=2
        # )
        return [np.array(heatmap), np.array(heatmapTensor)]

    def __make_gradcam_heatmap_fromTrainedModel(self, input):
        pred = np.squeeze(self.trained_model.predict(input), axis=0)  # 56x56x24
        return None, pred

    def __rgb2gray(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def __quantize(self, tensor, nBits):
        return (
            np.round(
                ((tensor - tensor.min()) / (tensor.max() - tensor.min()))
                * ((2**nBits) - 1)
            ),
            tensor.min(),
            tensor.max(),
        )

    def __inverseQuantize(self, tensor, nBits, minV, maxV):
        return np.array(((tensor * (maxV - minV) / (2**nBits - 1)) + minV))

        # # def __resizeAndFindEdgeFromBatch(self):
        # self.resizedImages = []
        # for i_b in range(len(self.dataset_x_files)):
        #     self.resizedImages.append(
        #         cv.resize(
        #             self.dataset_x_files[i_b],
        #             dsize=(56, 56),
        #             interpolation=cv.INTER_CUBIC,
        #         )
        #     )
        # self.resizedImages = np.array(self.resizedImages)
        # edge = []
        # for i_b in range(self.resizedImages.shape[0]):
        #     grayscale = self.__rgb2gray(self.resizedImages[i_b, ...])
        #     grayscale = cv.GaussianBlur(grayscale, [5, 5], 0)
        #     cannyOutput = cv.Canny(np.uint8(grayscale), 100, 200)
        #     kernel = np.ones((3, 3), np.uint8)
        #     # cannyOutput = cv.dilate(cannyOutput, kernel, iterations=1)
        #     edge.append(cannyOutput)
        #     # edge.append(np.zeros((56, 56)))
        # return np.array(edge).astype(bool)

    def __getOrderedImportantPacketIndex(self, importanceOfPackets):
        indexOfLossedPackets = np.argsort(importanceOfPackets)[::-1]
        return list(indexOfLossedPackets)

    def loadData(self, path, reshapeDims, normalize):
        (
            self.dataset_x_files,
            self.dataset_y_labels,
            self.file_names,
        ) = fn_Data_PreProcessing_ImgClass(path, reshapeDims, normalize)
        self.dataset_y_labels_int = [int(item) for item in self.dataset_y_labels]
        # self.mobile_model.summary()

        # self.latentOutputBatch = self.mobile_model.predict(np.array(self.dataset_x_files))
        self.latentOutputBatch = self.mobile_model.predict(
            tf.keras.applications.resnet50.preprocess_input(
                np.array(self.dataset_x_files)
            )
        )
        # self.latentOutputBatch = self.mobile_model.predict(tf.keras.applications.densenet.preprocess_input(np.array(self.dataset_x_files)))

        # x=tf.keras.applications.densenet.preprocess_input(np.array(self.dataset_x_files), data_format=None)
        # predicted = self.loaded_model.predict(np.array(x))
        # Top1_accuracy = np.argmax(predicted, axis=1)
        # # Top1_precision = np.max(predicted, axis=1)
        # Accuracy = np.sum(
        #     np.equal(Top1_accuracy, np.array(self.dataset_y_labels_int))
        # ) / len(self.dataset_y_labels_int)
        # print("accuracy",Accuracy)
        # print("KKKK",(predicted.shape))
        # print((np.max(predicted, axis=1)))
        # print(predicted[0])
        # print((self.latentOutputBatch[0].shape))
        # print((self.latentOutputBatch[1].shape))
        # print(type(self.latentOutputBatch))

        self.batchSize = self.latentOutputBatch.shape[0]
        self.H = self.latentOutputBatch.shape[1]
        self.W = self.latentOutputBatch.shape[2]
        self.C = self.latentOutputBatch.shape[3]

    def loadModel(
        self,
        model_path,
        mobile_model_path,
        cloud_model_path,
        trained_model_path,
        splitLayer,
    ):
        self.loaded_model = tf.keras.models.load_model(os.path.join(model_path))
        self.loaded_model.summary()
        loaded_model_config = self.loaded_model.get_config()
        loaded_model_name = loaded_model_config["name"]
        if os.path.isfile(mobile_model_path) and os.path.isfile(cloud_model_path):
            print(
                f"Sub-models of {loaded_model_name} split at {splitLayer} are available."
            )
            self.mobile_model = tf.keras.models.load_model(
                os.path.join(mobile_model_path)
            )
            self.cloud_model = tf.keras.models.load_model(
                os.path.join(cloud_model_path)
            )
        else:
            testModel = BrokenModel(self.loaded_model, splitLayer, None)
            testModel.splitModel()
            self.mobile_model = testModel.deviceModel
            self.cloud_model = testModel.remoteModel
            # Save the mobile and cloud sub-model
            self.mobile_model.save(mobile_model_path)
            self.cloud_model.save(cloud_model_path)
        self.trained_model = tf.keras.models.load_model(
            os.path.join(trained_model_path)
        )
        self.mobile_model.summary()
        # self.cloud_model.summary()

    def findHeatmaps(self, gradientRespectToLayer, modelName, dataSet):
        self.heatmapsBatch = []
        self.heatMapsChannelsBatch = []

        for i_b in range(len(self.dataset_x_files)):
            # a, b = self.__make_gradcam_heatmap(
            #     np.expand_dims(np.array(self.dataset_x_files)[i_b], axis=0),
            #     self.loaded_model,
            #     gradientRespectToLayer,
            #     np.array(self.dataset_y_labels_int)[i_b],
            # )
            a, b = self.__make_gradcam_heatmap_fromTrainedModel(
                np.expand_dims(np.array(self.dataset_x_files)[i_b], axis=0),
            )
            self.heatmapsBatch.append(a)
            self.heatMapsChannelsBatch.append(b)
        self.heatmapsBatch = np.array(self.heatmapsBatch)
        self.heatMapsChannelsBatch = np.array(self.heatMapsChannelsBatch)

    def __savePacketLossImages(self, lossedTensorBatchArray, case, modelName):
        mainPath = os.path.abspath("Korcan/Plots/" + modelName + "/tensorLoss/" + case)
        if not os.path.exists(mainPath):
            os.makedirs(mainPath)
        for label in self.dataset_y_labels:
            if not os.path.exists(os.path.join(mainPath, label)):
                os.makedirs(os.path.join(mainPath, label))

        lossedTensorBatchArray = np.array(lossedTensorBatchArray).astype(np.float64)
        shape = lossedTensorBatchArray.shape
        for i_b in range(len(lossedTensorBatchArray)):  # 9
            arr = np.empty((shape[1] * 16, shape[2] * 16))
            ind = 0
            for i_cx in range(16):
                for i_cy in range(16):
                    # lossedTensorBatchArray[i_b, :, :, ind] = self.__normalizeToUnit(
                    #     lossedTensorBatchArray[i_b, :, :, ind]
                    # )
                    arr[
                        i_cx * 56 : i_cx * 56 + 56, i_cy * 56 : i_cy * 56 + 56
                    ] = lossedTensorBatchArray[i_b, :, :, ind]
                    ind = ind + 1
            im = plt.matshow(arr, cmap="gray")
            plt.axis("off")
            plt.savefig(
                os.path.join(
                    mainPath, self.dataset_y_labels[i_b], self.file_names[i_b]
                ),
                bbox_inches="tight",
                dpi=400,
            )
            plt.close()

    def makePlot(self, pathAcc, pathLoss):
        # self.pdict
        # perc,type -> acc,loss

        # cases = ["Most important (GradCAM)","b",".",":",
        # "Least important (GradCAM)","g",".",":",
        # # "R_RS_FEC_10_90","m",".","-",
        # "Most important (Proxy)","b",".","-",
        # "Least important (Proxy)","g",".","-",]

        cases = [
            "Most important",
            "b",
            ".",
            "-",
            "Least important",
            "g",
            ".",
            "-",
            "Unprotected (IID)",
            "r",
            ".",
            "-",
            "Unprotected (Burst)",
            "m",
            ".",
            "-",

            # "R_RS_FEC_10_90","m",".","-",
            "FEC_20_80",
            "m",
            ".",
            "-",
            "FEC_30_70",
            "m",
            ".",
            "--",
            "FEC_40_60",
            "m",
            ".",
            "-.",
            "FEC_50_50",
            "m",
            ".",
            ":",
        ]

        types = sorted(list(set([i[1] for i in self.pdict.keys()])))
        seriesX = [[] for _ in range(len(types))]
        seriesY = [[] for _ in range(len(types))]
        seriesYmin = [[] for _ in range(len(types))]
        seriesYmax = [[] for _ in range(len(types))]
        for key, value in self.pdict.items():
            print(value)
            index = types.index(key[1])
            seriesX[index].append(float(key[0]))
            seriesY[index].append(float(value["acc"]))
            seriesYmin[index].append(float(value["min"]))
            seriesYmax[index].append(float(value["max"]))
        # plt.title("Top1 Accuracy")
        plt.xlabel("Percent Lost")
        plt.ylabel("Top-1 Accuracy")
        for s in range(len(seriesX)):
            print(types[s])
            mapping = cases.index(types[s])
            seriesX[s], seriesY[s] = zip(*sorted(zip(seriesX[s], seriesY[s])))
            if (
                types[s] == "Least important"
                or types[s] == "Most important"
                or types[s] == "Unprotected (IID)"
                or types[s] == "Unprotected (Burst)"
                or types[s] == "Most important(GradCAM)"
                or types[s] == "Least important(GradCAM)"
            ):
                plt.scatter(
                    seriesX[s],
                    seriesY[s],
                    s=25,
                    label="_nolegend_",
                    marker=cases[mapping + 2],
                    color=cases[mapping + 1],
                )
                plt.plot(
                    seriesX[s],
                    seriesY[s],
                    label=cases[mapping],
                    linestyle=cases[mapping + 3],
                    linewidth=1.2,
                    color=cases[mapping + 1],
                )
            else:
                print(types[s])
                plt.scatter(
                    seriesX[s],
                    seriesY[s],
                    s=25,
                    marker=cases[mapping + 2],
                    color=cases[mapping + 1],
                )
                plt.plot(
                    seriesX[s],
                    seriesY[s],
                    label=cases[mapping],
                    linestyle=cases[mapping + 3],
                    linewidth=1.2,
                    color=cases[mapping + 1],
                )
            plt.fill_between(seriesX[s], seriesYmin[s], seriesYmax[s])

        # reordering the labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # specify order
        # order=[0,2,1,3]
        # order=[3,5,6, 2,0,1,7,4]

        plt.legend(
            # [handles[i] for i in order], [labels[i] for i in order],
            loc="upper right",
            fontsize="xx-small",
            markerscale=0.7,
            # ncol=2,
            fancybox=True,
            # shadow=True,
            prop={"size": 8},
        )


        
        # plt.axis('off')
        plt.savefig(
            pathAcc,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        # types = list(set([i[1] for i in self.pdict.keys()]))
        # seriesX = [[] for _ in range(len(types))]
        # seriesY = [[] for _ in range(len(types))]
        # for key, value in self.pdict.items():
        #     index = types.index(key[1])
        #     seriesX[index].append(float(key[0]))
        #     seriesY[index].append(float(value["loss"]))
        # plt.title("Top1 Loss")
        # plt.xlabel("Percent Lost")
        # plt.ylabel("Loss")
        # for s in range(len(seriesX)):
        #     mapping = cases.index(types[s])
        #     seriesX[s], seriesY[s] = zip(*sorted(zip(seriesX[s], seriesY[s])))
        #     plt.scatter(
        #         seriesX[s],
        #         seriesY[s],
        #         label=cases[mapping],
        #         marker=cases[mapping + 2],
        #         color=cases[mapping + 1],
        #     )
        #     plt.plot(seriesX[s], seriesY[s], linewidth=0.5, color=cases[mapping + 1])
        # plt.legend(
        #     bbox_to_anchor=(1.04, 1),
        #     loc="upper left",
        #     ncol=2,
        #     fancybox=True,
        #     shadow=True,
        # )
        # plt.savefig(
        #     pathLoss,
        #     bbox_inches="tight",
        #     dpi=300,
        # )
        # plt.close()
        return

    def cleanPlot(self):
        self.pdict = {}

    def getMetrics(self, data):
        cce = tf.keras.losses.CategoricalCrossentropy()
        predicted = self.cloud_model.predict(tf.convert_to_tensor(data))
        Top1_accuracy = np.argmax(predicted, axis=1)
        # Top1_precision = np.max(predicted, axis=1)
        Accuracy = np.sum(
            np.equal(Top1_accuracy, np.array(self.dataset_y_labels_int))
        ) / len(self.dataset_y_labels_int)
        print(Accuracy)
        Loss = cce(tf.one_hot(self.dataset_y_labels_int, 1000), predicted).numpy()
        return {"acc": Accuracy, "loss": Loss}

    # def findPercentileLossPerChannelFM(
    #     self, percent, qBits, saveImages=False, bot=False
    # ):
    #     fmLBatch = []
    #     for i_b in range(self.heatMapsChannelsBatch.shape[0]):
    #         quantizedData, minVal, maxVal = self.__quantize(
    #             self.latentOutputBatch[i_b], qBits
    #         )
    #         fmL = np.copy(quantizedData)  ##Single Tensor i.e batch element
    #         lHC = np.zeros_like(quantizedData, dtype=bool)

    #         numberOfPointsToLose = math.ceil(
    #             np.prod(quantizedData.shape[:2]) * percent / 100
    #         )
    #         if numberOfPointsToLose != 0:
    #             for i_c in range(lHC.shape[2]):
    #                 if bot == False:
    #                     Inds = np.argsort(
    #                         self.heatMapsChannelsBatch[i_b, ..., i_c].flatten(),
    #                     )[-numberOfPointsToLose:]
    #                 else:
    #                     Inds = np.argsort(
    #                         self.heatMapsChannelsBatch[i_b, ..., i_c].flatten(),
    #                     )[:numberOfPointsToLose]
    #                 lHC[..., i_c][
    #                     np.unravel_index(
    #                         Inds, self.heatMapsChannelsBatch[i_b, ..., i_c].shape
    #                     )
    #                 ] = True
    #             # self.lHC_Batch.append(lHC)
    #             fmL[lHC] = 0
    #         # print("NR", np.count_nonzero(lHC), percent)
    #         fmL = self.__inverseQuantize(fmL, qBits, minVal, maxVal)
    #         fmLBatch.append(fmL)
    #     if bot == True:
    #         self.pdict[
    #             "{:.1f}".format(percent), "lossmapTargetedBot"
    #         ] = self.getMetrics(fmLBatch)
    #     else:
    #         self.pdict[
    #             "{:.1f}".format(percent), "lossmapTargetedTop"
    #         ] = self.getMetrics(fmLBatch)

    #     if saveImages:
    #         if bot == True:
    #             self.__savePacketLossImages(fmLBatch, "targetedPixelLoss" + "Least important")
    #         else:
    #             self.__savePacketLossImages(fmLBatch, "targetedPixelLoss" + "Most important")

    # def findPercentileRandomLossPerChannelFM(self, percent, qBits, saveImages=False):
    #     fmLRandomBatch = []
    #     for i_b in range(self.heatMapsChannelsBatch.shape[0]):
    #         quantizedData, minVal, maxVal = self.__quantize(
    #             self.latentOutputBatch[i_b], qBits
    #         )
    #         fmL = np.copy(quantizedData)
    #         lHC = np.zeros_like(quantizedData, dtype=bool)
    #         numberOfPointsToLose = math.ceil(
    #             np.prod(quantizedData.shape[:2]) * percent / 100
    #         )
    #         # print(numberOfPointsToLose)
    #         if numberOfPointsToLose != 0:
    #             for i_c in range(lHC.shape[2]):
    #                 rng = np.random.default_rng()
    #                 channel = np.zeros(np.prod(quantizedData.shape[:2]))
    #                 channel[:numberOfPointsToLose] = True
    #                 rng.shuffle(channel)
    #                 channel = np.resize(channel, quantizedData.shape[:2])
    #                 rng.shuffle(channel, axis=0)
    #                 rng.shuffle(channel, axis=1)
    #                 lHC[..., i_c] = channel
    #             fmL[lHC] = 0
    #         # print("R", np.count_nonzero(lHC), percent)
    #         fmL = self.__inverseQuantize(fmL, qBits, minVal, maxVal)
    #         fmLRandomBatch.append(fmL)
    #     self.pdict["{:.1f}".format(percent), "lossmapRandom"] = self.getMetrics(
    #         fmLRandomBatch
    #     )
    #     if saveImages:
    #         self.__savePacketLossImages(fmLRandomBatch, "randomPixelLoss")

    def packetLossSim(
        self,
        packetNum,
        qBits,
        percOfPacketLoss,
        case,
        fecPerc=None,
        protectedPerc=None,
        saveImages=False,
        modelName=None,
    ):
        # iteration = 1
        # if(case == "Unprotected (IID)" or case == "Random_RSCorrected" or case == "Random_RSCorrected_FECRemovesBOT"):
        #     iteration = 1
        # scores = []

        # for i in range(iteration):
        packetHeight = math.ceil(self.H / packetNum)  # 55/12, 4.66->5
        remainder = self.H % packetHeight  # 1
        if remainder != 0:
            pad = packetHeight - remainder
        else:
            pad = 0
        fmLPacketizedLoss = []
        importancePacketsTensor = []
        rng = np.random.default_rng()
        packetsLost = 0
        packetsSent = 0

        for i_b in range(self.batchSize):
            quantizedData, minVal, maxVal = self.__quantize(
                self.latentOutputBatch[i_b], qBits
            )
            # quantizedData = self.latentOutputBatch[i_b]
            fmL = np.copy(quantizedData)
            heatMap = np.copy(self.heatMapsChannelsBatch[i_b, ...])
            fmLChannelArray = np.dsplit(fmL, self.C)
            heatMapChannelArray = np.dsplit(heatMap, self.C)

            packetizedfmL = []
            packetizedheatMap = []
            for i_c in range(self.C):
                packetizedfmL = packetizedfmL + np.vsplit(
                    np.pad(
                        fmLChannelArray[i_c].squeeze(),
                        [(0, pad), (0, 0)],
                        mode="constant",
                        constant_values=0,
                    ),
                    packetNum,
                )
                packetizedheatMap = packetizedheatMap + np.vsplit(
                    np.pad(
                        heatMapChannelArray[i_c].squeeze(),
                        [(0, pad), (0, 0)],
                        mode="constant",
                        constant_values=0,
                    ),
                    packetNum,
                )
            importanceOfPackets = [
                np.sum(packetizedheatMap[i_p]) for i_p in range(len(packetizedheatMap))
            ]
            OrderedImportanceOfPacketsIndex = self.__getOrderedImportantPacketIndex(
                importanceOfPackets
            )
            # numOfPacketsToLose = math.floor(
            #     len(packetizedheatMap) * percOfPacketLoss / 100
            # )
            totalNumPackets = len(packetizedheatMap)

            if percOfPacketLoss != 0:
                obj = gbChannel.GBC(percOfPacketLoss / 100, 7)
                sim = obj.simulate(totalNumPackets)
                numOfPacketsToLose = (~sim).nonzero()[0].size
            else:
                sim = np.full((1,totalNumPackets),True)
                numOfPacketsToLose = 0

            if (
                # case == "Random_RSCorrected"
                # or
                case
                == "Random_RSCorrected_FECRemovesBOT"
            ):
                FECPacketCount = math.floor(totalNumPackets * fecPerc / 100)
                protectedPacketCount = math.floor(totalNumPackets * protectedPerc / 100)
                # unprotectedPacketCount = totalNumPackets - protectedPacketCount
                # if case == "Random_RSCorrected":
                #     packetsSent = packetsSent + FECPacketCount + totalNumPackets
                #     numOfPacketsToLose = math.floor(
                #         (FECPacketCount + totalNumPackets) * percOfPacketLoss / 100
                #     )
                #     packetsLost = packetsLost + numOfPacketsToLose
                if case == "Random_RSCorrected_FECRemovesBOT":
                    lowestImportanceIndex = OrderedImportanceOfPacketsIndex[
                        -FECPacketCount:
                    ]
                    OrderedImportanceOfPacketsIndex = OrderedImportanceOfPacketsIndex[
                        :-FECPacketCount
                    ]
                    for j in lowestImportanceIndex:
                        packetizedfmL[j][...] = 0
                    # unprotectedPacketCount = unprotectedPacketCount - len(
                    #     lowestImportanceIndex
                    # )
                    # if unprotectedPacketCount < 0:
                    #     print("Cannot delete more!\n")
                    #     return
                    packetsSent = packetsSent + totalNumPackets
                    packetsLost = packetsLost + numOfPacketsToLose

                newPacketizedFECChannel = []
                # for i_p in range(unprotectedPacketCount):
                #     newPacketizedFECChannel.append(0)  # Unprotected Packets
                for i_p in range(protectedPacketCount):
                    newPacketizedFECChannel.append(1)  # Protected Top Packets k
                for i_p in range(FECPacketCount):
                    newPacketizedFECChannel.append(2)  # Redundant packets n-k
                rng.shuffle(newPacketizedFECChannel)
                lostPackets = newPacketizedFECChannel[0:numOfPacketsToLose]
                # lostUnprotectedPackets = lostPackets.count(0)
                lostProtectedPackets = lostPackets.count(1)
                lostRedundantPackets = lostPackets.count(2)
                if (
                    lostProtectedPackets + lostRedundantPackets <= FECPacketCount
                ):  # RECOVERABLE no protected part will be lost only unprotected
                    pass
                    # unprotectedPackets = OrderedImportanceOfPacketsIndex[
                    #     protectedPacketCount:
                    # ]
                    # rng.shuffle(unprotectedPackets)
                    # indexOfLossedPackets = unprotectedPackets[0:lostUnprotectedPackets]
                else:  # CANNOT RECOVER,lostProtectedPackets valid
                    # unprotectedPackets = OrderedImportanceOfPacketsIndex[
                    #     protectedPacketCount:
                    # ]
                    protectedPackets = OrderedImportanceOfPacketsIndex[
                        0:protectedPacketCount
                    ]
                    # rng.shuffle(unprotectedPackets)
                    rng.shuffle(protectedPackets)
                    indexOfLossedPackets = (
                        # unprotectedPackets[:lostUnprotectedPackets]+
                        protectedPackets[:lostProtectedPackets]
                    )
            elif case == "Unprotected (Burst)":
                packetsSent = packetsSent + totalNumPackets
                # indexOfLossedPackets = list(range(0, totalNumPackets))
                indexOfLossedPackets = (~sim).nonzero()[0]
                # rng.shuffle(indexOfLossedPackets)
                # indexOfLossedPackets = indexOfLossedPackets[0:numOfPacketsToLose]
                packetsLost = packetsLost + len(indexOfLossedPackets)

            elif case == "Unprotected (IID)":
                packetsSent = packetsSent + totalNumPackets
                indexOfLossedPackets = list(range(0, totalNumPackets))
                rng.shuffle(indexOfLossedPackets)
                indexOfLossedPackets = indexOfLossedPackets[0:numOfPacketsToLose]
                packetsLost = packetsLost + len(indexOfLossedPackets)

            elif case == "Most important":
                packetsSent = packetsSent + totalNumPackets
                indexOfLossedPackets = OrderedImportanceOfPacketsIndex[
                    0:numOfPacketsToLose
                ]
                packetsLost = packetsLost + len(indexOfLossedPackets)
            elif case == "Least important":
                packetsSent = packetsSent + totalNumPackets
                OrderedImportanceOfPacketsIndex = OrderedImportanceOfPacketsIndex[::-1]
                indexOfLossedPackets = OrderedImportanceOfPacketsIndex[
                    0:numOfPacketsToLose
                ]
                packetsLost = packetsLost + len(indexOfLossedPackets)
            else:
                raise Exception("Case can only be Random,Top or Random_RSCorrected.")

            for j in indexOfLossedPackets:
                packetizedfmL[j][...] = 0

            channelReconstructed = [
                np.vstack(packetizedfmL[i : i + packetNum])
                for i in range(0, len(packetizedfmL), packetNum)
            ]
            if pad != 0:
                channelReconstructed = [
                    channelReconstructed[i][0:-pad, ...]
                    for i in range(0, len(channelReconstructed))
                ]
            packetizedImportanceMap = [
                np.ones_like(packetizedheatMap[i_p]) * np.sum(packetizedheatMap[i_p])
                for i_p in range(len(packetizedheatMap))
            ]
            channelReconstructedImportance = [
                np.vstack(packetizedImportanceMap[i : i + packetNum])
                for i in range(0, len(packetizedImportanceMap), packetNum)
            ]
            if pad != 0:
                channelReconstructedImportance = [
                    channelReconstructedImportance[i][0:-pad, ...]
                    for i in range(0, len(channelReconstructedImportance))
                ]
            tensorImportanceCompleted = np.dstack(channelReconstructedImportance)
            tensorCompleted = np.dstack(channelReconstructed)
            fmL = self.__inverseQuantize(tensorCompleted, qBits, minVal, maxVal)
            fmLPacketizedLoss.append(fmL)
            importancePacketsTensor.append(tensorImportanceCompleted)

        if saveImages:
            self.__savePacketLossImages(fmLPacketizedLoss, case, modelName)
            self.__savePacketLossImages(
                self.__normalizeToUnit(np.array(importancePacketsTensor)),
                "packetImportance",
                modelName,
            )
            return

        # scores.append(self.getMetrics(fmLPacketizedLoss))
        metrics = self.getMetrics(fmLPacketizedLoss)

        # with open("Korcan/Plots/"+modelName+"/"+case+"_"+str(packetNum)+"_"+str(percOfPacketLoss)+".npy", 'wb') as f:
        #         np.save(f, np.array(scores))

        # acc = 0
        # loss = 0
        # count = 0
        # for s in scores:
        #     count = count + 1
        #     acc = acc + s["acc"]
        #     loss = loss + s["loss"]
        if case == "Most important" or case == "Least important" or case == "Unprotected (IID)" or case == "Unprotected (Burst)":
            if not os.path.exists("Korcan/Plots/" + modelName + "/" + case):
                os.makedirs("Korcan/Plots/" + modelName + "/" + case)

            pdictKey = ("{:.3f}".format(percOfPacketLoss), case)
            pdictVal = {"acc": metrics["acc"], "loss": metrics["loss"], "min":0,"max":0}

            rand = int(random.randint(1, sys.maxsize))
            with open(
                "Korcan/Plots/"
                + modelName
                + "/"
                + case
                + "/key_"
                + "{:.3f}".format(percOfPacketLoss)
                + "_"
                + str(rand)
                + "_.pkl",
                "wb",
            ) as f:
                pickle.dump(pdictKey, f)
            with open(
                "Korcan/Plots/"
                + modelName
                + "/"
                + case
                + "/val_"
                + "{:.3f}".format(percOfPacketLoss)
                + "_"
                + str(rand)
                + "_.pkl",
                "wb",
            ) as f:
                pickle.dump(pdictVal, f)
        else:
            if not os.path.exists(
                "Korcan/Plots/"
                + modelName
                + "/"
                + case
                + "_"
                + str(fecPerc)
                + "_"
                + str(protectedPerc)
            ):
                os.makedirs(
                    "Korcan/Plots/"
                    + modelName
                    + "/"
                    + case
                    + "_"
                    + str(fecPerc)
                    + "_"
                    + str(protectedPerc)
                )

            pdictKey = ("{:.3f}".format(100 * packetsLost / packetsSent), case)
            pdictVal = {"acc": metrics["acc"], "loss": metrics["loss"], "min":0,"max":0}

            rand = int(random.randint(1, sys.maxsize))
            with open(
                "Korcan/Plots/"
                + modelName
                + "/"
                + case
                + "_"
                + str(fecPerc)
                + "_"
                + str(protectedPerc)
                + "/key_"
                + "{:.3f}".format(100 * packetsLost / packetsSent)
                + "_"
                + str(rand)
                + "_.pkl",
                "wb",
            ) as f:
                pickle.dump(pdictKey, f)
            with open(
                "Korcan/Plots/"
                + modelName
                + "/"
                + case
                + "_"
                + str(fecPerc)
                + "_"
                + str(protectedPerc)
                + "/val_"
                + "{:.3f}".format(100 * packetsLost / packetsSent)
                + "_"
                + str(rand)
                + "_.pkl",
                "wb",
            ) as f:
                pickle.dump(pdictVal, f)
        # self.pdict["{:.3f}".format(100 * packetsLost / packetsSent),case,] = {"acc": metrics["acc"], "loss": metrics["loss"]}


if __name__ == "__main__":
    # modelName = "efficientnetb0"
    # splitLayer = "block2b_add"
    # modelName = "resnet18"
    # splitLayer = "add_1"
    # modelName = "dense"
    # splitLayer = "pool2_conv"

    modelName = "resnet"
    splitLayer = "conv2_block1_add"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    modelPath = "deep_models_full/" + modelName + "_model.h5"
    mobile_model_path = (
        "deep_models_split/" + modelName + "_" + splitLayer + "_mobile_model.h5"
    )
    cloud_model_path = (
        "deep_models_split/" + modelName + "_" + splitLayer + "_cloud_model.h5"
    )

    trained_model_path = "/local-scratch/localhome/kuyanik/UnequalLossProtectionDeepFeatures_CI/model.05-0.00.h5"
    dataName = "/local-scratch/localhome/kuyanik/dataset/smallTest"
    # trained_model_path = "/project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/model.05-0.00.h5"
    # dataName = "/home/foniks/projects/def-ibajic/foniks/Project_1/largeTest"
    quantizationBits = 8

    # CREATE FOLDERS
    if not os.path.exists("Korcan/Plots"):
        os.makedirs("Korcan/Plots")
    if not os.path.exists("Korcan/Plots/" + modelName):
        os.makedirs("Korcan/Plots/" + modelName)

    module = pipeline()
    module.loadModel(
        modelPath, mobile_model_path, cloud_model_path, trained_model_path, splitLayer
    )
    module.loadData(dataName, [224, 224], False)
    module.findHeatmaps(splitLayer, modelName, dataName)

    packetCount = 8
    percLoss = int(sys.argv[1])
    case = sys.argv[2]

    if case == "1":
        case = "Most important"
    elif case == "2":
        case = "Least important"
    elif case == "3":
        case = "Unprotected (IID)"
    elif case == "4":
        case = "Unprotected (Burst)"
    # module.saveSuperImposedChannels(modelName)

    # saveImageLossPercent = 40
    # module.packetLossSim(
    #     packetCount,
    #     quantizationBits,
    #     saveImageLossPercent,
    #     "Most important",
    #     saveImages=True,
    #     modelName=modelName,
    # )
    # module.packetLossSim(
    #     packetCount,
    #     quantizationBits,
    #     saveImageLossPercent,
    #     "Least important",
    #     saveImages=True,
    #     modelName=modelName,
    # )
    # module.packetLossSim(
    #     packetCount,
    #     quantizationBits,
    #     saveImageLossPercent,
    #     "Unprotected (IID)",
    #     saveImages=True,
    #     modelName=modelName,
    # )

    # # module.packetLossSim(
    # #     packetCount,
    # #     quantizationBits,
    # #     saveImageLossPercent,
    # #     "Random_RSCorrected",
    # #     40,
    # #     60,
    # #     saveImages=True,
    # #     modelName=modelName,
    # # )
    # module.packetLossSim(
    #     packetCount,
    #     quantizationBits,
    #     saveImageLossPercent,
    #     "Random_RSCorrected_FECRemovesBOT",
    #     40,
    #     60,
    #     saveImages=True,
    #     modelName=modelName,
    # )

    if case == "Most important" or case == "Least important" or case == "Unprotected (IID)" or case == "Unprotected (Burst)":
        module.packetLossSim(packetCount, 8, percLoss, case, modelName=modelName)
    elif case == "makeplot":
        dirs = [
            name
            for name in os.listdir("Korcan/Plots/" + modelName)
            if os.path.isdir("Korcan/Plots/" + modelName)
        ]
        fpPairs = []
        for d in dirs:
            splitted = d.split("_")
            if len(splitted) == 5:
                fpPairs.append(splitted[3:])  # Random_RSCorrected_FECRemovesBOT_10_50

        dirNames = []
        dirNames.append("Most important")
        dirNames.append("Least important")
        # dirNames.append("TopG")
        # dirNames.append("BotG")
        for d in dirNames:
            listFiles = os.listdir("Korcan/Plots/" + modelName + "/" + d)
            for fname in listFiles:
                if fname[:3] == "key":
                    with open(
                        "Korcan/Plots/" + modelName + "/" + d + "/" + fname, "rb"
                    ) as f:
                        key = pickle.load(f)
                    with open(
                        "Korcan/Plots/" + modelName + "/" + d + "/" + "val" + fname[3:],
                        "rb",
                    ) as f:
                        val = pickle.load(f)

                    # if key[1] == "Most important":
                    #     key = list(key)
                    #     dp = "Most important"
                    #     key[1] = dp
                    #     key = tuple(key)
                    # elif key[1] == "Least important":
                    #     key = list(key)
                    #     dp = "Least important"
                    #     key[1] = dp
                    #     key = tuple(key)

                    # if(d == "Most important"):
                    #     key = list(key)
                    #     dp = "Most important (Proxy)"
                    #     key[1] = dp
                    #     key = tuple(key)
                    # elif(d == "Least important"):
                    #     key = list(key)
                    #     dp = "Least important (Proxy)"
                    #     key[1] = dp
                    #     key = tuple(key)
                    # elif(d == "TopG"):
                    #     key = list(key)
                    #     dp = "Most important (GradCAM)"
                    #     key[1] = dp
                    #     key = tuple(key)
                    # elif(d == "BotG"):
                    #     key = list(key)
                    #     dp = "Least important (GradCAM)"
                    #     key[1] = dp
                    #     key = tuple(key)

                    module.pdict[key] = val

        dirNames = []
        for fp in fpPairs:
            dirNames.append("Random_RSCorrected_FECRemovesBOT_" + fp[0] + "_" + fp[1])
        # dirNames.append("Random_RSCorrected_"+fp[0]+"_"+fp[1])
        dirNames.append("Unprotected (IID)")
        for d in dirNames:
            listFiles = os.listdir("Korcan/Plots/" + modelName + "/" + d)
            keyIndexes = []
            for i in range(len(listFiles)):
                if listFiles[i][:3] == "key":
                    keyIndexes.append(i)

            blacklist = []
            for i in range(len(keyIndexes)):
                lossPercInfo = listFiles[i].split("_")[1]
                if lossPercInfo not in blacklist:
                    allRunsWithSamePercentage = glob.glob(
                        "Korcan/Plots/"
                        + modelName
                        + "/"
                        + d
                        + "/key_"
                        + lossPercInfo
                        + "*"
                    )
                    blacklist.extend(allRunsWithSamePercentage)
                    val = []
                    for fname in allRunsWithSamePercentage:
                        with open(
                            "Korcan/Plots/"
                            + modelName
                            + "/"
                            + d
                            + "/"
                            + fname.split("/")[-1:][0],
                            "rb",
                        ) as f:
                            key = pickle.load(f)
                        with open(
                            "Korcan/Plots/"
                            + modelName
                            + "/"
                            + d
                            + "/"
                            + "val"
                            + fname.split("/")[-1:][0][3:],
                            "rb",
                        ) as f:
                            val.append(pickle.load(f))
                    acc = 0
                    loss = 0
                    count = 0
                    accList = []
                    for s in val:
                        count = count + 1
                        acc = acc + s["acc"]
                        accList.append(s["acc"])
                        loss = loss + s["loss"]
                    min = np.amin(np.array(accList))
                    max = np.amax(np.array(accList))
                    # print(count)
                    # print(key)
                    # print("a____")
                    # print(val)
                    # print("b____")
                    # print({"acc": acc/count, "loss": loss/count})
                    if key[1] == "Random_RSCorrected_FECRemovesBOT":
                        key = list(key)
                        dp = "FEC" + d[-6:]
                        key[1] = dp
                        key = tuple(key)
                    else:
                        key = list(key)
                        key[1] = "Unprotected"
                        key = tuple(key)
                    module.pdict[key] = {"acc": acc / count, "loss": loss / count, "min": min ,"max": max}

        ##WILL BE HANDLED DIFFERENTELY COMING UP!
        module.makePlot(
            "Korcan/Plots/" + modelName + "/AccuracyPlotPacketized",
            "Korcan/Plots/" + modelName + "/LossPlotPacketized",
        )
    else:
        fecPercent = int(sys.argv[3])
        protectPercent = int(sys.argv[4])
        module.packetLossSim(
            packetCount,
            8,
            percLoss,
            case,
            fecPercent,
            protectPercent,
            modelName=modelName,
        )

    # module.findPercentileLossPerChannelFM(
    #     saveImageLossPercent, quantizationBits, saveImages=True
    # )
    # module.findPercentileLossPerChannelFM(
    #     saveImageLossPercent, quantizationBits, saveImages=True,bot=True
    # )
    # module.findPercentileRandomLossPerChannelFM(
    #     saveImageLossPercent, quantizationBits, saveImages=True
    # )

    # fecPercent = [10]
    # protectPercent = [20, 50, 80]
    # for f in fecPercent:
    #     for p in protectPercent:
    #         fecProtectInfo = "fec" + str(f) + "protect" + str(p)

    #         for percLoss in np.concatenate((np.linspace(0, 4, 2),np.linspace(4, 15, 2)),axis=None):
    #             module.packetLossSim(packetCount, 8, percLoss, "Most important",modelName=modelName)
    #             module.packetLossSim(packetCount, 8, percLoss, "Least important",modelName=modelName)
    #             # module.packetLossSim(packetCount, 8, percLoss,"Unprotected (IID)",modelName=modelName)
    #             # module.packetLossSim(
    #             #     packetCount, 8, percLoss, "Random_RSCorrected", f, p,modelName=modelName
    #             # )
    #             # module.packetLossSim(
    #             #     packetCount, 8, percLoss, "Random_RSCorrected_FECRemovesBOT", f, p,modelName=modelName
    #             # )
    #         for percLoss in np.linspace(15, 50, 2):
    #             # module.packetLossSim(packetCount, 8, percLoss, "Least important",modelName=modelName)
    #             # module.packetLossSim(
    #             #     packetCount, 8, percLoss, "Random_RSCorrected", f, p,modelName=modelName
    #             # )
    #             # module.packetLossSim(
    #             #     packetCount, 8, percLoss, "Random_RSCorrected_FECRemovesBOT", f, p,modelName=modelName
    #             # )
    #         # for percLoss in np.linspace(50, 100, 2):
    #         #     module.packetLossSim(packetCount, 8, percLoss, "Random_RSCorrected", fecPercent, protectPercent)
    #         module.makePlot(
    #             "Korcan/Plots/"+modelName+"/AccuracyPlotPacketized2_" + fecProtectInfo,
    #             "Korcan/Plots/"+modelName+"/LossPlotPacketized2_" + fecProtectInfo,
    #         )
    #         module.cleanPlot()
