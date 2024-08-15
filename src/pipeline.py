import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib as mpl
import pickle
import scipy
import cv2 as cv
import copy

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
        print("Korcan Uyanik")
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
            matrixFeature = np.empty((iS[1] * 16, iS[2] * 16))
            matrixHeat = np.empty((iS[1] * 16, iS[2] * 16))
            ind = 0
            heatMap = self.__normalizeToUnit(heatMapBatch[i_b])
            for i_cx in range(16):
                for i_cy in range(16):
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
            matrixHeat = (matrixHeat - np.min(matrixHeat)) / (np.max(matrixHeat) - np.min(matrixHeat))
            matrixHeat = matrixHeat * 1.5 # HERE COMMENT
            matrixHeat[matrixHeat > 1] = 1
            # matrixHeat[matrixHeat < 0.8] = 0.6
            # matrixHeat[matrixHeat < 0.5] = 0.3
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
            self.dataset_x_files_sizes,
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
        # self.mobile_model.summary()
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

    def __savePacketLossImages(self, lossedTensorBatchArray,fmLPacketizedFEC, case, modelName):
        mainPath = os.path.abspath("Korcan/Plots/" + modelName + "/tensorLoss/" + case)
        if not os.path.exists(mainPath):
            os.makedirs(mainPath)
        for label in self.dataset_y_labels:
            if not os.path.exists(os.path.join(mainPath, label)):
                os.makedirs(os.path.join(mainPath, label))

        lossedTensorBatchArray = np.array(lossedTensorBatchArray).astype(np.float64)
        fmLPacketizedFEC = np.array(fmLPacketizedFEC).astype(np.bool_)
        shape = lossedTensorBatchArray.shape
        for i_b in range(len(lossedTensorBatchArray)):  # 9
            arrFEC = np.empty((shape[1] * 6, shape[2] * 4))
            arr = np.empty((shape[1] * 6, shape[2] * 4))
            ind = 0
            for i_cx in range(6):
                for i_cy in range(4):
                    # lossedTensorBatchArray[i_b, :, :, ind] = self.__normalizeToUnit(
                    #     lossedTensorBatchArray[i_b, :, :, ind]
                    # )
                    arrFEC[
                        i_cx * 56 : i_cx * 56 + 56, i_cy * 56 : i_cy * 56 + 56
                    ] = fmLPacketizedFEC[i_b, :, :, ind]
                    arr[
                        i_cx * 56 : i_cx * 56 + 56, i_cy * 56 : i_cy * 56 + 56
                    ] = lossedTensorBatchArray[i_b, :, :, ind]
                    ind = ind + 1
            plt.imshow(arr, cmap="gray",interpolation="none")
            plt.axis("off")
            plt.savefig(
                os.path.join(
                    mainPath, self.dataset_y_labels[i_b], self.file_names[i_b]
                ),
                bbox_inches="tight",
                dpi=400,
            )

            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_cmap", [(1, 0, 0, 0), (1, 0, 0, 1)])
            # Set the under value (0) to fully transparent
            # Display the data using the custom colormap
            plt.imshow(arrFEC, cmap=cmap,interpolation="none")

            plt.axis("off")
            plt.savefig(
                os.path.join(
                    mainPath, self.dataset_y_labels[i_b], "FECHighlight"+self.file_names[i_b]
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
            "Most important NS",
            "b",
            ".",
            ":",
            "Least important NS",
            "g",
            ".",
            ":",
            "Most important Weighted",
            "r",
            ".",
            "-",
            "Least important Weighted",
            "c",
            ".",
            "-",
            "Most important NS Weighted",
            "r",
            ".",
            ":",
            "Least important NS Weighted",
            "c",
            ".",
            ":",

            "Unprotected (IID)",
            "r",
            ".",
            "-",
            "Unprotected (Burst)",
            "y",
            ".",
            "-",
            "Unprotected (IID) NS",
            "r",
            ".",
            ":",
            "Unprotected (Burst) NS",
            "y",
            ".",
            ":",

            "FEC (IID)",
            "c",
            ".",
            "-",
            "FEC (Burst)",
            "y",
            ".",
            "-",
            "FEC (IID) NS",
            "c",
            ".",
            ":",
            "FEC (Burst) NS",
            "y",
            ".",
            ":",

            "FEC (IID) Weighted",
            "k",
            ".",
            "-",
            "FEC (IID) NS Weighted",
            "k",
            ".",
            ":",
        ]

        l = []
        for i in range(0, 115, 5):
            l.extend(["Unprotected (IID) EN_" + str(i), "k", ".", ":"])
        cases.extend(l)
        types = sorted(list(set([i[1] for i in self.pdict.keys()])))
        print("Types", types)
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
        plt.title("Top1 Accuracy")

        plt.xlabel("Percent Lost")
        # plt.ylabel("MSE")

        for s in range(len(seriesX)):
            if types[s] in l:
                plt.xlabel("BPP")
            mapping = cases.index(types[s])
            _, seriesYmax[s] = zip(*sorted(zip(seriesX[s], seriesYmax[s])))
            _, seriesYmin[s] = zip(*sorted(zip(seriesX[s], seriesYmin[s])))
            seriesX[s], seriesY[s] = zip(*sorted(zip(seriesX[s], seriesY[s])))
            if types[s] in l[-4:]:
                print(l[-4:])
                cases[-3:-2] = "r"
                print(l[-4:])
                plt.axhline(y=seriesY[s], color="red", linestyle="--")
            if types[s] in l[-8:-4]:
                cases[-7:-6] = "b"
                # plt.axhline(y=seriesY[s], color="blue", linestyle="--")
            # if (
            #     types[s] == "Least important"
            #     or types[s] == "Most important"
            #     or types[s] == "Unprotected (IID)"
            #     or types[s] == "Unprotected (Burst)"
            #     or types[s] == "Unprotected (IID) NS"
            #     or types[s] == "Unprotected (IID) EN"
            #     or types[s] == "Unprotected (Burst) NS"
            #     or types[s] == "FEC (IID)"
            #     or types[s] == "FEC (Burst)"
            # ):
            plt.scatter(
                seriesX[s],
                seriesY[s],
                s=25,
                # label="_nolegend_",
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
            # else:
            #     print(types[s])
            #     plt.scatter(
            #         seriesX[s],
            #         seriesY[s],
            #         s=25,
            #         marker=cases[mapping + 2],
            #         color=cases[mapping + 1],
            #     )
            #     plt.plot(
            #         seriesX[s],
            #         seriesY[s],
            #         label=cases[mapping],
            #         linestyle=cases[mapping + 3],
            #         linewidth=1.2,
            #         color=cases[mapping + 1],
            # )
            if types[s] == "FEC (IID) NS Weighted":
                plt.fill_between(
                    seriesX[s], seriesYmin[s], seriesYmax[s], alpha=0.3, facecolor="k"
                )
            elif types[s] == "FEC (IID) NS":
                plt.fill_between(
                    seriesX[s], seriesYmin[s], seriesYmax[s], alpha=0.4, facecolor="c"
                )

        # reordering the labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # specify order
        # order=[0,2,1,3]
        order=[0,3,6,4,5,1,2,7,]

        plt.legend(
            [handles[i] for i in order], [labels[i] for i in order],
            loc="upper right",
            fontsize="xx-small",
            markerscale=0.7,
            # ncol=2,
            fancybox=True,
            # shadow=True,
            prop={"size": 5},
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

    def myImportanceFunction(self, pfmL, packetNum):
        packetizedfmL = copy.deepcopy(pfmL)
        for i in range(packetizedfmL):
            packetizedfmL = copy.deepcopy(pfmL)
            packetizedfmL[i] = 0
            channelReconstructed = [
                np.vstack(packetizedfmL[i : i + packetNum])
                for i in range(0, len(packetizedfmL), packetNum)
            ]
            tensorCompleted = np.dstack(channelReconstructed)
            for ind in range(tensorCompleted.shape[2]):
                img_gray_cv2 = cv.cvtColor(
                    tensorCompleted[:, :, ind].astype("uint8"), cv.COLOR_GRAY2BGR
                )
                dst = cv.inpaint(
                    img_gray_cv2,
                    maskCompleted[:, :, ind].astype("uint8"),
                    7,
                    cv.INPAINT_TELEA,
                )
                dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
                tensorCompleted[:, :, ind] = dst

        return

    def find_nearest_index(self, array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array)):
            return idx - 1
        else:
            return idx

    def fn_caltec(self, lossMatrix, packetizedfmL,pktzNoLoss,extra):
        # figure out the number of channels in the tensor, the dimensionality of a
        # channel, the number of packets in the channel.
        num_channels = 256
        channel_width = 56
        rowsPerPacket = 7
        num_pkts_per_channel = 8
        pkt_obj = packetizedfmL
        # lost_map = lossMatrix[item_index,:,:]
        lost_map = lossMatrix
        lostChannels = []
        if np.size(lost_map) - np.count_nonzero(lost_map) == 0:
            print(f"No packets lost in item .")
            return pkt_obj
        # ------------------------------------------------------------------------ #
        for i_c in range(num_channels):
            if np.all(lost_map[i_c, :] == False) == True:
                # The entire channel has been knocked out. cannot recover from that,
                # so ignore this damaged channel and continue.
                print(
                    f"All packets were lost in this channel. Cannot repair channel {i_c}"
                )
                lostChannels.append(i_c)
                continue
            # -------------------------------------------------------------------- #
            for i_pkt in range(num_pkts_per_channel):
                if lost_map[i_c, i_pkt] == False:
                    # print(f"Repairing packet {i_pkt} in channel {i_c}")
                    existing_colocated_pkts_list = np.sort(
                        np.where(lost_map[:, i_pkt] == True)[0]
                    )
                    # print('list of existing colocated packets in other channels')
                    # print(existing_colocated_pkts_list)

                    existing_pkts_in_channel_list = np.sort(
                        np.where(lost_map[i_c, :] == True)[0]
                    )
                    nearest_neighbor_in_channel_index = self.find_nearest_index(
                        existing_pkts_in_channel_list, i_pkt
                    )

                    for test_pkt_id in range(len(existing_pkts_in_channel_list)):
                        nearest_neighbor_in_channel_idx = existing_pkts_in_channel_list[
                            nearest_neighbor_in_channel_index - test_pkt_id
                        ]
                        existing_colocated_pkts_neighbor_list = np.where(
                            lost_map[:, nearest_neighbor_in_channel_idx] == True
                        )[0]

                        candidate_channels = np.intersect1d(
                            existing_colocated_pkts_list,
                            existing_colocated_pkts_neighbor_list,
                        )
                        if len(candidate_channels) != 0:
                            # print('found a candidate')
                            break
                    if len(candidate_channels) == 0:
                        print(f"No candidate found. Cannot repair packet {i_pkt}")
                        continue

                    candidate_channels = np.hstack(([i_c], candidate_channels))

                    # print("The candidate channels for the correlation test are")
                    # print(candidate_channels)

                    nearest_neighbor_pkt = pkt_obj[
                        i_c, nearest_neighbor_in_channel_idx, :
                    ]
                    corrcoeff_matrix = np.zeros(
                        [len(candidate_channels), len(candidate_channels)]
                    )
                    # for i_row in range(rowsPerPacket):
                    #    corrcoeff_matrix += np.corrcoef([pkt_obj.packet_seq[item_index,nearest_neighbor_in_channel_idx,i_row,:,i] for i in candidate_channels])

                    corrcoeff_matrix = np.corrcoef(
                        [
                            np.reshape(
                                pkt_obj[i, nearest_neighbor_in_channel_idx, :],
                                (rowsPerPacket * channel_width),
                            )
                            for i in candidate_channels
                        ]
                    )
                    row_corrcoef_below = corrcoeff_matrix[0, :]
                    idx = np.argpartition(row_corrcoef_below, -2)[-2:]
                    indices_below = idx[np.argsort((-row_corrcoef_below)[idx])]
                    # print(f"The highest correlated channel is {candidate_channels[indices_below[1]]}")

                    # select colocated packet which gives the second maximum value in corrcoeff_matrix.
                    pkt_from_other_channel = pkt_obj[
                        candidate_channels[indices_below[1]], i_pkt, :
                    ]
                    neighbor_out_channel = pkt_obj[
                        candidate_channels[indices_below[1]],
                        nearest_neighbor_in_channel_idx,
                        :,
                    ]

                    # reshape both neighbor packets into vectors and then run least squares.
                    # vec_in_channel = np.reshape(nearest_neighbor_pkt,(np.shape(nearest_neighbor_pkt)[0]*np.shape(nearest_neighbor_pkt)[1]))
                    # vec_out_channel = np.reshape(neighbor_out_channel,(np.shape(neighbor_out_channel)[0]*np.shape(neighbor_out_channel)[1]))

                    vec_in_channel = nearest_neighbor_pkt
                    vec_out_channel = neighbor_out_channel

                    lumi_transf = np.polyfit(vec_out_channel, vec_in_channel, 1)
                    lumi_transf_fn = np.poly1d(lumi_transf)

                    # vec_corrected = lumi_transf_fn(np.reshape(pkt_from_other_channel,(np.shape(pkt_from_other_channel)[0]*np.shape(pkt_from_other_channel)[1])))
                    # pkt_corrected_1 = np.reshape(vec_corrected,(np.shape(pkt_from_other_channel)))
                    vec_corrected = lumi_transf_fn(pkt_from_other_channel)
                    pkt_corrected_1 = vec_corrected
                    print(pkt_corrected_1.shape)
                    pkt_obj[i_c, i_pkt, :] = pkt_corrected_1
                    # print(f'Packet {i_pkt} in channel {i_c} repaired.')

        if extra == True:
            for i_c in lostChannels:
                mse_values = []
                for a_c in range(num_channels):
                    mse = np.mean((pktzNoLoss[i_c, :, :] - pkt_obj[a_c, :, :])**2)
                    mse_values.append(mse)
                    # print(i_c, a_c,mse)
                    # print(pktzNoLoss[i_c, :, :],pkt_obj[a_c, :, :])
                # Find the index of the most similar matrix based on the lowest MSE value
                most_similar_index = np.argmin(mse_values)
                # print(i_c, most_similar_index)
                pkt_obj[i_c, :, :] = pkt_obj[most_similar_index, :, :]

        # for i_c in lostChannels:
        #     mse_values = [np.mean((pktzNoLoss[i_c,:,:] - pkt_obj[a_c, :, :])**2) for a_c in lostChannels]
        #     # Find the index of the most similar matrix based on the lowest MSE value
        #     print(mse_values)
        #     most_similar_index = np.argmin(mse_values)
        #     print(i_c,most_similar_index)
        #     pkt_obj[i_c, :, :] = pkt_obj[most_similar_index, :, :]
        #     corrcoeff_matrix = np.corrcoef(
        #     [
        #         np.reshape(
        #             pkt_obj[i, :, :],
        #             (rowsPerPacket * channel_width),
        #         )
        #         for i in range(num_channels)
        #     ]
        # )
        #     row_corrcoef_below = corrcoeff_matrix[0, :]
        #     idx = np.argpartition(row_corrcoef_below, -2)[-2:]
        #     indices_below = idx[np.argsort((-row_corrcoef_below)[idx])]
        #     # print(f"The highest correlated channel is {candidate_channels[indices_below[1]]}")
        return pkt_obj

    def rescale(self,values, new_min, new_max):
        # Find the current minimum and maximum values in the input list
        current_min = min(values)
        current_max = max(values)

        # Calculate the scaling factor for the range
        scale_factor = (new_max - new_min) / (current_max - current_min)

        # Rescale the values
        rescaled_values = [new_min + (x - current_min) * scale_factor for x in values]

        return rescaled_values

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
        qualityFactor=80,
    ):
        # iteration = 1
        # if(case == "Unprotected (IID)" or case == "Random_RSCorrected" or case == "FEC"):
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
        fmLPacketizedFEC = []
        mseList = []
        importancePacketsTensor = []
        rng = np.random.default_rng()
        packetsLost = 0
        packetsSent = 0

        batchBpp = []
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
                np.mean(packetizedheatMap[i_p]) for i_p in range(len(packetizedheatMap))
            ]
            importanceOfPacketsMin = np.min(importanceOfPackets)
            importanceOfPacketsMax = np.max(importanceOfPackets)
            importanceOfPackets = (importanceOfPackets - importanceOfPacketsMin) / (importanceOfPacketsMax - importanceOfPacketsMin)
            OrderedImportanceOfPacketsIndex = (
                self.__getOrderedImportantPacketIndex(importanceOfPackets)
            )

            importanceOfPacketsWeighted=importanceOfPackets.copy()

            if fecPerc is not None:

            #Method A
                # NoImportanceNotFECIndex = OrderedImportanceOfPacketsIndex[math.floor(len(importanceOfPackets)*40/100):math.ceil(len(importanceOfPackets)*60/100)]
                # NoImportanceFECIndex = OrderedImportanceOfPacketsIndex[math.floor(len(importanceOfPackets)*40/100):]
                # Start10PercentIndex = OrderedImportanceOfPacketsIndex[:math.ceil(len(importanceOfPackets)*10/100)]

                # Start20PercentNeighborsIndex = []
                # for number in Start10PercentIndex:
                #     if number > 0:
                #         Start20PercentNeighborsIndex.append(number - 1)
                #         Start20PercentNeighborsIndex.append(number + 1)
                #     else:
                #         Start20PercentNeighborsIndex.append(number + 1)

                # # Filter out negative values and duplicates
                # Start20PercentNeighborsIndex = list(set(filter(lambda x: x >= 0, Start20PercentNeighborsIndex)))

                # # Sort the result list
                # Start20PercentNeighborsIndex.sort()


                # maximumI=0
                # for i in NoImportanceNotFECIndex:
                #     if maximumI < importanceOfPackets[i]:
                #         maximumI = importanceOfPackets[i]

                # for i in Start20PercentNeighborsIndex:
                #     correlation_coefficient = 0
                #     indexHigher = i
                #     for j in NoImportanceFECIndex:
                #         if correlation_coefficient < np.corrcoef(packetizedfmL[i].flatten(), packetizedfmL[j].flatten())[0, 1]:
                #             correlation_coefficient = np.corrcoef(packetizedfmL[i].flatten(), packetizedfmL[j].flatten())[0, 1]
                #             indexHigher = j
                #     importanceOfPacketsWeighted[indexHigher] = maximumI

                # Method B
                # importanceOfChannels = []
                # for i in range(0, len(importanceOfPackets), 8):
                #     group_sum = sum(importanceOfPackets[i:i+8])
                #     importanceOfChannels.append(group_sum)

                # OrderedImportanceOfChannelsIndex = (
                #     self.__getOrderedImportantPacketIndex(importanceOfChannels)
                # )

                # importantChannelIndex = OrderedImportanceOfChannelsIndex[:math.floor(len(importanceOfChannels)/100*20)]
                # notimportantChannelIndex = OrderedImportanceOfChannelsIndex[math.floor(len(importanceOfChannels)/100*60):]

                # mse_matrix = np.zeros((self.C,self.C))
                
                
                # for i in notimportantChannelIndex:
                #     for j in importantChannelIndex:
                #         corr_coefficient = np.corrcoef(fmL[:,:,i].flatten(), fmL[:,:,j].flatten())[0, 1]
                #         mse_matrix[i, j] = abs(corr_coefficient)
                        
                # max_indices = np.argmax(mse_matrix, axis=1)
                # # sum_indices = np.argwhere(np.sum(mse_matrix, axis=1) == 0)
                # # complement = np.setdiff1d(max_indices, sum_indices)
                # for i in range(len(notimportantChannelIndex)):
                #     fmL[:,:,notimportantChannelIndex[i]] = fmL[:,:,max_indices[i]]

                # fmLChannelArray = np.dsplit(fmL, self.C)
                # packetizedfmL = []
                # for i_c in range(self.C):
                #     packetizedfmL = packetizedfmL + np.vsplit(
                #         np.pad(
                #             fmLChannelArray[i_c].squeeze(),
                #             [(0, pad), (0, 0)],
                #             mode="constant",
                #             constant_values=0,
                #         ),
                #         packetNum,
                #     )    

                # mse_matrix = np.sum(mse_matrix, axis=1)
                # min_mse = np.min(mse_matrix)
                # max_mse = np.max(mse_matrix)
                # channel_sim_scores = 1 - (mse_matrix - min_mse) / (max_mse - min_mse)
                # channel_sim_scores = [x for x in channel_sim_scores for _ in range(8)]
                # # importanceOfPacketsWeighted = channel_sim_scores
                # importanceOfPacketsWeighted =importanceOfPackets+np.array(channel_sim_scores)*0.5

                # OrderedChannelList = [value // 8 for value in OrderedImportanceOfPacketsIndex]
                # OrderedChannelListF= OrderedChannelList[:len(OrderedChannelList)//2]
                # OrderedChannelListF = list(set(OrderedChannelListF))
                # OrderedChannelListS= OrderedChannelList[len(OrderedChannelList)//2:]
                # OrderedChannelListS = list(set(OrderedChannelListS))

                # if len(OrderedChannelListF) <= len(OrderedChannelListS):
                #     for i in range(len(OrderedChannelListS)):
                #         fmL[:,:,OrderedChannelListS[i]] = fmL[:,:,OrderedChannelListF[i%len(OrderedChannelListF)]]
                # else:
                #     for i in range(len(OrderedChannelListS)):
                #         fmL[:,:,OrderedChannelListS[i]] = fmL[:,:,OrderedChannelListF[i]]

                # fmLChannelArray = np.dsplit(fmL, self.C)
                # packetizedfmL = []
                # for i_c in range(self.C):
                #     packetizedfmL = packetizedfmL + np.vsplit(
                #         np.pad(
                #             fmLChannelArray[i_c].squeeze(),
                #             [(0, pad), (0, 0)],
                #             mode="constant",
                #             constant_values=0,
                #         ),
                #         packetNum,
                #     )  
                # 
                # 
                # 

                # Method C
                # NoImportanceIndex = OrderedImportanceOfPacketsIndex[math.floor(len(importanceOfPackets)*50/100):]
                # maxX = min(importanceOfPackets[NoImportanceIndex]) - 1
                # selected_ind = [i for i in random.sample(NoImportanceIndex,math.floor(len(importanceOfPackets)*30/100))]
                # importanceOfPacketsWeighted[selected_ind] = maxX
                # Method D
                # NoImportanceIndex = OrderedImportanceOfPacketsIndex[math.floor(len(importanceOfPackets)*50/100):]
                # maxX = min(importanceOfPackets[NoImportanceIndex]) - 1
                # bin_size = 8
                # # Initialize a dictionary to store the bins
                # bins = {}
                # # Iterate through the random indices and assign them to bins
                # for index in NoImportanceIndex:
                #     bin_number = index // bin_size
                #     if bin_number not in bins:
                #         bins[bin_number] = []
                #     bins[bin_number].append(index)

                # # Convert the bins dictionary to a list
                # bins_list = [indices for _, indices in sorted(bins.items())]
                # sorted_bins = sorted(bins_list, key=len, reverse=True)
                # combined_list = [index for indices in sorted_bins for index in indices]
                # importanceOfPacketsWeighted[combined_list[:math.floor(len(importanceOfPackets)*30/100)]] = maxX

                # Method E
                # NoImportanceIndex = OrderedImportanceOfPacketsIndex[math.floor(len(importanceOfPackets)*50/100):]
                # maxX = min(importanceOfPackets[NoImportanceIndex]) - 1
                # NoImportanceIndex.sort()
                # step = len(NoImportanceIndex) / (math.floor(len(importanceOfPackets)*70/100)- 1)  # Calculate the step size
                # selected_indices = []
                # for i in range(math.floor(len(importanceOfPackets) * 70/100)):
                #     index = int(i * step)
                #     if 0 <= index < len(NoImportanceIndex):
                #         selected_indices.append(NoImportanceIndex[index])
                #     else:
                #         break  # Break the loop if index is out of bounds
                # importanceOfPacketsWeighted[selected_indices] = maxX

                # Method F
                NoImportanceIndex = OrderedImportanceOfPacketsIndex[math.floor(len(importanceOfPackets)*50/100):]
                TopImportanteIndex = OrderedImportanceOfPacketsIndex[:math.floor(len(importanceOfPackets)*30/100)]
                mse_matrix = np.zeros((len(importanceOfPackets),len(importanceOfPackets)))
                
                
                for i in TopImportanteIndex:
                    for j in NoImportanceIndex:
                        # mse = np.mean((packetizedfmL[i] - packetizedfmL[j])**2)
                        if(i%packetNum == j%packetNum):
                            mse = np.corrcoef(packetizedfmL[i].flatten(), packetizedfmL[j].flatten())[0, 1]
                            mse_matrix[i, j] = abs(mse)
                        
                # max_indices = np.argmax(mse_matrix, axis=1)
                # # sum_indices = np.sum(mse_matrix, axis=1)
                # # complement = np.setdiff1d(max_indices, sum_indices)
                # maxX = max(importanceOfPackets[NoImportanceIndex])
                # importanceOfPacketsWeighted[max_indices] = maxX

                mse_matrix = np.sum(mse_matrix, axis=1)
                min_mse = np.min(mse_matrix)
                max_mse = np.max(mse_matrix)
                channel_sim_scores = 1 - (mse_matrix - min_mse) / (max_mse - min_mse)
                # channel_sim_scores = [x for x in channel_sim_scores for _ in range(8)]
                # importanceOfPacketsWeighted = channel_sim_scores
                importanceOfPacketsWeighted =importanceOfPackets+np.array(channel_sim_scores)*0.6
                importanceOfPacketsWeighted[TopImportanteIndex] = 1
##################################################################
                # inSameChannelMatrix = np.zeros((len(importanceOfPackets),len(importanceOfPackets)))
                # for i in range(importanceOfPackets.shape[0]):
                #     # Access the i-th row of importanceOfPackets
                #     score = 1
                #     # Assign values to the corresponding row in inSameChannelMatrix
                #     inSameChannelMatrix[i, :] = score
##################################################################
                #Last one i did
                # ImportantPacketsIndex = OrderedImportanceOfPacketsIndex[:math.floor(len(importanceOfPackets)*10/100)]
                # NoImportanceIndex = OrderedImportanceOfPacketsIndex[math.floor(len(importanceOfPackets)*50/100):]

                # channelGlobalMap = np.zeros(( packetNum))
                # for i in range(len(importanceOfPackets)):
                #     channelGlobalMap[i% packetNum] = importanceOfPackets[i] + channelGlobalMap[i% packetNum]
                
                # channelGlobalMapMin = np.min(channelGlobalMap)
                # channelGlobalMapMax = np.max(channelGlobalMap)
                # channelGlobalMap = (channelGlobalMap - channelGlobalMapMin) / (channelGlobalMapMax - channelGlobalMapMin)


                # for i in NoImportanceIndex:
                #     importanceOfPacketsWeighted[i] = importanceOfPackets[i] + channelGlobalMap[i% packetNum] * 0.005
                # Method A1
                # NoImportanceIndex = OrderedImportanceOfPacketsIndex[math.floor(len(importanceOfPackets)*50/100):]
                # ImportanceIndex = OrderedImportanceOfPacketsIndex[:math.floor(len(importanceOfPackets)*50/100)]
                # NoImportanceIndexWeights = np.zeros_like(importanceOfPacketsWeighted)
                
                # for i in NoImportanceIndex:
                #     for j in ImportanceIndex:
                #         if j%packetNum == i%packetNum:
                #             corr_coefficient = np.corrcoef(packetizedfmL[i].flatten(), packetizedfmL[j].flatten())[0, 1]
                #             NoImportanceIndexWeights[i] = NoImportanceIndexWeights[i] + abs(corr_coefficient)

                # NoImportanceIndexWeightsMin = np.min(NoImportanceIndexWeights)
                # NoImportanceIndexWeightsMax = np.max(NoImportanceIndexWeights)
                # NoImportanceIndexWeights = (NoImportanceIndexWeights - NoImportanceIndexWeightsMin) / (NoImportanceIndexWeightsMax - NoImportanceIndexWeightsMin)

                # importanceOfPacketsWeighted[ImportanceIndex] = 3
                # for i in NoImportanceIndex:
                #     importanceOfPacketsWeighted[i] = importanceOfPacketsWeighted[i] + NoImportanceIndexWeights[i] * 0.5

                # Method A2
                # importanceOfPacketsSobel = []
                # for p in packetizedfmL:
                #     dx = scipy.ndimage.sobel(p, 1)
                #     dy = scipy.ndimage.sobel(p, 0)
                #     grad_magnitude = np.sqrt(dx**2 + dy**2)
                #     # grad_magnitude = np.sqrt(np.sum(np.square(gradients), axis=0))
                #     avg_grad_magnitude = np.mean(grad_magnitude)
                #     importanceOfPacketsSobel.append(avg_grad_magnitude)

                # importanceOfPacketsSobelMin = np.min(importanceOfPacketsSobel)
                # importanceOfPacketsSobelMax = np.max(importanceOfPacketsSobel)
                # importanceOfPacketsSobel = (importanceOfPacketsSobel - importanceOfPacketsSobelMin) / (importanceOfPacketsSobelMax - importanceOfPacketsSobelMin)
                
                # OrderedimportanceOfPacketsSobel = (
                #     self.__getOrderedImportantPacketIndex(importanceOfPacketsSobel)
                # )

                # NoImportanceIndex = OrderedImportanceOfPacketsIndex[math.floor(len(importanceOfPackets)*50/100):]


                # # maxX = max(importanceOfPackets[NoImportanceIndex])
                # # minX = min(importanceOfPackets[NoImportanceIndex])
                # # NoImportancePacketValues = importanceOfPacketsSobel[NoImportanceIndex]
                # # scaledSobelValues = self.rescale(NoImportancePacketValues,minX,maxX)
                # # importanceOfPacketsWeighted[NoImportanceIndex] = scaledSobelValues

                # ImportantPackets = OrderedImportanceOfPacketsIndex[:math.floor(len(importanceOfPackets)*30/100)]


                # alpha = 0.5
                # importanceOfPacketsWeighted[NoImportanceIndex] = importanceOfPackets[NoImportanceIndex] + alpha*importanceOfPacketsSobel[NoImportanceIndex]
                # importanceOfPacketsWeighted[ImportantPackets] = 1

                # Method A3
                # ImportantPackets = OrderedImportanceOfPacketsIndex[:16]
                # NoImportanceIndex = OrderedImportanceOfPacketsIndex[math.floor(len(importanceOfPackets)*50/100):]
                # maxx = importanceOfPackets[OrderedImportanceOfPacketsIndex[math.floor(len(importanceOfPackets)*50/100)]]
                
                # importanceOfPacketsCorr=np.zeros_like(importanceOfPackets)

                # for ip in ImportantPackets:
                #     for nnp in NoImportanceIndex:
                #         corr_coefficient = np.corrcoef(packetizedfmL[ip].flatten(), packetizedfmL[nnp].flatten())[0, 1]
                #         importanceOfPacketsCorr[nnp]= importanceOfPacketsCorr[nnp] + corr_coefficient

                # OrderedCorr = self.__getOrderedImportantPacketIndex(importanceOfPacketsCorr)
                # for cp in OrderedCorr[:math.floor(len(importanceOfPackets)*10/100)]:
                #     importanceOfPacketsWeighted[cp] = maxx

######################################
            # importanceOfPacketsWeighted=importanceOfPackets.copy()
            # IndexesLowGradient = OrderedimportanceOfPacketsSobel[math.floor(len(importanceOfPackets)*90/100):]
            # IndexesNoImportanceNotFEC = OrderedImportanceOfPacketsIndex[math.floor(len(importanceOfPackets)*50/100):math.ceil(len(importanceOfPackets)*60/100)]
            # for i in IndexesLowGradient:
            #     importanceOfPacketsWeighted[i] = max([importanceOfPacketsWeighted[i] for i in IndexesNoImportanceNotFEC])
                
            # importanceOfPacketsWeighted[OrderedimportanceOfPacketsSobel[len(importanceOfPackets)*90/100:]]

            # for i in range (len(importanceOfPackets)*60/100,len(importanceOfPackets)):
            #     OrderedImportanceOfPacketsIndex

            # importanceOfPackets[] importanceOfPacketsSobel

            # if fecPerc is not None:
            #     points_list = [0] * len(OrderedImportanceOfPacketsIndex)
            #     # Generate random indices to place the points
            #     random_indices = random.sample(range(len(OrderedImportanceOfPacketsIndex)), math.floor(len(packetizedfmL) * fecPerc / 100))
            #     # Set the points at the random indices to 1
            #     for index in random_indices:
            #         points_list[index] = -1
            #     importanceOfPacketsWeighted =importanceOfPackets+np.array(points_list)

            OrderedImportanceOfPacketsIndexWeighted = (
                self.__getOrderedImportantPacketIndex(importanceOfPacketsWeighted)
            )

            # if fecPerc not None:
            #     fecIndexs = OrderedImportanceOfPacketsIndexWeighted[-math.floor(len(importanceOfPackets)*fecPerc/100):]
            #     otherIndexs = OrderedImportanceOfPacketsIndexWeighted[:-math.floor(len(importanceOfPackets)*fecPerc/100)]
            #     sorted_list = sorted(fecIndexs)

            #     result = [math.floor(num / packetNum) for num in sorted_list]
                    
                    
            # OrderedImportanceOfPacketsIndexWeighted = OrderedImportanceOfPacketsIndex
            tensorFECCompleted = np.zeros_like(fmL)

            numOfPacketsToLose = math.floor(
                len(packetizedheatMap) * percOfPacketLoss / 100
            )
            totalNumPackets = len(packetizedheatMap)

            indexOfInterpolatedPackets = []

            if case == "FEC (Burst)" or case == "FEC (Burst) NS":
                if percOfPacketLoss != 0:
                    flag = False
                    while not flag:
                        obj = gbChannel.GBC(percOfPacketLoss / 100, 8)
                        sim = obj.simulate(totalNumPackets)
                        numOfPacketsToLose = (~sim).nonzero()[0].size
                        perc = round(numOfPacketsToLose / totalNumPackets * 100)
                        if (
                            perc == percOfPacketLoss
                            or (perc - 1) == percOfPacketLoss
                            or (perc + 1) == percOfPacketLoss
                        ):
                            flag = True
                else:
                    sim = np.full((1, totalNumPackets), True)
                    numOfPacketsToLose = 0

                indexOfLossedPackets = (~sim).nonzero()[0]
                FECPacketCount = math.floor(totalNumPackets * fecPerc / 100)
                protectedPacketCount = math.floor(totalNumPackets * protectedPerc / 100)
                lowestImportanceIndex = OrderedImportanceOfPacketsIndex[
                    -FECPacketCount:
                ]
                OrderedImportanceOfPacketsIndex = (
                    OrderedImportanceOfPacketsIndex[:-FECPacketCount]
                )

                packetsSent = packetsSent + totalNumPackets
                packetsLost = packetsLost + numOfPacketsToLose

                common_elementsProtected = np.intersect1d(
                    indexOfLossedPackets, OrderedImportanceOfPacketsIndex
                )
                common_elementsFEC = np.intersect1d(
                    indexOfLossedPackets, lowestImportanceIndex
                )
                # lostUnprotectedPackets = lostPackets.count(0)
                lostProtectedPackets = len(common_elementsProtected)
                lostRedundantPackets = len(common_elementsFEC)
                if (
                    lostProtectedPackets + lostRedundantPackets <= FECPacketCount
                ):  # RECOVERABLE no protected part will be lost only unprotected
                    indexOfLossedPackets = lowestImportanceIndex
                    # indexOfLossedPackets = []
                else:  # CANNOT RECOVER,lostProtectedPackets valid
                    indexOfLossedPackets = np.append(
                        indexOfLossedPackets, lowestImportanceIndex
                    )
                    indexOfInterpolatedPackets = indexOfLossedPackets
                    pass

            elif case == "FEC (IID)" or case == "FEC (IID) NS":
                fecMask = copy.deepcopy(packetizedfmL)
                totalNumPackets = len(packetizedheatMap)
                FECPacketCount = math.floor(totalNumPackets * fecPerc / 100)
                lowestImportanceIndexFEC = OrderedImportanceOfPacketsIndex[
                -FECPacketCount:
                ]
                for j in range(len(packetizedfmL)):
                    fecMask[j][...] = 0
                for j in lowestImportanceIndexFEC:
                    fecMask[j][...] = 1
                channelFECReconstructed = [
                    np.vstack(fecMask[i : i + packetNum])
                    for i in range(0, len(fecMask), packetNum)
                ]
                tensorFECCompleted = np.dstack(channelFECReconstructed)

                indexOfLossedPackets = list(range(0, totalNumPackets))
                rng.shuffle(indexOfLossedPackets)
                indexOfLossedPackets = indexOfLossedPackets[0:numOfPacketsToLose]

                FECPacketCount = math.floor(totalNumPackets * fecPerc / 100)
                protectedPacketCount = math.floor(totalNumPackets * protectedPerc / 100)
                lowestImportanceIndex = OrderedImportanceOfPacketsIndex[
                    -FECPacketCount:
                ]
                OrderedImportanceOfPacketsIndex = (
                    OrderedImportanceOfPacketsIndex[:-FECPacketCount]
                )

                packetsSent = packetsSent + totalNumPackets
                packetsLost = packetsLost + numOfPacketsToLose

                common_elementsProtected = np.intersect1d(
                    indexOfLossedPackets, OrderedImportanceOfPacketsIndex
                )
                common_elementsFEC = np.intersect1d(
                    indexOfLossedPackets, lowestImportanceIndex
                )
                # lostUnprotectedPackets = lostPackets.count(0)
                lostProtectedPackets = len(common_elementsProtected)
                lostRedundantPackets = len(common_elementsFEC)
                if (
                    lostProtectedPackets + lostRedundantPackets <= FECPacketCount
                ):  # RECOVERABLE no protected part will be lost only unprotected
                    indexOfLossedPackets = lowestImportanceIndex
                    # indexOfInterpolatedPackets = indexOfLossedPackets
                    # indexOfInterpolatedPackets nothing
                else:  # CANNOT RECOVER,lostProtectedPackets valid
                    # indexOfInterpolatedPackets = indexOfLossedPackets.copy()
                    indexOfLossedPackets = np.append(
                        indexOfLossedPackets, lowestImportanceIndex
                    )
                    indexOfInterpolatedPackets = indexOfLossedPackets
                    pass
            elif case == "FEC (IID) Weighted" or case == "FEC (IID) NS Weighted":
                fecMask = copy.deepcopy(packetizedfmL)
                totalNumPackets = len(packetizedheatMap)
                FECPacketCount = math.floor(totalNumPackets * fecPerc / 100)
                lowestImportanceIndexFECWeighted = OrderedImportanceOfPacketsIndexWeighted[
                    -FECPacketCount:
                ]
                for j in range(len(packetizedfmL)):
                    fecMask[j][...] = 0
                for j in lowestImportanceIndexFECWeighted:
                    fecMask[j][...] = 1
                channelFECReconstructed = [
                    np.vstack(fecMask[i : i + packetNum])
                    for i in range(0, len(fecMask), packetNum)
                ]
                tensorFECCompleted = np.dstack(channelFECReconstructed)


                indexOfLossedPackets = list(range(0, totalNumPackets))
                rng.shuffle(indexOfLossedPackets)
                indexOfLossedPackets = indexOfLossedPackets[0:numOfPacketsToLose]

                FECPacketCount = math.floor(totalNumPackets * fecPerc / 100)
                protectedPacketCount = math.floor(totalNumPackets * protectedPerc / 100)
                lowestImportanceIndex = OrderedImportanceOfPacketsIndexWeighted[
                    -FECPacketCount:
                ]
                OrderedImportanceOfPacketsIndexWeighted = (
                    OrderedImportanceOfPacketsIndexWeighted[:-FECPacketCount]
                )

                packetsSent = packetsSent + totalNumPackets
                packetsLost = packetsLost + numOfPacketsToLose

                common_elementsProtected = np.intersect1d(
                    indexOfLossedPackets, OrderedImportanceOfPacketsIndexWeighted
                )
                common_elementsFEC = np.intersect1d(
                    indexOfLossedPackets, lowestImportanceIndex
                )
                # lostUnprotectedPackets = lostPackets.count(0)
                lostProtectedPackets = len(common_elementsProtected)
                lostRedundantPackets = len(common_elementsFEC)
                if (
                    lostProtectedPackets + lostRedundantPackets <= FECPacketCount
                ):  # RECOVERABLE no protected part will be lost only unprotected
                    indexOfLossedPackets = lowestImportanceIndex
                    # indexOfInterpolatedPackets = indexOfLossedPackets
                else:  # CANNOT RECOVER,lostProtectedPackets valid
                    # indexOfInterpolatedPackets = indexOfLossedPackets.copy()
                    indexOfLossedPackets = np.append(
                        indexOfLossedPackets, lowestImportanceIndex
                    )
                    indexOfInterpolatedPackets = indexOfLossedPackets
                    pass

            elif case == "Unprotected (Burst)":
                if percOfPacketLoss != 0:
                    flag = False
                    while not flag:
                        obj = gbChannel.GBC(percOfPacketLoss / 100, 8)
                        sim = obj.simulate(totalNumPackets)
                        numOfPacketsToLose = (~sim).nonzero()[0].size
                        perc = round(numOfPacketsToLose / totalNumPackets * 100)
                        if (
                            perc == percOfPacketLoss
                            or (perc - 1) == percOfPacketLoss
                            or (perc + 1) == percOfPacketLoss
                        ):
                            flag = True
                else:
                    sim = np.full((1, totalNumPackets), True)
                    numOfPacketsToLose = 0

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

            elif case == "Most important NS":
                packetsSent = packetsSent + totalNumPackets
                indexOfLossedPackets = OrderedImportanceOfPacketsIndex[
                    0:numOfPacketsToLose
                ]
                
                indexOfInterpolatedPackets = indexOfLossedPackets
                packetsLost = packetsLost + len(indexOfLossedPackets)

            elif case == "Least important":
                packetsSent = packetsSent + totalNumPackets
                OrderedImportanceOfPacketsIndex = OrderedImportanceOfPacketsIndex[::-1]
                
                indexOfLossedPackets = OrderedImportanceOfPacketsIndex[0:numOfPacketsToLose]
                packetsLost = packetsLost + len(indexOfLossedPackets)

            elif case == "Least important NS":
                packetsSent = packetsSent + totalNumPackets
                OrderedImportanceOfPacketsIndex = OrderedImportanceOfPacketsIndex[::-1]
                
                indexOfLossedPackets = OrderedImportanceOfPacketsIndex[0:numOfPacketsToLose]
                indexOfInterpolatedPackets = indexOfLossedPackets
                packetsLost = packetsLost + len(indexOfLossedPackets)

            elif case == "Most important Weighted":
                packetsSent = packetsSent + totalNumPackets
                indexOfLossedPackets = OrderedImportanceOfPacketsIndexWeighted[0:numOfPacketsToLose]
                
                packetsLost = packetsLost + len(indexOfLossedPackets)

            elif case == "Most important NS Weighted":
                packetsSent = packetsSent + totalNumPackets
                indexOfLossedPackets = OrderedImportanceOfPacketsIndexWeighted[0:numOfPacketsToLose]
                indexOfInterpolatedPackets = indexOfLossedPackets
                packetsLost = packetsLost + len(indexOfLossedPackets)

            elif case == "Least important Weighted":
                packetsSent = packetsSent + totalNumPackets
                OrderedImportanceOfPacketsIndexWeighted = OrderedImportanceOfPacketsIndexWeighted[::-1]
                
                indexOfLossedPackets = OrderedImportanceOfPacketsIndexWeighted[0:numOfPacketsToLose]
                
                packetsLost = packetsLost + len(indexOfLossedPackets)

            elif case == "Least important NS Weighted":
                packetsSent = packetsSent + totalNumPackets
                OrderedImportanceOfPacketsIndexWeighted = OrderedImportanceOfPacketsIndexWeighted[::-1]
                
                indexOfLossedPackets = OrderedImportanceOfPacketsIndexWeighted[0:numOfPacketsToLose]
                
                indexOfInterpolatedPackets = indexOfLossedPackets
                packetsLost = packetsLost + len(indexOfLossedPackets)

            elif case == "Unprotected (IID) NS":
                packetsSent = packetsSent + totalNumPackets
                indexOfLossedPackets = list(range(0, totalNumPackets))
                rng.shuffle(indexOfLossedPackets)
                indexOfLossedPackets = indexOfLossedPackets[0:numOfPacketsToLose]
                indexOfInterpolatedPackets = indexOfLossedPackets
                packetsLost = packetsLost + len(indexOfLossedPackets)

            elif case == "Unprotected (Burst) NS":
                if percOfPacketLoss != 0:
                    flag = False
                    while not flag:
                        obj = gbChannel.GBC(percOfPacketLoss / 100, 8)
                        sim = obj.simulate(totalNumPackets)
                        numOfPacketsToLose = (~sim).nonzero()[0].size
                        perc = round(numOfPacketsToLose / totalNumPackets * 100)
                        if (
                            perc == percOfPacketLoss
                            or (perc - 1) == percOfPacketLoss
                            or (perc + 1) == percOfPacketLoss
                        ):
                            flag = True
                else:
                    sim = np.full((1, totalNumPackets), True)
                    numOfPacketsToLose = 0
                packetsSent = packetsSent + totalNumPackets
                indexOfLossedPackets = (~sim).nonzero()[0]
                indexOfInterpolatedPackets = (~sim).nonzero()[0]
                packetsLost = packetsLost + len(indexOfLossedPackets)

            elif case == "Unprotected (IID) EN":
                packetsSent = packetsSent + totalNumPackets
                indexOfLossedPackets = list(range(0, totalNumPackets))
                rng.shuffle(indexOfLossedPackets)
                indexOfLossedPackets = indexOfLossedPackets[0:numOfPacketsToLose]
                packetsLost = packetsLost + len(indexOfLossedPackets)

                tensorEncodedBufferSize = 0

                for j in range(len(packetizedfmL)):
                    # print(len(packetizedfmL))
                    if qualityFactor == 110:
                        encimg = packetizedfmL[j].astype("uint8")
                        data_encode = np.array(encimg)
                        byte_encode = data_encode.tobytes()
                        tensorEncodedBufferSize = (
                            tensorEncodedBufferSize + len(byte_encode) * 8
                        )
                    if qualityFactor == 105:
                        # High to Low index 33,1,21,199,6,512 etc to 100,99,98
                        division_length = (
                            len(OrderedImportanceOfPacketsIndex) // 100
                        )
                        # Divide the array into equal divisions
                        divided_array = [
                            OrderedImportanceOfPacketsIndex[
                                i : i + division_length
                            ]
                            for i in range(
                                0,
                                len(OrderedImportanceOfPacketsIndex),
                                division_length,
                            )
                        ]
                        index = None
                        for i, division in enumerate(divided_array):
                            if j in division:
                                index = i
                                index = abs(index - 100)
                                break
                        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), index]
                        result, encimg = cv.imencode(
                            ".jpg", packetizedfmL[j].astype("uint8"), encode_param
                        )
                        data_encode = np.array(encimg)
                        # Converting the array to bytes.
                        byte_encode = data_encode.tobytes()
                        tensorEncodedBufferSize = (
                            tensorEncodedBufferSize + len(byte_encode) * 8
                        )
                        decimg = cv.imdecode(encimg, cv.IMREAD_GRAYSCALE)
                        packetizedfmL[j] = np.array(decimg)

                    else:
                        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), qualityFactor]
                        result, encimg = cv.imencode(
                            ".jpg", packetizedfmL[j].astype("uint8"), encode_param
                        )
                        # Converting the image into numpy array
                        data_encode = np.array(encimg)
                        # Converting the array to bytes.
                        byte_encode = data_encode.tobytes()
                        tensorEncodedBufferSize = (
                            tensorEncodedBufferSize + len(byte_encode) * 8
                        )
                        decimg = cv.imdecode(encimg, cv.IMREAD_GRAYSCALE)
                        packetizedfmL[j] = np.array(decimg)

                batchBpp.append(
                    tensorEncodedBufferSize / float(self.dataset_x_files_sizes[i_b])
                )
            else:
                raise Exception("Case can only be Random,Top or Random_RSCorrected.")

            mask = []
            for j in range(len(packetizedfmL)):
                mask.append(np.zeros_like(packetizedfmL[j]))

            packetsWithoutLoss = copy.deepcopy(packetizedfmL)

            for j in indexOfLossedPackets:
                packetizedfmL[j][...] = 0

            for j in indexOfInterpolatedPackets:
                mask[j][...] = 1

            channelReconstructed = [
                np.vstack(packetizedfmL[i : i + packetNum])
                for i in range(0, len(packetizedfmL), packetNum)
            ]

            channelReconstructedNoLoss = [
                np.vstack(packetsWithoutLoss[i : i + packetNum])
                for i in range(0, len(packetsWithoutLoss), packetNum)
            ]

            maskR = [
                np.vstack(mask[i : i + packetNum])
                for i in range(0, len(mask), packetNum)
            ]

            # if pad != 0:
            #     channelReconstructed = [
            #         channelReconstructed[i][0:-pad, ...]
            #         for i in range(0, len(channelReconstructed))
            #     ]
            #     channelReconstructedNoLoss = [
            #         channelReconstructedNoLoss[i][0:-pad, ...]
            #         for i in range(0, len(channelReconstructedNoLoss))
            #     ]
            #     maskR = [maskR[i][0:-pad, ...] for i in range(0, len(maskR))]

            # packetizedImportanceMap = [
            #     np.ones_like(packetizedheatMap[i_p]) * np.sum(packetizedheatMap[i_p])
            #     for i_p in range(len(packetizedheatMap))
            # ]

            packetizedImportanceMap = [
                np.ones_like(packetizedheatMap[i_p]) * importanceOfPacketsWeighted[i_p]
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
                channelReconstructed = [
                    channelReconstructed[i][0:-pad, ...]
                    for i in range(0, len(channelReconstructed))
                ]
                channelReconstructedNoLoss = [
                    channelReconstructedNoLoss[i][0:-pad, ...]
                    for i in range(0, len(channelReconstructedNoLoss))
                ]
                maskR = [maskR[i][0:-pad, ...] for i in range(0, len(maskR))]

            tensorImportanceCompleted = np.dstack(channelReconstructedImportance)
            tensorCompleted = np.dstack(channelReconstructed)
            tensorCompletedNoLoss = np.dstack(channelReconstructedNoLoss)
            maskCompleted = np.dstack(maskR)

            if (
                case == "Unprotected (IID) NS"
                or case == "Unprotected (Burst) NS"
                or case == "FEC (Burst) NS"
                or case == "FEC (IID) NS"
                or case == "Most important NS"
                or case == "Least important NS"
                or case == "Most important NS Weighted"
                or case == "Least important NS Weighted"
                or case == "FEC (IID) NS Weighted"
            ):
                # for ind in range(tensorCompleted.shape[2]):
                #     img_gray_cv2 = cv.cvtColor(
                #         tensorCompleted[:, :, ind].astype("uint8"), cv.COLOR_GRAY2BGR
                #     )
                #     dst = cv.inpaint(
                #         img_gray_cv2,
                #         maskCompleted[:, :, ind].astype("uint8"),
                #         7,
                #         cv.INPAINT_TELEA,
                #     )
                #     dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
                #     tensorCompleted[:, :, ind] = dst

                channelnum = tensorCompleted.shape[2]
                pktCount = 8
                lossmap = np.full(channelnum * pktCount, True)
                lossmap[indexOfInterpolatedPackets] = False
                lossmap = np.reshape(lossmap, (channelnum, pktCount))

                pktz = np.array(packetizedfmL).reshape(
                    np.array(packetizedfmL).shape[0], -1
                )
                pktz = pktz.reshape(-1, 8, pktz.shape[1])

                pktzNoLoss = np.array(packetsWithoutLoss).reshape(
                    np.array(packetsWithoutLoss).shape[0], -1
                )
                pktzNoLoss = pktzNoLoss.reshape(-1, 8, pktzNoLoss.shape[1])

                pktz = self.fn_caltec(lossmap, pktz,pktzNoLoss,False)

                pktz = pktz.reshape(-1, pktz.shape[2])
                pktz = pktz.reshape(pktz.shape[0], 7, -1)

                channelReconstructed = [
                    np.vstack(pktz[i : i + packetNum])
                    for i in range(0, len(pktz), packetNum)
                ]
                tensorCompleted = np.dstack(channelReconstructed)

            mse = np.mean((tensorCompleted - tensorCompletedNoLoss) ** 2)
            mseList.append(mse)

            fmL = self.__inverseQuantize(tensorCompleted, qBits, minVal, maxVal)

            fmLPacketizedFEC.append(tensorFECCompleted)
            fmLPacketizedLoss.append(fmL)
            importancePacketsTensor.append(tensorImportanceCompleted)

        if saveImages:
            self.__savePacketLossImages(fmLPacketizedLoss,fmLPacketizedFEC, case, modelName)
            self.__savePacketLossImages(
                self.__normalizeToUnit(np.array(importancePacketsTensor)),fmLPacketizedFEC,
                "packetImportance",
                modelName,
            )
            return

        # with open("Korcan/Plots/"+modelName+"/"+case+"_"+str(packetNum)+"_"+str(percOfPacketLoss)+".npy", 'wb') as f:
        #         np.save(f, np.array(scores))

        # acc = 0
        # loss = 0
        # count = 0
        # for s in scores:
        #     count = count + 1
        #     acc = acc + s["acc"]
        #     loss = loss + s["loss"]

        # if (
        #     case == "Most important"
        #     or case == "Least important"
        #     or case == "Most important NS"
        #     or case == "Least important NS"
        #     or case == "Most important Weighted"
        #     or case == "Least important Weighted"
        #     or case == "Most important NS Weighted"
        #     or case == "Least important NS Weighted"
        #     or case == "Unprotected (IID)"
        #     or case == "Unprotected (Burst)"
        #     or case == "Unprotected (IID) NS"
        #     or case == "Unprotected (Burst) NS"
        #     or case == "FEC (IID)"
        #     or case == "FEC (Burst)"
        #     or case == "FEC (IID) NS"
        #     or case == "FEC (Burst) NS"
        #     or case == "Unprotected (IID) EN"
        # ):
        if case == "Unprotected (IID) EN":
            case = "Unprotected (IID) EN_" + str(qualityFactor)
            percOfPacketLoss = sum(batchBpp) / float(len(batchBpp))

        # if not os.path.exists("Korcan/Plots/" + modelName + "/" + case):
        #     os.makedirs("Korcan/Plots/" + modelName + "/" + case, exist_ok=True)

        pdictKey = ("{:.3f}".format(percOfPacketLoss), case)
        metrics = self.getMetrics(fmLPacketizedLoss)
        pdictVal = {
            "acc": metrics["acc"],
            "loss": metrics["loss"],
            "min": 0,
            "max": 0,
        }
        # pdictVal = {
        #     "acc": sum(mseList) / len(mseList),
        #     "loss": 0,
        #     "min": 0,
        #     "max": 0,
        # }

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
    # else:
    #     if not os.path.exists(
    #         "Korcan/Plots/"
    #         + modelName
    #         + "/"
    #         + case
    #         + "_"
    #         + str(fecPerc)
    #         + "_"
    #         + str(protectedPerc)
    #     ):
    #         os.makedirs(
    #             "Korcan/Plots/"
    #             + modelName
    #             + "/"
    #             + case
    #             + "_"
    #             + str(fecPerc)
    #             + "_"
    #             + str(protectedPerc)
    #         )

    #     pdictKey = ("{:.3f}".format(100 * packetsLost / packetsSent), case)
    #     pdictVal = {
    #         "acc": metrics["acc"],
    #         "loss": metrics["loss"],
    #         "min": 0,
    #         "max": 0,
    #     }

    #     rand = int(random.randint(1, sys.maxsize))
    #     with open(
    #         "Korcan/Plots/"
    #         + modelName
    #         + "/"
    #         + case
    #         + "_"
    #         + str(fecPerc)
    #         + "_"
    #         + str(protectedPerc)
    #         + "/key_"
    #         + "{:.3f}".format(100 * packetsLost / packetsSent)
    #         + "_"
    #         + str(rand)
    #         + "_.pkl",
    #         "wb",
    #     ) as f:
    #         pickle.dump(pdictKey, f)
    #     with open(
    #         "Korcan/Plots/"
    #         + modelName
    #         + "/"
    #         + case
    #         + "_"
    #         + str(fecPerc)
    #         + "_"
    #         + str(protectedPerc)
    #         + "/val_"
    #         + "{:.3f}".format(100 * packetsLost / packetsSent)
    #         + "_"
    #         + str(rand)
    #         + "_.pkl",
    #         "wb",
    #     ) as f:
    #         pickle.dump(pdictVal, f)
    # self.pdict["{:.3f}".format(100 * packetsLost / packetsSent),case,] = {"acc": metrics["acc"], "loss": metrics["loss"]}


if __name__ == "__main__":
    # modelName = "efficientnetb0"
    # splitLayer = "block2b_add"
    # modelName = "resnet18"
    # splitLayer = "add_1"
    # modelName = "dense"
    # splitLayer = "pool2_conv"

    percLoss = int(sys.argv[1])
    case = sys.argv[2]
    qualityFactor = sys.argv[4]

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

    trained_model_path = "/local-scratch/localhome/kuyanik/UnequalResearch/model.05-0.00.h5"
    dataName = "/local-scratch/localhome/kuyanik/dataset/smallTest"
    # trained_model_path = "/project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/model.05-0.00.h5"
    # dataName = "/home/foniks/projects/def-ibajic/foniks/Project_1/largeTest"
    quantizationBits = 8

    module = pipeline()
    module.loadModel(
        modelPath, mobile_model_path, cloud_model_path, trained_model_path, splitLayer
    )
    module.loadData(dataName, [224, 224], False)
    module.findHeatmaps(splitLayer, modelName, dataName)

    packetCount = 8

    if case == "1":
        case = "Most important"
    elif case == "2":
        case = "Least important"
    elif case == "3":
        case = "Unprotected (IID)"
    elif case == "4":
        case = "Unprotected (Burst)"
    elif case == "5":
        case = "FEC (IID)"
    elif case == "6":
        case = "FEC (Burst)"
    elif case == "7":
        case = "Unprotected (IID) NS"
    elif case == "8":
        case = "Unprotected (Burst) NS"
    elif case == "9":
        case = "Unprotected (IID) EN"
    elif case == "11":
        case = "FEC (IID) NS"
    elif case == "12":
        case = "FEC (Burst) NS"
    elif case == "13":
        case = "Most important NS"
    elif case == "14":
        case = "Least important NS"
    elif case == "15":
        case = "Most important Weighted"
    elif case == "16":
        case = "Least important Weighted"
    elif case == "17":
        case = "Most important NS Weighted"
    elif case == "18":
        case = "Least important NS Weighted"
    elif case == "19":
        case = "FEC (IID) Weighted"
    elif case == "20":
        case = "FEC (IID) NS Weighted"

    if case == "10":
        # module.saveSuperImposedChannels(modelName)
        saveImageLossPercent = 0
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
        #     "Most important NS",
        #     saveImages=True,
        #     modelName=modelName,
        # )
        # module.packetLossSim(
        #     packetCount,
        #     quantizationBits,
        #     saveImageLossPercent,
        #     "Least important NS",
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
        # #     "Unprotected (Burst)",
        # #     saveImages=True,
        # #     modelName=modelName,
        # # )
        # module.packetLossSim(
        #     packetCount,
        #     quantizationBits,
        #     saveImageLossPercent,
        #     "Unprotected (IID) NS",
        #     saveImages=True,
        #     modelName=modelName,
        # )
        # module.packetLossSim(
        #     packetCount,
        #     quantizationBits,
        #     saveImageLossPercent,
        #     "Unprotected (Burst) NS",
        #     saveImages=True,
        #     modelName=modelName,
        # )
        # module.packetLossSim(
        #     packetCount,
        #     quantizationBits,
        #     saveImageLossPercent,
        #     "Unprotected (IID) EN",
        #     saveImages=True,
        #     modelName=modelName,
        # )

        module.packetLossSim(
            packetCount,
            quantizationBits,
            saveImageLossPercent,
            "FEC (IID)",
            30,
            70,
            saveImages=True,
            modelName=modelName,
        )

        # module.packetLossSim(
        #     packetCount,
        #     quantizationBits,
        #     saveImageLossPercent,
        #     "FEC (IID) NS",
        #     30,
        #     70,
        #     saveImages=True,
        #     modelName=modelName,
        # )

        module.packetLossSim(
            packetCount,
            quantizationBits,
            saveImageLossPercent,
            "FEC (IID) Weighted",
            30,
            70,
            saveImages=True,
            modelName=modelName,
        )

        # module.packetLossSim(
        #     packetCount,
        #     quantizationBits,
        #     saveImageLossPercent,
        #     "FEC (IID) NS Weighted",
        #     30,
        #     70,
        #     saveImages=True,
        #     modelName=modelName,
        # )

        # module.packetLossSim(
        #     packetCount,
        #     quantizationBits,
        #     saveImageLossPercent,
        #     "FEC (Burst) NS",
        #     30,
        #     70,
        #     saveImages=True,
        #     modelName=modelName,
        # )
        # module.packetLossSim(
        #     packetCount,
        #     quantizationBits,
        #     saveImageLossPercent,
        #     "FEC (IID)",
        #     30,
        #     70,
        #     saveImages=True,
        #     modelName=modelName,
        # )
        # module.packetLossSim(
        #     packetCount,
        #     quantizationBits,
        #     saveImageLossPercent,
        #     "FEC (Burst)",
        #     30,
        #     70,
        #     saveImages=True,
        #     modelName=modelName,
        # )
        sys.exit()

    if (
        case == "Most important"
        or case == "Least important"
        or case == "Most important NS"
        or case == "Least important NS"
        or case == "Unprotected (IID)"
        or case == "Unprotected (Burst)"
        or case == "Unprotected (IID) NS"
        or case == "Unprotected (Burst) NS"
        or case == "Unprotected (IID) EN"
        or case == "Most important Weighted"
        or case == "Least important Weighted"
        or case == "Most important NS Weighted"
        or case == "Least important NS Weighted"
    ):
        module.packetLossSim(
            packetCount,
            8,
            percLoss,
            case,
            modelName=modelName,
            qualityFactor=int(qualityFactor),
        )
    elif case == "makeplot":
        # dirs = [
        #     name
        #     for name in os.listdir("Korcan/Plots/" + modelName)
        #     if os.path.isdir("Korcan/Plots/" + modelName)
        # ]
        # fpPairs = []
        # for d in dirs:
        #     splitted = d.split("_")
        #     if len(splitted) == 5:
        #         fpPairs.append(splitted[3:])  # FEC_10_50

        dirNames = []
        dirNames.append("Most important")
        dirNames.append("Least important")
        # dirNames.append("Most important NS")
        # dirNames.append("Least important NS")

        # dirNames.append("Most important Weighted")
        # dirNames.append("Most important NS Weighted")
        # dirNames.append("Least important Weighted")
        # dirNames.append("Least important NS Weighted")

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
                    module.pdict[key] = val

        dirNames = []

        dirNames.append("FEC (IID) Weighted")
        dirNames.append("FEC (IID) NS Weighted")
        dirNames.append("FEC (IID)")
        dirNames.append("FEC (IID) NS")

        dirNames.append("Unprotected (IID)")
        dirNames.append("Unprotected (IID) NS")

        # dirNames.append("FEC (Burst)")
        # dirNames.append("FEC (Burst) NS")
        # dirNames.append("Unprotected (Burst)")
        # dirNames.append("Unprotected (Burst) NS")
        # dirNames = []

        # set the directory path
        # dir_path = "Korcan/Plots/" + modelName
        # # set the string to search for at the beginning of directory names
        # start_string = "Unprotected (IID) EN"
        # # get the list of directories in the directory path that start with the string
        # dirs = [
        #     d
        #     for d in os.listdir(dir_path)
        #     if os.path.isdir(os.path.join(dir_path, d)) and d.startswith(start_string)
        # ]
        # dirNames = dirNames + dirs
        # print(dirNames)

        tTestIID = {}
        tTestBurst = {}

        for d in dirNames:
            listFiles = os.listdir("Korcan/Plots/" + modelName + "/" + d)
            keyIndexes = []
            for i in range(len(listFiles)):
                if listFiles[i][:3] == "key":
                    keyIndexes.append(i)

            blacklist = []
            for i in keyIndexes:
                lossPercInfo = listFiles[i].split("_")[1]  # 90.000
                if float(lossPercInfo) not in blacklist:
                    allRunsWithSamePercentage = glob.glob(
                        "Korcan/Plots/"
                        + modelName
                        + "/"
                        + d
                        + "/key_"
                        + lossPercInfo
                        + "*"
                    )
                    # print(allRunsWithSamePercentage)
                    # print(blacklist)
                    blacklist.extend([float(lossPercInfo)])
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
                        # print(fname)
                        # print(key, val)
                    acc = 0
                    loss = 0
                    count = 0
                    accList = []
                    for s in val:
                        count = count + 1
                        acc = acc + s["acc"]
                        accList.append(s["acc"])
                        loss = loss + s["loss"]

                    std = np.std(np.array(accList))
                    min = (acc / count) - std
                    max = (acc / count) + std

                    if d == "FEC (IID)":
                        print(lossPercInfo,std)
                    # min = np.amin(np.array(accList))
                    # max = np.amax(np.array(accList))
                    if d == "Unprotected (IID)":
                        tTestIID[lossPercInfo] = accList
                    elif d == "Unprotected (Burst)":
                        tTestBurst[lossPercInfo] = accList
                    # print(count)
                    # print(key)
                    # print("a____")
                    # print(val)
                    # print("b____")
                    # print({"acc": acc/count, "loss": loss/count})
                    # if key[1] == "FEC (IID)":
                    #     key = list(key)
                    #     dp = "FEC (IID)" + d[-6:]
                    #     key[1] = dp
                    #     key = tuple(key)
                    # elif key[1] == "FEC (Burst)":
                    #     key = list(key)
                    #     dp = "FEC (Burst)" + d[-6:]
                    #     key[1] = dp
                    #     key = tuple(key)
                    # else:
                    #     key = list(key)
                    #     key[1] = "Unprotected"
                    #     key = tuple(key)
                    # print(key, acc / count, count)
                    module.pdict[key] = {
                        "acc": acc / count,
                        "loss": loss / count,
                        "min": min,
                        "max": max,
                    }
        module.makePlot(
            "Korcan/Plots/" + modelName + "/AccuracyPlotPacketized",
            "Korcan/Plots/" + modelName + "/LossPlotPacketized",
        )
        # tTestDict = {}
        # tTestDictL = {}
        # tTestDictG = {}
        # for k in tTestIID.keys():
        #     a = np.array(tTestIID[k])
        #     b = np.array(tTestBurst[k])
        #     # a=np.around(a, decimals=4)
        #     # b=np.around(b, decimals=4)
        #     print(a)
        #     # print(b)
        #     # print(k)
        #     if int(float(k)) == 0:
        #         continue
        #     tTestDict[int(float(k))] = scipy.stats.ttest_ind(a, b, equal_var=True)
        #     tTestDictL[int(float(k))] = scipy.stats.ttest_ind(
        #         a, b, alternative="less", equal_var=True
        #     )
        #     tTestDictG[int(float(k))] = scipy.stats.ttest_ind(
        #         a, b, alternative="greater", equal_var=True
        #     )
        #     # c = scipy.stats.ttest_ind(a,b,equal_var=True)

        # x = []
        # y = []
        # y2 = []
        # for k, v in dict(sorted(tTestDict.items())).items():
        #     x.append(k)
        #     y.append(v[0])
        #     y2.append(v[1])

        # xl = []
        # yl = []
        # y2l = []
        # for k, v in dict(sorted(tTestDictL.items())).items():
        #     xl.append(k)
        #     yl.append(v[0])
        #     y2l.append(v[1])

        # xg = []
        # yg = []
        # y2g = []
        # for k, v in dict(sorted(tTestDictG.items())).items():
        #     xg.append(k)
        #     yg.append(v[0])
        #     y2g.append(v[1])

        # # keys = tTestDict.keys()
        # # keys = tTestDict.keys()
        # plt.xlabel("Percent Lost")
        # plt.ylabel("T value")
        # plt.scatter(
        #     x,
        #     y,
        #     s=25,
        #     label="_nolegend_",
        # )
        # plt.plot(
        #     x,
        #     y,
        #     linewidth=1.2,
        # )
        # plt.scatter(
        #     xl,
        #     yl,
        #     s=25,
        #     label="_nolegend_",
        # )
        # plt.plot(
        #     xl,
        #     yl,
        #     linewidth=1.2,
        # )
        # plt.scatter(
        #     xg,
        #     yg,
        #     s=25,
        #     label="_nolegend_",
        # )
        # plt.plot(
        #     xg,
        #     yg,
        #     linewidth=1.2,
        # )
        # # # plt.axis('off')
        # plt.savefig(
        #     "Korcan/Plots/" + modelName + "/tTest",
        #     bbox_inches="tight",
        #     dpi=300,
        # )
        # plt.close()

        # plt.xlabel("Percent Lost")
        # plt.ylabel("P values")
        # plt.hlines(y=0.05, xmin=0, xmax=100, linewidth=1, color="r")
        # plt.scatter(
        #     x,
        #     y2,
        #     s=25,
        #     label="two-sided",
        # )
        # plt.plot(
        #     x,
        #     y2,
        #     linewidth=1.2,
        # )
        # plt.scatter(
        #     xl,
        #     y2l,
        #     s=25,
        #     label="less",
        # )
        # plt.plot(
        #     xl,
        #     y2l,
        #     linewidth=1.2,
        # )
        # plt.scatter(
        #     xg,
        #     y2g,
        #     s=25,
        #     label="greater",
        # )
        # plt.plot(
        #     xg,
        #     y2g,
        #     linewidth=1.2,
        # )
        # plt.legend(
        #     # [handles[i] for i in order], [labels[i] for i in order],
        #     loc="upper right",
        #     fontsize="xx-small",
        #     markerscale=0.7,
        #     # ncol=2,
        #     fancybox=True,
        #     # shadow=True,
        #     prop={"size": 8},
        # )
        # plt.savefig(
        #     "Korcan/Plots/" + modelName + "/tTestP",
        #     bbox_inches="tight",
        #     dpi=300,
        # )

    elif (
        case == "FEC (IID)"
        or case == "FEC (Burst)"
        or case == "FEC (IID) NS"
        or case == "FEC (Burst) NS"
        or case == "FEC (IID) NS Weighted"
        or case == "FEC (IID) Weighted"
    ):
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
    #             #     packetCount, 8, percLoss, "FEC", f, p,modelName=modelName
    #             # )
    #         for percLoss in np.linspace(15, 50, 2):
    #             # module.packetLossSim(packetCount, 8, percLoss, "Least important",modelName=modelName)
    #             # module.packetLossSim(
    #             #     packetCount, 8, percLoss, "Random_RSCorrected", f, p,modelName=modelName
    #             # )
    #             # module.packetLossSim(
    #             #     packetCount, 8, percLoss, "FEC", f, p,modelName=modelName
    #             # )
    #         # for percLoss in np.linspace(50, 100, 2):
    #         #     module.packetLossSim(packetCount, 8, percLoss, "Random_RSCorrected", fecPercent, protectPercent)
    #         module.makePlot(
    #             "Korcan/Plots/"+modelName+"/AccuracyPlotPacketized2_" + fecProtectInfo,
    #             "Korcan/Plots/"+modelName+"/LossPlotPacketized2_" + fecProtectInfo,
    #         )
    #         module.cleanPlot()
