import os



if __name__ == "__main__":
    labelFile = "/media/sf_Downloads/ILSVRCdatabase/caffe.txt"
    with open(labelFile) as file:
        listOfFilenameLabel = [line.split(" ")[0] for line in file]

    trainDir = "/media/sf_Downloads/ILSVRCdatabase/ILSVRC2012_img_train"
    folderNames = [name for name in os.listdir(trainDir)]
    for name in folderNames:
        ind = listOfFilenameLabel.index(name)
        os.rename(os.path.join(trainDir, name),os.path.join(trainDir, str(ind)))
    
    valDir = "/media/sf_Downloads/ILSVRCdatabase/ILSVRC2012_img_val"
    folderNames = [name for name in os.listdir(valDir) if os.path.isfile(os.path.join(valDir,name))]
    print(folderNames)

    clslocFile = "/media/sf_Downloads/ILSVRCdatabase/map_clsloc.txt"
    with open(clslocFile) as file:
        clslocS = [line.split(" ")[:2] for line in file]
    print(clslocS)

    mappingFileName = "/media/sf_Downloads/ILSVRCdatabase/ILSVRC2012_validation_ground_truth.txt"
    with open(mappingFileName) as file:
        mapping = [int(line[:-1]) for line in file]
    # print(mapping)

    for name in folderNames:
        idName = int(name.split("_")[2][:-5])
        label = mapping[idName-1]
        # print(label)
        fileName = [item for item in clslocS if item[1] == str(label)]
        ind = listOfFilenameLabel.index(fileName[0][0])
        os.rename(os.path.join(valDir, name),os.path.join(valDir, str(ind)+".JPEG"))