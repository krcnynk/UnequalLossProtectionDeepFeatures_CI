# import zipfile
# with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
#     zip_ref.extractall(directory_to_extract_to)
import os
import shutil

trainDir = "/media/sf_Downloads/ILSVRC2012_img_train"
trainDir2 = "/media/sf_Downloads/ILSVRC2012_img_trainNew"
os.mkdir(os.path.join(trainDir2))
folderNames = [name for name in os.listdir(trainDir)]
# print(folderNames)
for name in folderNames:
    fileNames = [fname for fname in os.listdir(os.path.join(trainDir, name))]
    os.mkdir(os.path.join(trainDir2, name))
    count = 0
    for fname in fileNames:
        if(count < 15):
            shutil.copyfile(os.path.join(trainDir, name,fname), os.path.join(trainDir2, name,fname))
            count = count + 1
        # print(os.path.join(trainDir, name,fname))

    # os.rename(os.path.join(trainDir, name),os.path.join(trainDir, str(ind)))
    