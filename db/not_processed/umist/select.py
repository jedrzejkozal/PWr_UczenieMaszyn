import cv2
import os

whole = cv2.imread('umist_cropped.jpg')
number_of_rows = 24
number_of_columns = 24

saveDir = "croped/"

i_offset = 2
j_offset = 4
i_size = 111
j_size = 89


def cleanup():
    os.system("rm -rf " + saveDir)


def createDir(dirName):
    try:
        os.mkdir(dirName)
    except FileExistsError:
        pass


def createDirForClass(classIndex):
    newClassDir = saveDir + str(classIndex)
    createDir(newClassDir)


def cropImages():
    classIndex = 0
    savedImages = 0
    imgs_in_class = [38, 35, 26, 24, 26, 23, 19, 22, 20, 32, 34, 34, 26, 30, 19, 26, 26, 33, 48, 34, 999999] #is it? no it is not :)

    for j in range(number_of_columns):
        for i in range(number_of_rows):
            i_index = i_offset + i * (i_size + i_offset)
            j_index = j_offset + j * (j_size + j_offset)
            chunk = whole[i_index:i_index+i_size, j_index:j_index+j_size]

            fileIndex = str(j*number_of_rows + (i+1))
            if savedImages >= imgs_in_class[classIndex]:
                classIndex += 1
                createDirForClass(classIndex)
                savedImages = 0
            classDir = str(classIndex) + "/"

            imgSaveDir = saveDir + classDir + "img" + fileIndex + ".jpg"
            cv2.imwrite(imgSaveDir, chunk)
            savedImages += 1


if __name__ == '__main__':
    cleanup()
    createDir(saveDir)
    createDirForClass(0)
    cropImages()
