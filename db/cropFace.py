import cv2
import os


def cropImgToFace(image):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    haarcascade_frontalface_default_path = os.path.join(ROOT_DIR,'haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(haarcascade_frontalface_default_path) # Create the haar cascade

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(grayscale_image,
        scaleFactor=1.1,    #compensate difrence in dinstance from face to camera
        minNeighbors=7,     #how many objects must be around something to call it face
        minSize=(64, 64),   #size of mask, wich will be used to detect faces
        flags = cv2.CASCADE_SCALE_IMAGE)

    if len(faces) != 1:
        print ("Found {0} faces!".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = grayscale_image[y: y+h, x: x + w]
        face = cv2.resize(face,(64,64))
        return face


def saveImg(img, directory):
    cv2.imwrite(directory, img)


def createDirectory(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        print("{} directory already exist".format(dir))


def processAllImgsInDirectory(directory):
    createDirectory("croped_" + directory)
    classesFolders = os.listdir(directory)
    for classDir in range(len(classesFolders)-1):

        imagesDir = directory + str(classDir)
        imagesInDir = os.listdir(imagesDir)
        createDirectory("croped_"+imagesDir)
        print("class number: ", "croped_"+imagesDir)

        for file in imagesInDir:
            filePath = os.path.join(imagesDir, file)

            if filePath[-9:] == ".DS_Store":
                continue
            img = cv2.imread(filePath)
            face = cropImgToFace(img)

            saveDir = "croped_" + filePath
            saveImg(face, saveDir)


if __name__ == '__main__':
    dir = input("Give database directory: ")
    processAllImgsInDirectory(dir)
