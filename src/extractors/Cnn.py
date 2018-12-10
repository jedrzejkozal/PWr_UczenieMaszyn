from utils import convertListOfNdarraysToNdarray

from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras import optimizers
from keras.utils import np_utils


class Cnn:
    def __init__(self, xTrain, yTrain, numberOfClasses):
        self.__numClasses = numberOfClasses

    def fit(self, xTrain, yTrain):
        xTrainCnn = self.__reshapeX(xTrain)
        yTrainCnn = self.__reshapeY(yTrain)
        self._engine = self.__trainModel(xTrainCnn, yTrainCnn)
        #accuracy = self.__calcAccuracy()

        print("Cnn model:")
        print(self._engine.summary())

        numLayersToRemove = 2
        self.__removeLastLayers(numLayersToRemove)
        print("Cnn model after removing classifying part:")
        print(self._engine.summary())


    def transform(self, x):
        xCnn = self.__reshapeX(x)

        features = []
        for elem in xCnn:
            elemReshaped = elem.reshape((1,) + elem.shape)
            bathFeatures = self._engine.predict(elemReshaped, batch_size = 1, verbose = 0)
            features.append(bathFeatures[0])

        return convertListOfNdarraysToNdarray(features)

    def fit_transform(self, x, y):
        self.fit(x,y)
        return self.transform(x)

    def __reshapeX(self, x):
        numOfChannels = 1
        return x.reshape((x.shape[0], x.shape[1], x.shape[2], numOfChannels))

    def __reshapeY(self, y):
        return np_utils.to_categorical(y, num_classes=self.__numClasses)

    def __removeLastLayers(self, numLayersToRemove):
        #https://github.com/keras-team/keras/issues/2371
        for i in range(numLayersToRemove):
            self._engine.pop()
        self._engine.build(None)

    '''
    def __calcCnnAccuracy():
        score = self._engine.evaluate(xTestCnn, yTestCnn, verbose=0)
        return score[1]
    '''

    def __trainModel(self, xTrainCnn, yTrainCnn):
        cnn = Sequential()

        numOfChannels = 1
        imgSideLen = xTrainCnn.shape[1]
        cnn.add(Conv2D(input_shape=(imgSideLen, imgSideLen, numOfChannels),
                       filters=32, kernel_size=(7, 7), strides=(5, 5)))
        #https://medium.com/deeper-learning/glossary-of-deep-learning-batch-normalisation-8266dcd2fa82
        #TODO: observation: does not learn at all without this normalization, tbd why
        cnn.add(BatchNormalization())
        cnn.add(Activation('relu'))
        cnn.add(Dropout(0.5))

        cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        cnn.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1)))
        cnn.add(Activation('relu'))
        cnn.add(Dropout(0.5))

        cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        cnn.add(Flatten())
        cnn.add(Dense(200, activation='relu'))
        cnn.add(Dense(self.__numClasses, activation='softmax'))

        cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(), metrics=['accuracy'])
        cnn.fit(xTrainCnn, yTrainCnn, batch_size=32, epochs=50)

        return cnn
