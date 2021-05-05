from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras import regularizers


def get_model(image_channels: int = 1, classes: int = 3):
    model = models.Sequential()
    model.add(Conv2D(16, kernel_size=3,
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.),
                     activity_regularizer=regularizers.l2(0.),
                     input_shape=(224, 224, image_channels)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.05))

    model.add(Conv2D(32, kernel_size=3, activation='relu',
                     kernel_regularizer=regularizers.l2(0.),
                     activity_regularizer=regularizers.l2(0.)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.05))

    model.add(Flatten())
    model.add(Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(0.),
                    activity_regularizer=regularizers.l2(0.)))
    model.add(Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.),
                    activity_regularizer=regularizers.l2(0.)))
    model.add(Dropout(0.05))
    model.add(Dense(classes, activation='softmax'))
    return model


if __name__ == '__main__':
    face_recognition_model = get_model()
    print(face_recognition_model.summary())
