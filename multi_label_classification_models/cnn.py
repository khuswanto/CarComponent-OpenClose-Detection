from keras import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D


def create_model(small=True):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(3, 224, 224)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='sigmoid'))
    return model
