from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


batch_size = 128 #Definição de bacth: https://www.youtube.com/watch?v=nhqo0u1a6fw&t=146s
num_classes = 10
epochs = 12

img_rows, img_cols = 28, 28

# carrega automaticamente o dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#tranforma as matrizes das imagens em arrays para facilitar a computação da rede neural
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

#cria automaticamente o encoding para cada label do dataset
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#definição da rede neural. Dica: tente colocar mais convoluções, MLP's e veja os resultados
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#otimizadores: mesmo video do batch
#loss: https://arxiv.org/abs/1702.05659
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#nesse momento o algoritmo de backpropagation faz o treinamento usando (x_train y_train) e avaliando os resultados com (x_test, y_test)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#print dos resultados do treinamento
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])