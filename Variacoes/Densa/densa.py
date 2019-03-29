from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
import matplotlib.pyplot as plt
componentes =  Sequential()

componentes.add(Conv2D(32, (3, 3),input_shape= (150,150, 3) , activation='relu'))
componentes.add(Conv2D(32, (3, 3), activation='relu'))
componentes.add(MaxPooling2D((2,2), strides=(2,2)))


componentes.add(Conv2D(64, (3, 3), activation='relu'))
componentes.add(MaxPooling2D((2,2), strides=(2,2)))
componentes.add(Flatten())
componentes.add(Dense(units=512, activation='relu'))
componentes.add(Dropout(0.5))
componentes.add(Dense(units=256, activation='relu'))
componentes.add(Dropout(0.3))
componentes.add(Dense(units=128, activation='relu'))
componentes.add(Dropout(0.3))
componentes.add(Dense(units=5, activation='softmax'))

componentes.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 25
num_p_treino = 1417 #verificar no final
num_p_teste = 340 #idem

training_set = train_datagen.flow_from_directory('Dados/treino',
                                                 target_size = (150,150),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Dados/teste',
                                            target_size = (150,150),
                                            batch_size = batch_size,
                                            class_mode = 'categorical')


Dados_Grafs = componentes.fit_generator(training_set,
                         steps_per_epoch = (num_p_treino/batch_size), 
                         epochs = 500,
                         validation_data = test_set,
                         validation_steps = (num_p_teste/batch_size))

componentes.save_weights('pesos_def.h5')


# list all data in history
print(Dados_Grafs.history.keys())
# summarize history for accuracy
plt.figure(1)
plt.plot(Dados_Grafs.history['categorical_accuracy'])
plt.plot(Dados_Grafs.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.figure(2)
plt.plot(Dados_Grafs.history['loss'])
plt.plot(Dados_Grafs.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


