from keras.applications import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# path to training and validation data
train_data_loc = r"C:\\Users\\kille\\ML_Assignment_2\\charts"
val_data_loc = r"C:\\Users\\kille\\ML_Assignment_2\\charts"

# size of the input images
img_width, img_height = 224, 224

# an instance of the VGG16 model with pre-trained weights
b_mod = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the layers of the pre-trained model
for layer in b_mod.layers:
    layer.trainable = False

# new layers to the pre-trained model
x = b_mod.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(1, activation='sigmoid')(x)

# new model with the pre-trained model
mod = Model(inputs=b_mod.input, outputs=pred)

# Compile the model with a binary crossentropy loss and an Adam optimizer
mod.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])

# Set up data augmentation for the training data and validation data
trn_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)


batch_size = 16

# number of training and validation samples
nb_train_samp = 1600
nb_val_samp = 400

# number of epochs
epochs = 10

# Train the model with the data generators
hist = mod.fit(
    trn_datagen.flow_from_directory(train_data_loc, target_size=(img_width, img_height), batch_size=batch_size,
                                    class_mode='binary'),
    steps_per_epoch=nb_train_samp // batch_size,
    epochs=epochs,
    validation_data=val_datagen.flow_from_directory(val_data_loc, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='binary'),
    validation_steps=nb_val_samp // batch_size)

# Evaluate the model on the test data
tst_datagen = ImageDataGenerator(rescale=1. / 255)
test_data_loc = "C:\\Users\\kille\\ML_Assignment_2"

test_generator = tst_datagen.flow_from_directory(
    test_data_loc,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_loss, test_acc = mod.evaluate(test_generator, steps=len(test_generator))

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
