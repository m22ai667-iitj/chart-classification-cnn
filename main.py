import numpy as np
import tensorflow as tensor
from tensorflow import keras
from keras.models import Sequential
import pandas as pd
from matplotlib import pyplot as plt
% matplotlib inline
import os
import cv2
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# paths to image and csv
train_path = r"C:\\Users\\kille\\ML_Assignment_2\\charts\\train_val"
test_path = r"C:\\Users\\kille\\ML_Assignment_2\\test"
train_path_lab = r"C:\\Users\\kille\\ML_Assignment_2\\charts\\train_path_labels.csv"
train_val_lab = pd.read_csv(train_path_lab)

# loading training dataset in array(numpy)
imgs = []
lbls = []
for f_name in os.listdir(train_path):
    if f_name.endswith('.png'):
        # Load and resize imgs to (128, 128) with 3 color channels
        img = cv2.imread(os.path.join(train_path, f_name))
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_array = np.array(img)
        # Append the array to the list of images
        imgs.append(img_array)
        lbls.append(f_name)

# Converting string labels to numerical labels
lab_enc = LabelEncoder()
lbls = lab_enc.fit_transform(lbls)

# Converting lists to NumPy arrays
imgs = np.array(imgs)
lbls = np.array(lbls)
# Save the arrays in NumPy format
np.save('x_train.npy', imgs)
np.save('y_train.npy', lbls)
x_trn = np.load('x_train.npy')
y_trn = np.load('y_train.npy')

x_trn.shape

x_trn[:5]
y_trn[:5]

# load test dataset in numpy array
imgs = []
lbls = []
for f_name in os.listdir(test_path):
    if f_name.endswith('.png'):
        # Load the images and resize them to (128, 128) with 3 color channels
        img = cv2.imread(os.path.join(test_path, f_name))
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        abs  # img = Image.open(os.path.join(test_dir, filename))
        img_array = np.array(img)
        # Append the array to the list of images
        imgs.append(img_array)
        lbls.append(f_name)

# Convert the string labels to numerical labels
lab_enc = LabelEncoder()
lbls = lab_enc.fit_transform(lbls)

# Convert the lists to NumPy arrays
imgs = np.array(imgs)
lbls = np.array(lbls)
# Save the arrays in NumPy format
np.save('x_test.npy', imgs)
np.save('y_test.npy', lbls)
x_tst = np.load('x_test.npy')
y_tst = np.load('y_test.npy')

x_tst.shape

# check the images loaded
plt.figure(figsize=(10, 2))
plt.imshow(x_trn[10])
plt.imshow(x_trn[208])
plt.imshow(x_trn[444])


image_classes = ['line', 'dot_line', 'hbar_categorical', 'vbar_categorical', 'pie']
image_classes[0]

label_map = {'line': 0, 'dot_line': 1, 'hbar_categorical': 2, 'vbar_categorical': 3, 'pie': 4}
y_trn = np.array([label_map[label] for label in train_val_lab['type']])
y_trn
y_trn.shape
y_tst.shape


# function to test the chart sample
def img_smp(x, y, index):
    plt.figure(figsize=(10, 2))
    plt.imshow(x[index])
    plt.xlabel(image_classes[y[index]])


img_smp(x_trn, y_trn, 0)
img_smp(x_trn, y_trn, 208)
img_smp(x_trn, y_trn, 444)

# normalizing the image
x_trn = x_trn / 255
x_tst = x_trn / 255

x_tst.shape


y_trn_index = train_val_lab['image_index']
y_trn_type = train_val_lab['type']

y_trn_type[:5]

# simple nn to test

model_final = Sequential([
    Flatten(input_shape=(128, 128, 3)),
    Dense(3000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(5, activation='softmax')
])
# Compiling the model
model_final.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_final.fit(x_trn, y_trn, epochs=10)

# Splitting the images and labels into training and validation sets
from sklearn.model_selection import train_test_split

x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn, y_trn, test_size=0.2, random_state=42)

model_final.evaluate(x_tst, y_tst)

y_pred = model_final.predict(x_tst)
y_pred
y_pred_classes = [np.argmax(ele) for ele in y_pred]
# print("classificaton report : \n",classification_report(y_test,y_pred_classes))


# since the accuracy is very low and we modify our nn

print("Train Img Shape:", x_trn.shape)
print("Train Labels Shape:", y_trn.shape)
print("Test Img Shape:", x_tst.shape)
print("Test Labels Shape:", y_tst.shape)

# modify the model architecture to cmnn
cnn_model_final = Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])
# Compile the model
cnn_model_final.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
history = cnn_model_final.fit(x_trn, y_trn, batch_size=1000, epochs=50, validation_data=(x_tst, y_tst))
# Plot the obtained loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

cnn_model_final.evaluate(x_tst, y_tst)

img_smp(x_tst, y_tst, 1)
img_smp(x_tst, y_tst, 50)
img_smp(x_tst, y_tst, 25)
img_smp(x_tst, y_tst, 30)

# we can see some wrong predictions

y_pred = cnn_model_final.predict(x_tst)
y_pred[:5]

y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]

y_tst[:5]

# test actual and predicted

img_smp(x_tst, y_tst, 15)
image_classes[y_classes[15]]


print("classification Summary: \n", classification_report(y_tst, y_classes))

# Generate the confusion matrix
conf_mat = confusion_matrix(y_tst, y_classes)
print('Confusion Matrix:')
print(conf_mat)

# Plot the confusion matrix
import seaborn as sn

plt.figure(figsize=(10, 10))
sn.heatmap(conf_mat, annot=True, fmt='d')
plt.xlabel('Predicted')


from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Loading the pre-trained model
vgg16_mod = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Replace the final classification layer with a new layer
x = vgg16_mod.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)
pt_model = tensor.keras.Model(inputs=vgg16_mod.input, outputs=predictions)

# Freeze the weights of all layers except the new classification layer
for layer in pt_model.layers:
    layer.trainable = False

# Compile the model with categorical crossentropy loss and Adam optimizer
pt_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

pt_model.summary()

# Set up data generators for image augmentation and feeding data to the model
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1. / 255)

# flow method generates batches of augmented data
train_generator = train_datagen.flow(x_trn, y_trn, batch_size=32)
test_generator = train_datagen.flow(x_tst, y_tst, batch_size=32)

# Train the model with early stopping
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
history = pt_model.fit(train_generator, epochs=100, validation_data=test_generator, callbacks=[es])
