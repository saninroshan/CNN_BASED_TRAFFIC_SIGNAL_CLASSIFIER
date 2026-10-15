import tensorflow as tf
from tensorflow.keras import layers,models
train_dataset=tf.keras.utils.image_dataset_from_directory(
    r"C:\SANIN\Internship\materials\DATA",validation_split=0.2,subset="training",seed=123,image_size=(128,128),batch_size=32
)
validation_dataset=tf.keras.utils.image_dataset_from_directory(
        r"C:\SANIN\Internship\materials\DATA",validation_split=0.2,subset="training",seed=123,image_size=(128,128),batch_size=32
)

class_names=train_dataset.class_names
print("classes",class_names)
model=models.Sequential([
    layers.Rescaling(1/255,input_shape=(128,128,3)),layers.Conv2D(32, (3,3), activation='relu'),layers.MaxPooling2D(),layers.Conv2D(64, (3,3), activation='relu'),layers.MaxPooling2D(),layers.Flatten(),layers.Dense(128, activation='relu'),layers.Dense(len(class_names), activation='softmax')
])
model.compile(
    optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']
)
model.fit(
    train_dataset,validation_data=validation_dataset,epochs=1000
)
model.save("traffic_classifier.keras")
print("model saved successfully")
