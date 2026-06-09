import tensorflow as tf
from dataset.bird_dataset import BirdDataset
from training.models import model_cnn
from sklearn.model_selection import train_test_split

# import os
# print(os.getcwd())
dataset=BirdDataset("dataset/processed/logmel_128")
X,y=dataset.get_data()

tf_dataset=tf.data.Dataset.from_tensor_slices(
    (X,y)
)

cnn=model_cnn(num_classes=len(dataset.label_map))
cnn.summary()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

train_ds=tf.data.Dataset.from_tensor_slices((X_train,y_train)).shuffle(len(X_train)).batch(64).prefetch(tf.data.AUTOTUNE)
test_ds=tf.data.Dataset.from_tensor_slices((X_test,y_test)).shuffle(len(X_test)).batch(64).prefetch(tf.data.AUTOTUNE)

cnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=cnn.fit(train_ds,validation_data=test_ds,epochs=10)