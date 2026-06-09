from tensorflow.keras.layers import MaxPooling2D,Conv2D,Flatten,Dense,Dropout
from tensorflow.keras.models import Sequential

def model_cnn(num_classes):
    CNN=Sequential([
        Conv2D(16,(3,3),activation='relu',input_shape=(128,313,1)),
        MaxPooling2D(2,2),
        Conv2D(32,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128,activation='relu'),
        Dropout(0.3),
        Dense(num_classes,activation='softmax')
    ])
    return CNN