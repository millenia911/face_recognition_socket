import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow_addons import losses as tfa_losses
import numpy as np

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# architecture
backbone = tf.keras.applications.efficientnet.EfficientNetB0(
                                                input_shape=(64, 64, 3),
                                                include_top=False,
                                                weights= None)

backbone.trainable = False

EFFICIENTNET_LEVEL_ENDPOINTS = [
    'block1a_project_bn',
    'block2b_add',
    'block3b_add',
    'block5c_add',
    'block7a_project_bn',
]
output_layers = [backbone.get_layer(layer_name).output
                 for layer_name in EFFICIENTNET_LEVEL_ENDPOINTS]

base_model = Model(inputs=backbone.input, outputs=output_layers)

def conv_block_up(input, iteration, kernel_size=(3,3), strides_size=(1,1), upsample=True):
    b = input
    for i in range(iteration):
        # expand
        b = layers.Conv2D(128, kernel_size=(1,1), padding='same')(b)
        b = layers.BatchNormalization()(b)
        b = layers.Activation('swish')(b)
        # depthwise
        b = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides_size, padding='same')(b)
        b = layers.BatchNormalization()(b)
        b = layers.Activation('swish')(b)
        _b = b
        # squeeze-exitacion
        b = layers.GlobalAveragePooling2D()(b)       #Sqz
        b = layers.Reshape((1,1,-1))(b)              #Rsh
        b = layers.Conv2D(4, kernel_size=(1,1))(b)   #Red
        b = layers.Conv2D(128, kernel_size=(1,1))(b) #Exp
        b = layers.Multiply()([_b, b])               #Ext  
        # projection
        b = layers.Conv2D(96, kernel_size=(1,1), padding='same')(b)
        b = layers.BatchNormalization()(b)
        b = layers.Activation('swish')(b)
        if upsample:
          b = layers.UpSampling2D(size=(2,2), interpolation='bilinear')(b)

    return b

# output_6 = conv_block_up(base_model.output[5], 4)
output_5 = conv_block_up(base_model.output[4], 4)
output_4 = conv_block_up(base_model.output[3], 3)
output_3 = conv_block_up(base_model.output[2], 2)
output_2 = conv_block_up(base_model.output[1], 1)
output_1 = conv_block_up(base_model.output[0], 1, upsample=False)

# x = layers.Add()([output_6, output_5])
x = layers.Add()([output_5, output_4])
x = layers.Add()([x, output_3])
x = layers.Add()([x, output_2])
x = layers.Add()([x, output_1])

x = layers.Dropout(0.3)(x)
x = layers.Conv2D(128, kernel_size=(5,5), strides=(4,4), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('swish')(x)

x = layers.Dropout(0.3)(x)
x = layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('swish')(x)

x = layers.Dropout(0.3)(x)
x = layers.Conv2D(7, kernel_size=(3,3), strides=(2,2), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.GlobalMaxPool2D()(x)
x = layers.Activation('softmax')(x)

final_model = Model(inputs=base_model.input, outputs=x)
final_model.summary()

final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7), 
                    loss= tfa_losses.SigmoidFocalCrossEntropy(reduction=tf.keras
                                                              .losses.Reduction
                                                              .AUTO),
                    metrics=['accuracy'])

final_model.trainable = False
final_model.load_weights("./ckpt0001\w02_dw_swish\ckpt01")


###################################################
# inference
###################################################
from keras.preprocessing import image
import cv2

path = 'cropped.png'
img = image.load_img(path, target_size=(64,64))

x = image.img_to_array(img)  
x = np.expand_dims(x, axis=0)
# x = x * (1./255)        
images = np.vstack([x])

classes = final_model.predict(images)  
print('--------------------------------------------------')
confidence = (np.amax(classes)/1)*100
predicted_class_index = np.argmax(classes)
predicted_class = class_names[predicted_class_index]
print(path + ' adalah gambar ' + '"' + predicted_class + '"')
print(f'dengan confidence = {confidence}%')