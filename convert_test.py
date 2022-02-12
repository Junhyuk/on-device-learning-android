import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter
import tensorflow.keras.backend as K

DEFAULT_BATCH_SIZE = 32

input_size = 224
output_size = 5

DEFAULT_INPUT_SIZE = 64
DEFAULT_BATCH_SIZE = 16
LEARNING_RATE = 0.001

def test_mobilenet_v2_saved_model():
    input_size = DEFAULT_INPUT_SIZE
    output_size = 4
    base = bases.MobileNetV2Base(image_size=input_size)

    head_model = tf.keras.Sequential([
        layers.Flatten(input_shape=(7, 7, 1280)),
        layers.Dropout(0.25),
        layers.Dense(
            units=32,
            activation='relu',
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01)),
        layers.Dropout(0.25),
        layers.Dense(
            units=32,
            activation='relu',
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01)),
        layers.Dense(
            units=4,
            activation='softmax',
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01)),
    ])

    head_model.compile(loss="categorical_crossentropy", optimizer="sgd")
    converter = TFLiteTransferConverter(
        output_size, base, heads.KerasModelHead(head_model),
        optimizers.SGD(LEARNING_RATE), DEFAULT_BATCH_SIZE)

    models = converter.convert_and_save('custom_keras_on_device_model')
    models = converter._convert()

test_mobilenet_v2_saved_model()

