from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Conv2D, Dropout, Input, Add, Activation
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import CSVLogger, ModelCheckpoint

from preprocess import train_generator, TARGET_SIZE, NUM_OF_CLASSES
from utils import BilinearUpSampling2D

KEEP_PROB = 0.5
WEIGHT_DECAY = 1e-3
STRIDE = 32

NUM_TRAIN_SAMPLE = 572  # 424
NUM_TEST_SAMPLE = 143  # 107


def fcn(input_shape, stride, num_of_classes, weight_decay):
    # this could also be the output a different Keras model or layer
    input_tensor = Input(shape=input_shape)  # this assumes K.image_data_format() == 'channels_last'
    image_size = input_shape[:2]

    # create the base pre-trained model
    base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    x = base_model.output

    # add Fully convolutional network
    fc6 = Conv2D(filters=4096,
                 kernel_size=(7, 7),
                 activation='relu',
                 padding='same',
                 dilation_rate=(2, 2),
                 kernel_initializer='he_normal',
                 kernel_regularizer=l2(weight_decay),
                 name='fc6')(x)
    drop6 = Dropout(rate=KEEP_PROB, name='drop6')(fc6)

    fc7 = Conv2D(filters=4096,
                 kernel_size=(1, 1),
                 activation='relu',
                 padding='same',
                 dilation_rate=(2, 2),
                 kernel_initializer='he_normal',
                 kernel_regularizer=l2(weight_decay),
                 name='fc7')(drop6)
    drop7 = Dropout(rate=KEEP_PROB, name='drop7')(fc7)

    # predictor
    score = Conv2D(filters=num_of_classes,
                   kernel_size=(1, 1),
                   activation='linear',
                   padding="valid",
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay),
                   name='score')(drop7)

    # now to upscale to actual image size
    if stride == 32:
        upscore = BilinearUpSampling2D(target_size=image_size, name='upscore32')(score)
    else:
        pool4 = base_model.get_layer('block4_pool').output
        pool4_shape = pool4.get_shape().as_list()
        upscore2 = BilinearUpSampling2D(target_size=tuple(pool4_shape[1:3]), name='upscore2')(score)
        score_pool4 = Conv2D(filters=num_of_classes,
                             kernel_size=(1, 1),
                             activation='linear',
                             padding="valid",
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(weight_decay),
                             name='score_pool4')(pool4)
        fuse_pool4 = Add(name='fuse_pool4')([score_pool4, upscore2])
        if stride == 16:
            upscore = BilinearUpSampling2D(target_size=image_size, name='upscore16')(fuse_pool4)
        elif stride == 8:
            pool3 = base_model.get_layer('block3_pool').output
            pool3_shape = pool3.get_shape().as_list()
            upscore4 = BilinearUpSampling2D(target_size=tuple(pool3_shape[1:3]), name='upscore4')(fuse_pool4)
            score_pool3 = Conv2D(filters=num_of_classes,
                                 kernel_size=(1, 1),
                                 activation='linear',
                                 padding="valid",
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=l2(weight_decay),
                                 name='score_pool3')(pool3)
            fuse_pool3 = Add(name='fuse_pool3')([score_pool3, upscore4])
            upscore = BilinearUpSampling2D(target_size=image_size, name='upscore8')(fuse_pool3)
        else:
            raise NotImplementedError
    upscore = Activation("softmax")(upscore)
    model = Model(inputs=base_model.input, outputs=upscore)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False
    return model


model = fcn(input_shape=(*TARGET_SIZE, 3), stride=STRIDE, num_of_classes=NUM_OF_CLASSES, weight_decay=WEIGHT_DECAY)

# compile the model (should be done *after* setting layers to non-trainable)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='categorical_crossentropy')
print(model.summary())

# continue training if needed
# model = load_model("./models/fcn{}.h5".format(STRIDE), custom_objects={'BilinearUpSampling2D': BilinearUpSampling2D})

# train the model on the new data for a few epochs
csv_logger = CSVLogger('./logs/fcn{}_training.log'.format(STRIDE))
checkpointer = ModelCheckpoint(filepath='./models/fcn_{epoch}.h5', verbose=1, period=5)
model.fit_generator(train_generator("./data/train.txt"), epochs=200, steps_per_epoch=NUM_TRAIN_SAMPLE,
                    validation_data=train_generator("./data/test.txt"), validation_steps=NUM_TEST_SAMPLE,
                    callbacks=[csv_logger, checkpointer])

model.save("./models/fcn{}.h5".format(STRIDE))