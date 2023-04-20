import os

import numpy as np
import pandas as pd

os.environ.setdefault('TF_KERAS', '1')
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from efficientnet.tfkeras import EfficientNetB4
import tensorflow.keras.backend as K
import warnings
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score

nb_classes = 6
batch_size = 32
img_size = 224
nb_epochs = 2


def score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # 归一化
    rotation_range=45,  # 旋转角度
    width_shift_range=0.1,  # 水平偏移
    height_shift_range=0.1,  # 垂直偏移
    shear_range=0.1,  # 随机错切变换的角度
    zoom_range=0.25,  # 随机缩放的范围
    horizontal_flip=True,  # 随机将一半图像水平翻转
    fill_mode='nearest'
)  # 填充像素的方

test_datagen = ImageDataGenerator(
    rescale=1. / 255
)

import tensorflow.keras as keras


def get_model():
    models = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3),
                            backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    # models.load_weights(path)
    x = models.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model_final = Model(inputs=models.input, outputs=predictions)
    model_final.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',
                        metrics=['acc'])
    return model_final


train_path = 'train/'
test_path = 'validation/'

train = pd.read_csv('train.csv')
train.columns = ['id', 'label']

# 构建提交数据集的样例
file_name = os.listdir('./validation')
test = pd.DataFrame()
test['id'] = file_name
test['label'] = -1

from sklearn.model_selection import StratifiedKFold

data = train.copy()
print(data.groupby(['label']).size())


def get_s(model):
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    oof_train = np.zeros((len(data), nb_classes))
    oof_test = np.zeros((len(test), nb_classes))
    for index, (tra_index, val_index) in enumerate(skf.split(data['id'].values, data['label'].values)):
        K.clear_session()
        print('========== {} =========='.format(index))
        train = pd.DataFrame({'id': data.iloc[tra_index]['id'].values,
                              'label': data.iloc[tra_index]['label'].values})
        print(train.head())
        train['label'] = train['label'].astype(str)

        valid = pd.DataFrame({'id': data.iloc[val_index]['id'].values,
                              'label': data.iloc[val_index]['label'].values})

        valid['label'] = valid['label'].astype(str)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train,
            directory="./train",
            x_col="id",
            y_col="label",
            batch_size=batch_size // 4,
            shuffle=True,
            class_mode="categorical",
            target_size=(img_size, img_size),
            # save_format='JPEG'
            # verbose=False
        )

        valid_generator = test_datagen.flow_from_dataframe(
            dataframe=valid,
            directory="./train",
            x_col="id",
            y_col="label",
            batch_size=1,
            shuffle=False,
            class_mode="categorical",
            # verbose=False,
            target_size=(img_size, img_size),
            # save_format='JPEG'
        )

        model_final = get_model()
        if index == 0: model_final.summary()
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, min_lr=0.0001, verbose=True)
        earlystopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=True, mode='max')
        checkpoint = ModelCheckpoint("load_{}_{}.h5".format(model, index), monitor='val_acc', verbose=False,
                                     save_best_only=True, save_weights_only=True, mode='max')
        model_final.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_generator.n // train_generator.batch_size,
                                  validation_data=valid_generator,
                                  validation_steps=valid_generator.n // valid_generator.batch_size,
                                  epochs=nb_epochs,
                                  callbacks=[checkpoint, earlystopping, reduce_lr],
                                  verbose=True)

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test,
            directory="./tmp_validation",
            x_col="id",
            target_size=(img_size, img_size),
            batch_size=1,
            # save_format='JPEG',
            shuffle=False,
            class_mode=None
        )
        print('predict', index)
        model_final.load_weights("load_{}_{}.h5".format(model, index))

        oof_test += model_final.predict_generator(test_generator) / skf.n_splits

        predict = model_final.predict_generator(valid_generator)
        oof_train[val_index] = predict

    return oof_train, oof_test, train_generator


oof_train4, oof_test4, train_generator = get_s('EfficientNetB5')

oof_train = oof_train4
oof_test = oof_test4

map_index = train_generator.class_indices
print(map_index)
data['label'] = data['label'].map(map_index)


def invert_dict(d):
    return {v: k for k, v in d.items()}


map_index = invert_dict(map_index)
print(map_index)


def new_index(x):
    p = []
    for i in x:
        p.append(map_index[i])
    return p


print(oof_train)
base_score = score(data['label'].values, np.argmax(oof_train, axis=1))
print(base_score)
predicted_class_indices = new_index(np.argmax(oof_test, axis=1))
test['label'] = list(predicted_class_indices)
test.to_csv('./submit_{}.csv'.format(str(base_score).replace('.', '_')), index=False)
