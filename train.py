import tensorflow.keras
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.optimizers import Adam
from PIL import Image
import tensorflow as tf

from nets.segnet import convnet_segnet

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# ---------------------------------------------#
#   定义输入图片的高和宽，以及种类数量
#   others + 斑马线 = 2
# ---------------------------------------------#
HEIGHT = 416
WIDTH = 416
NCLASSES = 2


def main():
    log_dir = "logs/"
    model = convnet_segnet(classes_num=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
    # ---------------------------------------------------------------------#
    # 加载主干特征提取网络vgg16 的权重、使用的是迁移学习的思想
    # by_name = False
    # 的时候按照网络的拓扑结构加载权重
    # by_name = True
    # 的时候就是按照网络层名称进行加载
    # ---------------------------------------------------------------------#
    weights_path = "models/vgg16_weights_tf.h5"
    model.load_weights(weights_path, by_name=True)

    # 打开训练数据集的txt
    with open("dataset/train.txt", "r") as f:
        lines = f.readlines()

    # ---------------------------------------------#
    #   将数据打乱 更有利于训练
    #   训练 ： 验证  =  9 ： 1
    #   参考的代码中要 np.random.seed(10101) 不明白为什么要 我认为无所谓
    # ---------------------------------------------#
    # np.random.seed(10101)
    np.random.shuffle(lines)
    # np.random.seed(None)
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # -------------------------------------------------------------------------------#
    #   训练参数的设置
    #   checkpoint用于设置权值保存的细节，period_save用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    # -------------------------------------------------------------------------------#
    period_save = 10
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=period_save)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # -------------------------------------------------------------------------------#
    #   这里使用的是迁移学习的思想，主干部分提取出来的特征是通用的
    #   所以我们可以不训练主干部分先，因此训练部分分为两步，分别是冻结训练和解冻训练
    #   冻结训练是不训练主干的，解冻训练是训练主干的。
    #   由于训练的特征层变多，解冻后所需显存变大
    # -------------------------------------------------------------------------------#
    trainable_layer = 15
    for i in range(trainable_layer):
        model.layers[i].trainable = False
        print(model.layers[i].name)
    print('freeze the first {} layers of total {} layers.'.format(trainable_layer, len(model.layers)))

    if True:
        lr = 1e-3
        batch_size = 4
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr),
                      metrics=['accuracy'])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[checkpoint, reduce_lr, early_stopping])

    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    if True:
        lr = 1e-4
        batch_size = 4
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr),
                      metrics=['accuracy'])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=100,
                            initial_epoch=50,
                            callbacks=[checkpoint, reduce_lr, early_stopping])


# 若训练数据量很大的项目，使用model.fit() 则一次送进所有训练数据会导致内存泄漏
# 所以若数据量很大，则采用分批次训练，每个批次送入训练batch_size份数据(1份 = 1张训练图 + 1张验证图)
# 采用model_fit_generator()进行训练，传入的不是数据文件，是生成器(yield),训练数据是通过该生成器产生的
def generate_arrays_from_file(lines, batch_size):
    n = len(lines)
    i = 0
    while True:
        X_train = []
        Y_train = []
        for _ in range(batch_size):
            if i == 0:  # 一个epoch训练结束，重新打散
                np.random.shuffle(lines)
            # -------------------------------------#
            #   读取  训练图片 并进行 归一化和resize
            #   BICUBIC -> 双三次插值法
            # -------------------------------------#
            name = lines[i].split(';')[0]
            img = Image.open("dataset/jpg/" + name)
            img = img.resize((HEIGHT, WIDTH), Image.Resampling.BICUBIC)
            img = np.array(img) / 255
            X_train.append(img)

            # -------------------------------------#
            #   读取  标签图片 并进行 归一化和resize
            #   NEAREST -> 最近邻插值法
            #   这里 resize 的尺寸 要和 convnet_segnet中最终的尺寸一样
            #   最后展开成 n行 自适应列的 形状
            # -------------------------------------#
            name = lines[i].split(';')[1].split()[0]
            label = Image.open("dataset/png/" + name)
            label = label.resize((int(WIDTH / 2), int(HEIGHT / 2)), Image.Resampling.NEAREST)

            #  这里任意取 png 中的一个通道，因为2个通道都是相等的
            if len(np.shape(label)) == 3:
                label = np.array(label)[:, :, 0]
            # 展开成 一维的 矩阵 shape为（921600，） 而不是(921600，1)->这个是2维的
            # def reshape(a, newshape, order='C') ,若改成(-1, 1),则shape变成了(921600, 1)
            label = np.reshape(np.array(label), [-1])
            one_hot_label = np.eye(NCLASSES)[np.array(label, np.int32)]
            Y_train.append(one_hot_label)

            i = (i + 1) % n
        yield (np.array(X_train), np.array(Y_train))

if __name__ == "__main__":
    main()
