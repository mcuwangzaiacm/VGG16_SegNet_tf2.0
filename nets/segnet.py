from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from nets.convnet import get_convnet_encoder

def segnet_decoder(feat, classes_num):
    # 是否需要重新赋值， feat的原址数据改动是否会造成影响
    segnet_out = feat
    #26,26,512 -> 26,26,512
    segnet_out = ZeroPadding2D((1, 1))(segnet_out)
    segnet_out = Conv2D(512, (3, 3), padding='valid')(segnet_out)
    segnet_out = BatchNormalization()(segnet_out)

    # 26,26,512 -> 52,52,256
    segnet_out = UpSampling2D((2, 2))(segnet_out)
    segnet_out = ZeroPadding2D((1, 1))(segnet_out)
    segnet_out = Conv2D(256, (3, 3), padding='valid')(segnet_out)
    segnet_out = BatchNormalization()(segnet_out)

    # 52,52,256 -> 104,104,128
    segnet_out = UpSampling2D((2, 2))(segnet_out)
    segnet_out = ZeroPadding2D((1, 1))(segnet_out)
    segnet_out = Conv2D(128, (3, 3), padding='valid')(segnet_out)
    segnet_out = BatchNormalization()(segnet_out)

    # 104,104,128 -> 208,208,64
    segnet_out = UpSampling2D((2, 2))(segnet_out)
    segnet_out = ZeroPadding2D((1, 1))(segnet_out)
    segnet_out = Conv2D(64, (3, 3), padding='valid')(segnet_out)
    segnet_out = BatchNormalization()(segnet_out)

    # 208,208,64 -> 208,208,2(classes_num)
    segnet_out = Conv2D(classes_num, (3, 3), padding='same')(segnet_out)

    return segnet_out



def convnet_segnet(classes_num, input_height=416, input_width=416, encoder=3):
    # 从 nets.convent 的 主干网络(vgg部分网络) 导出数据
    img_input, convnet_out = get_convnet_encoder(input_height, input_width)

    # 将特征传入segnet网络
    segnet_out = segnet_decoder(convnet_out, classes_num)

    # 将 segnet网络输出的结果进行 reshape(x, -1) , -1表示未知
    # 行：x = int(input_height/2) * int(input_width/2)  列：-1
    # 注意 这里是  reshape 不是 resize
    segnet_out = Reshape((int(input_height/2) * int(input_width/2), -1))(segnet_out)
    segnet_out = Softmax()(segnet_out)
    model = Model(img_input, segnet_out)
    model.model_name = "convent_segnet"

    return model

