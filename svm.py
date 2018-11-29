import dataset
import numpy
import cv2


# 1.读取灰度照片
imgs = ["img/svm/dog_head_2.png",
        "img/svm/dog_head_1.png", "img/svm/me_head.png"]

# 对图像进行卷积
# 这个是设置的滤波，也就是卷积核
core = numpy.array([[-1, -1, 0],
                    [-1, 0, 1],
                    [0, 1, 1]])
for img in imgs:
    # 1.1显示照片
    img_data = dataset.read_img(img)
    dataset.show_img(img_data)
    for i in range(0, 5):
        ater_convolution_head_img_data = dataset.convolute(img_data, core)
        # dataset.show_img(ater_convolution_head_img_data)
        ater_pooling_head_img_data = dataset.pooling(
            ater_convolution_head_img_data, 2, 2)
        dataset.show_img(ater_pooling_head_img_data)
        img_data = ater_pooling_head_img_data
# 对图像进行池化
