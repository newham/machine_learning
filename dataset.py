import numpy
import matplotlib.pyplot as plt


def get_data_list(file_path):
    data_file = open(file_path, "r")
    data_list = data_file.readlines()
    data_file.close()
    return data_list


def get_img_label(data):
    return int(data[0])


def get_scaled_data(data):
    all_values = data.split(',')
    scaled_data = numpy.asfarray(all_values[1:])/255*0.99+0.01
    return scaled_data


def get_all_values(data):
    return data.split(',')


def get_img_array(data):
    all_values = data.split(',')
    image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    return image_array


def get_img_data(data):
    all_values = data.split(',')
    image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    label = get_img_label(all_values)
    return label, image_array


def show_img(data):
    plt.imshow(data, cmap='gray')
    plt.show()
    pass


def save_img(data, name):
    plt.imsave(name, data, cmap='gray')
    pass


def load_img(name):
    img_data = plt.imread(name)
    rgb_weight = [0.2125, 0.7154, 0.0721, 0]
    img_data_gray = numpy.dot(img_data, rgb_weight)*0.99+0.01
    return img_data_gray.reshape(28*28)
