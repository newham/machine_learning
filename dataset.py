import numpy
import matplotlib.pyplot as plt


def get_data_list(file_path):
    data_file = open(file_path, "r")
    data_list = data_file.readlines()
    data_file.close()
    return data_list


def get_img_label(data):
    return int(data[0])


def get_kdd_scaled_data(data):
    pass


def get_scaled_data(data):
    all_values = data.split(',')
    scaled_data = numpy.asfarray(all_values[1:])/255*0.99+0.01
    label = int(all_values[0])
    return label, scaled_data


def get_scaled_inputs(data):
    inputs = numpy.asfarray(data[:]) / 255 * 0.99 + 0.01
    return inputs


def get_img_array(data, row, col):
    all_values = data.split(',')
    image_array = numpy.asfarray(all_values[1:]).reshape((row, col))
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
    img_data_gray = read_img(name)
    row, col = numpy.shape(img_data_gray)
    return img_data_gray.reshape(row*col)


def read_img(name):
    img_data = plt.imread(name)
    if len(img_data[0][0]) == 4:
        rgb_weight = [0.2125, 0.7154, 0.0721, 0]
    else:
        rgb_weight = [0.2125, 0.7154, 0.0721]
    img_data_gray = numpy.dot(img_data, rgb_weight)*0.99+0.01
    return img_data_gray


def get_targets_data(size, index):
    targets = numpy.zeros(size)+0.01
    targets[index] = 0.99
    return targets


def get_kdd_CICIDS_data(file_path):
    data_list = get_data_list(file_path)
    label_name = data_list[0]
    return label_name, data_list[1:]


def get_kdd_UNSW_data(file_path):
    data_list = get_data_list(file_path)
    return data_list[1:]


def normalize(data):
    m = numpy.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]


def relu(data):
    in_row, in_col = numpy.shape(data)
    for i in range(0, in_row):
        for j in range(0, in_col):
            if data[i][j] < 0:
                data[i][j] = 0
    return data
