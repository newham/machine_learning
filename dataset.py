from numpy import asfarray


def get_data_list(file_path):
    data_file = open(file_path, "r")
    data_list = data_file.readlines()
    data_file.close()
    return data_list


def get_img_label(data):
    return int(data[0])

# 255 ->0.9


def get_scaled_data(data):
    all_values = data.split(',')
    scaled_data = asfarray(all_values[1:])/255*0.99+0.01
    return scaled_data

def get_all_values(data):
    return data.split(',')

def get_img_array(data):
    all_values = data.split(',')
    image_array = asfarray(all_values[1:]).reshape((28, 28))
    return image_array
