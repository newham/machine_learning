def print_process(index, size):  # 打印进度,放在for循环里，其中index指单次循环的索引，size是集合的长度，一般放在for循环的最后一列
    per = (index + 1) / size
    progress(per * 100)


def progress(percent, width=50):
    '''进度打印功能'''
    if percent <= 1:
        percent = 1
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' %
                width) % (int(width * percent / 100) * ">")  # 字符串拼接的嵌套使用
    print('\r%s %d%%' % (show_str, percent), end='')
    if percent == 100:
        print(" OK")


def get_data_list(file_path):
    data_file = open(file_path, "r")
    data_list = data_file.readlines()
    data_file.close()
    return data_list


kdd_labels = {}


def get_data_array(record):
    data_array = record.rstrip('\n').split(',')
    return data_array


def analysis():
    data_list = get_data_list("kdd/kddcup.data.corrected")
    size = len(data_list)
    for index, record in enumerate(data_list):
        data_array = get_data_array(record)
        label = data_array[-1]
        if label in kdd_labels:
            kdd_labels[label] = kdd_labels[label] + 1
        else:
            kdd_labels[label] = 1
        print_process(index, size)

    print(kdd_labels)


def main():
    analysis()


if __name__ == '__main__':
    main()
