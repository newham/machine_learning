import sys
import time


def print_process(index, size):  # 打印进度
    per = (index+1)/size
    progress(per*100)


def progress(percent, width=50):
    '''进度打印功能'''
    if percent <= 1:
        percent = 1
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' %
                width) % (int(width * percent/100)*">")  # 字符串拼接的嵌套使用
    print('\r%s %d%%' % (show_str, percent), end='')
    if percent == 100:
        print(" OK")


def main():

    for i in range(1, 100):
        time.sleep(0.01)
        print_process(i, 100)


if __name__ == '__main__':
    main()
