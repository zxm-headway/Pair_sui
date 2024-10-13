import os
# 它提供了一个灵活的日志记录系统，用于记录应用程序的运行时信息，以及调试和错误信息。
import logging
# 它提供了处理逗号分隔值（CSV）文件的功能。CSV文件是一种常见的文件格式，用于存储表格数据。
import csv
# OrderedDict是一个有序字典，它记住了字典中元素的添加顺序，可以按照添加顺序遍历字典的键值对。
from collections import OrderedDict

# 创建一个唯一的日志文件名并返回日志文件的编号（log_count）。
# 它接收一个参数 dir_path，表示目录路径。
# 函数的目的是创建一个唯一的日志文件名并返回日志文件的编号（log_count）。
def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    # 判断文件路径是否已经存在，使用os.path.exists(file_path)来检查文件是否存在。
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count

# 用于配置日志记录的设置。函数接受多个参数，包括 folder（日志文件夹路径）、
# name（日志文件名）、level（日志记录的级别）、console_level（控制台输出的日志级别）和 no_console（是否禁用控制台输出）。
def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):

    if not os.path.exists(folder):
        os.makedirs(folder)
    # 清除之前的日志处理程序，以确保不会重复添加处理程序
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    # 清除之前的日志处理程序，以确保不会重复添加处理程序
    logging.root.handlers = []
    # 构建日志文件的路径，并将其存储在logpath变量中。
    logpath = os.path.join(folder, name + ".log")
    # 打印出日志文件路径。
    print("All logs will be saved to %s" %logpath)
    # 将根日志记录器的日志级别设置为level
    logging.root.setLevel(level)
    # 创建一个日志记录的格式化器（formatter）
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 创建一个日志文件处理程序（logfile），使用logging.FileHandler(logpath)构造日志文件处理程序，并将其日志级别设置为level。
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    # 设置日志文件处理程序的格式化器为之前创建的日志记录格式化器。
    logfile.setFormatter(formatter)
    # 将日志文件处理程序添加到根日志记录器中，使用logging.root.addHandler(logfile)完成。
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    # 最后，返回日志文件夹路径folder。
    return folder


