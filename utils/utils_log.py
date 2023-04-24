# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils_log.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-08-26
# * Version     : 0.1.082600
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import datetime
import tensorflow as tf

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def printlog(info):
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 80 + f"{nowtime}")
    print(str(info) + "\n")


@tf.function
def printbar_tf():
    ts = tf.timestamp()
    today_ts = ts % (24 * 60 * 60)
    hour = tf.cast(today_ts / 3600 + 8, tf.int32) % tf.constant(24)
    minute = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return (tf.strings.format("0{}", m))
        else:
            return (tf.strings.format("{}", m))
        
    timestring = tf.strings.join([
        timeformat(hour), 
        timeformat(minute),
        timeformat(second),
    ], separator = ":")
    tf.print("==========" * 8, end = "")
    tf.print(timestring)




__all__ = [
    "printlog",
    "printbar_tf",
]



# 测试代码 main 函数
def main():
    # printbar("test test test")
    printbar_tf()

if __name__ == "__main__":
    main()
