# -*- coding: utf-8 -*-

# ***************************************************
# * File        : email.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-21
# * Version     : 0.1.042101
# * Description : description
# * Link        : https://github.com/lyhue1991/bugrobot
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import smtplib
import traceback 
from email.mime.text import MIMEText

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


'''
def send_msg(receivers, subject, msg = ""):
    """
    send message to Email

    Examples:
    ---------
    >> subject = "info@train_model.py"  # 邮件主题
    >> msg = "auc=0.98" # 邮件内容
    >> receivers = ["265011xxxx@qq.com"]  # 收件人
    >> send_msg(receivers, subject, msg)
    """
    # 设置服务器所需信息
    mail_host = 'smtp.yeah.net'  
    mail_user = 'bugrobot'  
    mail_pass = 'NPWPJBSIVXRTYUOB'   #密码(部分邮箱为授权码) 
    sender = 'bugrobot@yeah.net'

    # 构造邮件内容
    message = MIMEText(msg, 'plain', 'utf-8')  
    message['Subject'] = subject
    message['From'] = sender     
    message['To'] = receivers[0]  

    # 登录并发送邮件
    try:
        smtpObj = smtplib.SMTP() 
        #连接到服务器
        smtpObj.connect(mail_host, 25)
        #登录到服务器
        smtpObj.login(mail_user, mail_pass) 
        #发送
        smtpObj.sendmail(sender, receivers, message.as_string()) 
        #退出
        smtpObj.quit() 
        return 'send_msg success'
    except smtplib.SMTPException as e:
        error = 'send_msg error : ' + str(e)
        print(error)
        return error


def monitor_run(function, receivers):
    """
    send the bug information to Email if any exception occor

    Examples:
    ---------
    >> receivers = ["265011xxxx@qq.com"] #收件人
    >> def f():
    >>    return 1/0
    >> monitor_run(f,receivers)
    """
    try:
        function()
    except Exception as e:
        error_msg = traceback.format_exc()
        send_msg(receivers, "error@" + function.__name__, error_msg)
        raise e
'''



# 测试代码 main 函数
def main():
    import bugrobot
    receivers = ["wangzhefengr@163.com"]  # 收件人
    subject = "info@train_model.py"  # 邮件主题
    
    # 发送模型运行信息
    msg = "auc=0.98" # 邮件内容
    bugrobot.send_msg(receivers, subject, msg)

    # 异常发生时
    def f():
        return 1 / 0
    bugrobot.monitor_run(f, receivers)


if __name__ == "__main__":
    main()
