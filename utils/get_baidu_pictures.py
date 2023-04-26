# -*- coding: utf-8 -*-

# ***************************************************
# * File        : download_baidu_pictures.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-25
# * Version     : 0.1.042519
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
import re
import time
import datetime
from tqdm import tqdm
from urllib.parse import quote

import requests

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class _BaiduPictures:

    def __init__(self, keyword, needed_pics_num = 100, save_dir = None):
        from fake_useragent import UserAgent 
        
        self.save_dir = save_dir if save_dir is not None else './{}'.format(keyword)
        self.name_ = keyword
        self.name = quote(self.name_) 
        self.needed_pics_num = needed_pics_num
        self.times = str(int(time.time()*1000)) 
        self.url = (
            'https://image.baidu.com/search/acjson?'
            'tn=resultjson_com&logid=8032920601831512061&'
            'ipn=rj&ct=201326592&is=&fp=result&fr=&word={}&'
            'cg=star&queryWord={}&cl=2&lm=-1&ie=utf-8&'
            'oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&'
            'copyright=&s=&se=&tab=&width=&height=&face=&'
            'istype=&qc=&nc=1&expermode=&nojc=&isAsync=&'
            'pn={}&rn=30&gsm=1e&{}='
        )
        self.headers = {'User-Agent': UserAgent().random}

    def get_one_html(self, url, pn):
        response = requests.get(
            url = url.format(self.name, self.name, pn, self.times), 
            headers = self.headers
        ).content.decode('utf-8')
        return response
    
    def parse_html(self, regex, html):
        content = regex.findall(html)
        return content
    
    def get_two_html(self, url):
        response = requests.get(url = url, headers = self.headers).content
        return response

    def run(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        response = self.get_one_html(self.url, 0)
        regex1 = re.compile('"displayNum":(.*?),')
        ori_num = self.parse_html(regex1,response)[0] 
        num = min(int(ori_num), self.needed_pics_num)
        print(f'{ori_num} {self.name_} pictures founded. start downloading {num} pictures...', file = sys.stderr) 
    
        if int(num) % 30 == 0:
            pn = int(int(num) / 30)
        else:
            pn = int(int(num) // 30 + 2)
        
        cnt, loop = 0, tqdm(total = num)
        for i in range(pn): 
            try:
                resp = self.get_one_html(self.url, i * 30)
                regex2 = re.compile('"middleURL":"(.*?)"')
                urls = [x for x in self.parse_html(regex2,resp) if x.startswith('http')]
                for u in urls:
                    try:
                        content = self.get_two_html(u) 
                        img_name = '{}.jpg'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                        img_path = os.path.join(self.save_dir, img_name)
                        with open(img_path, 'wb') as f:
                            f.write(content)
                        cnt += 1
                        loop.update(1)
                        if cnt >= num:
                            break
                    except Exception as err:
                        pass
                else:
                    continue
                break
            except Exception as err:
                print(err, file = sys.stderr)
            time.sleep(1.0) 
        loop.close()
        print('saved {} pictures in dir {}'.format(cnt, self.save_dir), file = sys.stderr)


def download(keyword, needed_pics_num = 100, save_dir = None):
    spider = _BaiduPictures(keyword, needed_pics_num, save_dir)
    spider.run()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
