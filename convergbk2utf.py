# -*- coding:utf-8 -*-

import os,sys
import chardet

def convert( filename, in_enc = "GBK", out_enc="UTF8" ):
    try:
        print "convert " + filename,
        content = open(filename).read()
        result = chardet.detect(content)#通过chardet.detect获取当前文件的编码格式串，返回类型为字典类型
        coding = result.get('encoding')#获取encoding的值[编码格式]
        if coding != 'utf-8':#文件格式如果不是utf-8的时候，才进行转码
            # print coding + "to utf-8!",
            new_content = content.decode(in_enc,"ignore").encode(out_enc)
            open(filename, 'w').write(new_content)
            print " done"
        else:
            print coding
    except IOError,e:
    # except:
        print " error"


def explore(dir):
    for root, dirs, files in os.walk('.'):
        for file in files:
            path = os.path.join(root, file)
            convert(path)

def main():
    for (dirname, dirs, files) in os.walk('.'):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(dirname, file)
                print(path)
                convert(path)


if __name__ == "__main__":
    main()