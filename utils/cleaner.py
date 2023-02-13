"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/3/3 14:24
@Email : handong_xu@163.com
"""

"""
本代码是对语料进行预处理
"""
import re
import unicodedata

class EditingMethod():
    """
    全半角的转换
    """
    def __init__(self):
        pass

    ##### 全角转半角 #####
    def strQ2B(self,seq):
        restring = ''
        for unit in seq:
            temp = ord(unit)
            if temp == 12288:
                temp = 32
            elif (temp >= 65281 and temp <= 65374):
                temp -= 65248

            restring += chr(temp)
        return restring

    ##### 半角转全角 #####
    def strB2Q(self,seq):
        restring = ''
        for unit in seq:
            temp = ord(unit)
            if temp == 32:
                temp = 12288
            elif (temp >= 32 and temp <= 126):
                temp += 65248
            restring += chr(temp)
        return restring

class CleanData(EditingMethod):
    """
    去掉无效（不规范）字符
    """
    def __init__(self):
        super(CleanData,self).__init__()


    def _is_whitespace(self,char):
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _is_control(self,char):
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat in ("Cc", "Cf"):
            return True
        return False

    def _clean_text(self,text):
        text = self.strQ2B(text)
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

class Cleaner(CleanData):
    """
    清理语料，并基于标点进行分词
    """
    def __init__(self):
        super(Cleaner,self).__init__()
        self._sp_sep = {
        ',': self._check_comma_and_full_stop,
        '.': self._check_comma_and_full_stop}
        self.cust_pattern = r'[\(\)\.,:;\n"\t\-\+/&#%]'
        self.cust_pattern_dict = {
            '(': 0,
            ')': 1,
            '.': 2,
            ',': 3,
            ':': 4,
            ';': 5,
            '\n': 6,
            '\t': 7,
            '-': 8,
            '+': 9,
            '/': 10,
            '&': 11,
            '#': 12,
            '%': 13
        }

    def _check_comma_and_full_stop(self,s, seq):
        start, end = s.span(0)
        start_pre = start - 1
        if start_pre < 0 or end >= len(seq):
            return False
        if seq[start_pre].isdigit() and seq[end].isdigit():
            return True
        return False




    def pre_process(self,seq, pattern,pattern_dict, is_upper=False):
        def add_blank_space(s):
            if s.group() in self._sp_sep:
                _func = self._sp_sep[s.group()]
                if _func(s, seq):
                    return s.group()
            tmp = []
            start, end = s.span(0)
            start_pre = start - 1
            if start_pre >= 0:
                if seq[start_pre] != ' ':
                    tmp.append(' ')
            tmp.append(s.group())
            if end < len(seq):
                if seq[end] != ' ' and seq[end] not in pattern_dict:
                    tmp.append(' ')
            return ''.join(tmp)

        if is_upper:
            seq = seq.upper()
        seq = re.sub(pattern, add_blank_space, seq)
        return seq

    def remove_repeat_blank(self,seq):
        seq = re.sub(' +', ' ', seq.strip())
        return seq

    def common_cleaner(self,seq, is_upper=False, remove_blank=True):
        # print(seq)
        seq = self.strQ2B(seq)
        # print(seq)
        # s = input()
        seq = self._clean_text(seq)
        pattern = self.cust_pattern
        pattern_dict = self.cust_pattern_dict
        seq = self.pre_process(seq, pattern, pattern_dict, is_upper).strip()
        if remove_blank:
            seq = self.remove_repeat_blank(seq)
        return seq

