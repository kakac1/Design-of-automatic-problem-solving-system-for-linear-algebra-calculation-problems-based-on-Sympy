# coding=utf8
# Product Name:PyCharm
# Project Name:project
# @File : trueOrFalse .py
# author: 柯坤程
# Time：2023/3/2
import numpy as np
import operator

from algori import *


class TrueOrFalse:
    def __init__(self):
        self.__ReverNum = f'{setReverNum(7)}'  # 逆序数     1
        self.__DetMat = setDetMat().tolist()  # 计算行列式   2
        self.__Mat = setMat().tolist()  # 生成一个矩阵              3
        self.__inverMat = setDetMat(3, 1, 10, 0, 3, inver=True).tolist()  # 生成一个可逆矩阵    4
        self.__matAdd = [setDetMat(5).tolist(), setDetMat(5).tolist()]  # 生成两个矩阵          5
        self.__matRed = [setDetMat(5).tolist(), setDetMat(5).tolist()]  # 生成两个矩阵         6
        self.__matMul = [setMat(3, 4).tolist(), setMat(4, 3).tolist()]  # 生成两个矩阵--------------7
        self.__matChuMat1 = setChuMat(inver=True).tolist()  # 生成一个分块矩阵8
        self.__matChuMat2 = setChuMat(inver=True).tolist()  # 生成一个分块矩阵9
        self.__SolveEquations = setSolveEquations(returnlist=True)  # 生成一个矩阵等式AX=B----------------10
        self.__matRel = setVector(4, 4, ram=True, returnlist=True)  # 生成4个4维的线性相关性随机的向量组11
        self.__Matrank = setMat(4, 5).tolist()  # 生成一个4行5列的矩阵12
        self.__matDot = [setMat(1, 3, 5, 20).tolist(), setMat(1, 3, 5, 20).tolist()]  # 生成两个向量13
        self.__matStesp = setMat(4, 5).tolist()  # 生成一个矩阵14
        self.__SymMat = setSymMat(4).tolist()  # 生成一个二次矩阵15
        self.__linear_combination = setlinear_combination(returnlist=True)  # 生成（A,X,B）向量B能被向量组A表示，向量系数为X---------16
        self.__diaMat = setQuadratic_Matrix(returnlist=True)  # 已知对称矩阵A，求一可逆矩阵P，使得PAP为对角阵17
        self.__orVector = setVector(3, 3, rel=False, returnlist=True)  # 向量组正交化18
        self.__HomEquation = setHomEquation(3, 4, returnlist=True)  # 解齐次方程组19
        self.__NotHomEquation = setNotHomEquation(4, 6, returnlist=True)  # 解非齐次方程组20
        self.__topics = {'第一题': self.__ReverNum, '第二题': self.__DetMat, '第三题': self.__Mat,
                         '第四题': self.__inverMat, '第五题': self.__matAdd,
                         '第六题': self.__matRed, '第七题': self.__matMul, '第八题': self.__matChuMat1,
                         '第九题': self.__matChuMat2, '第十题': self.__SolveEquations
            , '第十一题': self.__matRel, '第十二题': self.__Matrank, '第十三题': self.__matDot,
                         '第十四题': self.__Matrank, '第十五题': self.__SymMat
            , '第十六题': self.__linear_combination, '第十七题': self.__diaMat, '第十八题': self.__orVector,
                         '第十九题': self.__HomEquation, '第二十题': self.__NotHomEquation}
        self.__ans = {'第一题': None, '第二题': None, '第三题': None, '第四题': None, '第五题': None,
                      '第六题': None, '第七题': None, '第八题': None, '第九题': None, '第十题': None
            , '第十一题': None, '第十二题': None, '第十三题': None, '第十四题': None, '第十五题': None
            , '第十六题': None, '第十七题': None, '第十八题': None, '第十九题': None, '第二十题': None}  # 填写的答案
        self.__score = {'第一题': None, '第二题': None, '第三题': None, '第四题': None, '第五题': None,
                        '第六题': None, '第七题': None, '第八题': None, '第九题': None, '第十题': None
            , '第十一题': None, '第十二题': None, '第十三题': None, '第十四题': None, '第十五题': None
            , '第十六题': None, '第十七题': None, '第十八题': None, '第十九题': None, '第二十题': None}  # 判断答案是否正确

    def get__topics(self):
        return self.__topics

    def get__ans(self):
        return self.__ans

    def get__score(self):
        return self.__score



    def __str__(self):
        x = f"1、求{self.__topics['第一题']}的逆序数？答案：{self.__ans['第一题']} 得分：{self.__score['第一题']}\n" \
            f"2、求行列式{self.__topics['第二题']}的值？答案：{self.__ans['第二题']} 得分：{self.__score['第二题']}\n" \
            f"3、求矩阵{self.__topics['第三题']}的最简型？答案：{self.__ans['第三题']} 得分：{self.__score['第三题']}\n" \
            f"4、求矩阵{self.__topics['第四题']}的可逆矩阵？答案：{self.__ans['第四题']} 得分：{self.__score['第四题']}\n" \
            f"5、求矩阵{self.__topics['第五题'][0]}和{self.__topics['第五题'][1]}相加后得矩阵？答案：{self.__ans['第五题']} 得分：{self.__score['第五题']}\n" \
            f"6、求矩阵{self.__topics['第六题'][0]}和{self.__topics['第六题'][1]}相减后得矩阵？答案：{self.__ans['第六题']} 得分：{self.__score['第六题']}\n" \
            f"7、求矩阵{self.__topics['第七题'][0]}和{self.__topics['第七题'][1]}相乘后得矩阵？答案：{self.__ans['第七题']} 得分：{self.__score['第七题']}\n" \
            f"8、求分块矩阵{self.__topics['第八题']}的值？答案：{self.__ans['第八题']} 得分：{self.__score['第八题']}\n" \
            f"9、求分块矩阵{self.__topics['第九题']}的转置？答案：{self.__ans['第九题']} 得分：{self.__score['第九题']}\n" \
            f"10、求矩阵方程AX=B，其中A={self.__topics['第十题'][0]},B={self.__topics['第十题'][2]}.答案：{self.__ans['第十题']} 得分：{self.__score['第十题']}\n" \
            f"11、判断向量组{self.__topics['第十一题']}是否线性相关？答案：{self.__ans['第十一题']} 得分：{self.__score['第十一题']}\n" \
            f"12、求{self.__topics['第十二题']}的秩？答案：{self.__ans['第十二题']} 得分：{self.__score['第十二题']}\n" \
            f"13、求向量{self.__topics['第十三题'][0]}和量{self.__topics['第十三题'][1]}的内积？答案：{self.__ans['第十三题']} 得分：{self.__score['第十三题']}\n" \
            f"14、求矩阵{self.__topics['第十四题']}的阶梯型？答案：{self.__ans['第十四题']} 得分：{self.__score['第十四题']}\n" \
            f"15、求矩阵{self.__topics['第十五题']}的阶梯型？答案：{self.__ans['第十五题']} 得分：{self.__score['第十五题']}\n" \
            f"16、求向量{self.__topics['第十六题'][2]}被向量组{self.__topics['第十六题'][0]}的线性组合？答案：{self.__ans['第十六题']} 得分：{self.__score['第十六题']}\n" \
            f"17、已知对称矩阵{self.__topics['第十七题'][0]}求一可逆矩阵P，使得PAP为对角阵？答案：{self.__ans['第十七题']} 得分：{self.__score['第十七题']}\n" \
            f"18、对向量组{self.__topics['第十八题']}进行正交化。答案：{self.__ans['第十八题']} 得分：{self.__score['第十八题']}\n" \
            f"19、对齐次方程组{self.__topics['第十九题'][0]}={self.__topics['第十九题'][1]}的解。答案：{self.__ans['第十九题']} 得分：{self.__score['第十九题']}\n" \
            f"20、对非齐次方程组{self.__topics['第二十题'][0]}={self.__topics['第二十题'][1]}的解。答案：{self.__ans['第二十题']} 得分：{self.__score['第二十题']}\n"
        return x


if __name__ == '__main__':
    kk = TrueOrFalse()
    print(kk)
    print('第1题', kk.get__topics()['第一题'])
    print('第2题', kk.get__topics()['第二题'])
    print('第3题', kk.get__topics()['第三题'])
    print('第4题', kk.get__topics()['第四题'])
    print('第5题', kk.get__topics()['第五题'])
    print('第6题', kk.get__topics()['第六题'])
    print('第7题', kk.get__topics()['第七题'])
    print('第8题', kk.get__topics()['第八题'])
    print('第9题', kk.get__topics()['第九题'])
    print('第10题', kk.get__topics()['第十题'])
    print('第11题', kk.get__topics()['第十一题'])
    print('第12题', kk.get__topics()['第十二题'])
    print('第13题', kk.get__topics()['第十三题'])
    print('第14题', kk.get__topics()['第十四题'])
    print('第15题', kk.get__topics()['第十五题'])
    print('第16题', kk.get__topics()['第十六题'])
    print('第17题', kk.get__topics()['第十七题'])
    print('第18题', kk.get__topics()['第十八题'])
    print('第19题', kk.get__topics()['第十九题'])
    print('第20题', kk.get__topics()['第二十题'])
