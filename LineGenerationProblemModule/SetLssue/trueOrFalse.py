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
    """
1. 求逆序列的逆序数
2. 行列式求值
3. 初等行变换化矩阵的最简型
4. 矩阵求逆
5. 矩阵相加
6.矩阵相减
7.矩阵相乘
8.分块矩阵的计算
9.分块矩阵的转置
10. 矩阵方程 AX=B  XA=B   求解
11. 判定向量组的线性相关、线性无关性
12. 求矩阵或向量组的秩
13. 求向量组内积
14. 初等行变换化矩阵的阶梯型
15. 写出二次型矩阵，求秩
16. 求一个向量a被向量组B表示的线性组合
17. 已知对称矩阵A，求一可逆矩阵P，使得PAP为对角阵
18 向量组正交化
19. 解齐次线性方程组
20. 解非齐次线性方程组

    """

    def __init__(self):
        self.__ReverNum = f'{setReverNum(7)}'  # 逆序数1
        self.__DetMat = setDetMat()  # 计算行列式2
        self.__Mat = setMat()  # 生成一个矩阵3
        self.__inverMat = setDetMat(3, 1, 10, 0, 3, inver=True)  # 生成一个可逆矩阵4
        self.__matAdd = (setDetMat(5), setDetMat(5))  # 生成两个矩阵5
        self.__matRed = (setDetMat(5), setDetMat(5))  # 生成两个矩阵6
        self.__matMul = (setMat(3, 4), setMat(4, 3))  # 生成两个矩阵--------------7
        self.__matChuMat1 = setChuMat(inver=True)  # 生成一个分块矩阵8
        self.__matChuMat2 = setChuMat(inver=True)  # 生成一个分块矩阵9
        self.__SolveEquations = setSolveEquations()  # 生成一个矩阵等式AX=B----------------10
        self.__matRel = setVector(4, 4, ram=True)  # 生成4个4维的线性相关性随机的向量组11
        self.__Matrank = setMat(4, 5)  # 生成一个4行5列的矩阵12
        self.__matDot = (setMat(1, 3, 5, 20), setMat(1, 3, 5, 20))  # 生成两个向量13
        self.__matStesp = setMat(4, 5)  # 生成一个矩阵14
        self.__SymMat = setSymMat(4)  # 生成一个二次矩阵15
        self.__linear_combination = setlinear_combination()  # 生成（A,X,B）向量B能被向量组A表示，向量系数为X---------16
        self.__diaMat = setQuadratic_Matrix()  # 已知对称矩阵A，求一可逆矩阵P，使得PAP为对角阵17
        self.__orVector = setVector(3, 3, rel=False)  # 向量组正交化18
        self.__HomEquation = setHomEquation(3, 4)  # 解齐次方程组19
        self.__NotHomEquation = setNotHomEquation(4, 6)  # 解非齐次方程组20
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
        """
        :return:所有题目的字典
        """
        return self.__topics

    def get__ans(self):
        """
        :return:所有填入的答案
        """
        return self.__ans

    def get__score(self):
        """
        :return:所以题目的正误
        """
        return self.__score

    def full_ReverNum(self, answer) -> None:
        """
       判断逆序数是否正确1
        :param answer: 要判断的答案（一个数字）
        :return:
        """
        self.__ans['第一题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getReverNum(self.__ReverNum)):
            self.__score['第一题'] = True
        else:
            self.__score['第一题'] = False
        return

    def full_DetMat(self, answer):  # 判断行列式的值是否正确2
        """
        判断answer是否为行列式的值
        :param answer:要判断的答案 （一个数字）
        :return:
        """
        self.__ans['第二题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getDetMat(self.__DetMat)):
            self.__score['第二题'] = True
        else:
            self.__score['第二题'] = False
        return

    def full_MiniMat(self, answer):  # 判断行列式是否是最简型3
        """
        判断answer是否为最简型
        :param answer:要判断的答案 （一个矩阵）
        :return:
        """
        self.__ans['第三题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getMiniMat(self.__topics['第三题'])):
            self.__score['第三题'] = True
        else:
            self.__score['第三题'] = False
        return

    def full_inverMat(self, answer):
        """
        判断可逆矩阵是否正确4
        :param answer: 输入的答案（一个矩阵）
        :return:
        """
        self.__ans['第四题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getMatInver(self.__topics['第四题'])):
            self.__score['第四题'] = True
        else:
            self.__score['第四题'] = False
        return

    def full_MatAdd(self, answer):
        """
        判断相加后的矩阵是否正确
        :param answer: 输入的答案（一个矩阵）
        :return:
        """
        self.__ans['第五题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getAddMat(self.__topics['第五题'][0], self.__topics['第五题'][1])):
            self.__score['第五题'] = True
        else:
            self.__score['第五题'] = False
        return

    def full_MatRed(self, answer):
        """
        判断相减后的矩阵是否正确
        :param answer: 输入的答案（一个矩阵）
        :return:
        """
        self.__ans['第六题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getRedMat(self.__topics['第六题'][0], self.__topics['第六题'][1])):
            self.__score['第六题'] = True
        else:
            self.__score['第六题'] = False
        return

    def full_MatMul(self, answer):
        """
         判断相乘后的矩阵是否正确
        :param answer: 输入的答案（一个矩阵）
        :return:
        """
        self.__ans['第七题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getMulMat(self.__topics['第七题'][0], self.__topics['第七题'][1])):
            self.__score['第七题'] = True
        else:
            self.__score['第七题'] = False
        return

    def full_ChuMat_num(self, answer):
        """
         判断分块矩阵计算后的值是否正确
        :param answer: 输入的答案（一个矩阵）
        :return:
        """
        self.__ans['第八题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getDetMat(self.__matChuMat1)):
            self.__score['第八题'] = True
        else:
            self.__score['第八题'] = False
        return

    def full_ChuMat_T(self, answer):
        """
        # 判断分块矩阵的转置是否正确
        :param answer: 输入的答案（一个矩阵）
        :return:
        """
        self.__ans['第九题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getMatT(self.__topics['第九题'])):
            self.__score['第九题'] = True
        else:
            self.__score['第九题'] = False
        return

    def full_SolveEquations(self, answer):
        """
        矩阵方程 AX=B求解
        :param answer:输入的答案（一个矩阵）
        :return:
        """
        self.__ans['第十题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == self.__topics['第十题'][1]):
            self.__score['第十题'] = True
        else:
            self.__score['第十题'] = False
        return

    def full_matRel(self, answer):
        """
        # 判断是否线性相关
        :param answer: True  或者 False
        :return:
        """
        if answer == '是':
            answer = True
        if answer == '否':
            answer = False
        self.__ans['第十一题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getRelevant(self.__topics['第十一题'])):
            self.__score['第十一题'] = True
        else:
            self.__score['第十一题'] = False
        return

    def full_RankMat(self, answer):
        """
        # 判断秩是否正确
        :param answer: 要判断的答案（一个数字）
        :return:
        """
        self.__ans['第十二题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getmatRank(self.__topics['第十二题'])):
            self.__score['第十二题'] = True
        else:
            self.__score['第十二题'] = False
        return

    def full_MatDot(self, answer):
        """
        # 判断内积是否正确
        :param answer: 要判断的答案（一个数字）
        :return:
        """
        self.__ans['第十三题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getMatDot(self.__topics['第十三题'][0], self.__topics['第十三题'][1])):
            self.__score['第十三题'] = True
        else:
            self.__score['第十三题'] = False
        return

    def full_StepMat(self, answer):  # 判断矩阵的阶梯型是否正确，矩阵的阶梯型并不是唯一的，所以通过阶梯型的最简型来判断，矩阵的最简型是唯一的
        """
        # 判断矩阵的阶梯型是否正确
        :param answer: 要判断的答案（一个矩阵）
        :return:
        """
        self.__ans['第十四题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getMiniMat(self.__topics['第十四题'])):
            self.__score['第十四题'] = True
        else:
            self.__score['第十四题'] = False
        return

    def full_SymMat(self, answer):
        """
        # 判断二次型的秩是否正确
        :param answer: 要判断的答案（一个数字）
        :return:
        """
        self.__ans['第十五题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getmatRank(self.__topics['第十五题'])):
            self.__score['第十五题'] = True
        else:
            self.__score['第十五题'] = False
        return

    def full_linear_combination(self, answer):
        """
        # 判断表示的系数是否正确
        :param answer: 要判断的答案（一个矩阵）
        :return:
        """

        self.__ans['第十六题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == self.__topics['第十六题'][1]):
            self.__score['第十六题'] = True
        else:
            self.__score['第十六题'] = False
        return

    def full_diaMat(self, answer):  # 判断PAP是否为对角阵,对称矩阵对角化后的对角元素不按顺序的话是唯一的
        """
        已知对称矩阵A，求一可逆矩阵P，使得PAP为对角阵
        :param answer:要判断的答案（一个矩阵）
        :return:
        """
        self.__ans['第十七题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        answerDia = answer * self.__topics['第十七题'][0] * answer  # 先计算可逆矩阵与对称矩阵与可逆矩阵相乘的结果
        for i in range(len(np.array(answerDia))):  # 先判断answerDia是不是对角矩阵
            for j in range(len(np.array(answerDia)[i])):
                if i != j and np.array(answerDia)[i][j] != 0:
                    self.__score['第十七题'] = False
                    return  # 如果不是对角矩阵直接返回
        diaList = []  # 取出答案的生成的对角矩阵的元素列表
        answerList = []  # 取出题目的答案的对角矩阵的元素
        for i in range(len(np.array(answerDia))):  # 取出答案的生成的对角矩阵的元素
            diaList.append(np.array(answerDia)[i][i])
        for i in range(len(np.array(self.__diaMat[2]))):  # 取出题目的答案的对角矩阵的元素
            answerList.append(np.array(self.__diaMat[2])[i][i])
        for i in answerDia:  # 如果题目答案的对角矩阵不在计算后的对角矩阵中，则错误
            if i not in diaList:
                self.__score['第十七题'] = False
        self.__score['第十七题'] = True  # 如果符合以上条件则正确
        return

    def full_orVector(self, answer):  # 判断正交后的矩阵是否正确（的进行单位化）,必须按照顺序
        """
        向量组正交化 是否正确
        :param answer:输入的答案（一组向量组）
        :return:
        """
        list = []
        for i in range(np.shape(np.array(answer))[1]):  # 把输入的矩阵化成向量组
            list.append(Matrix(answer.col(i)))
        answer = list
        self.__ans['第十八题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        if (answer == getOrMat(self.__topics['第十八题'])):
            self.__score['第十八题'] = True
        else:
            self.__score['第十八题'] = False
        return

    def full_HomEquationUntie(self, answer):
        """
        判断齐次方程组的解是否正确
        :param mat: 传入的基础解系
        :return:
        """
        self.__ans['第十九题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        sum = answer[0]
        for i in range(len(answer) - 1):  # 将所有的基础解系加起来,验算解
            sum = getAddMat(sum, answer[i + 1])
        if (self.__topics['第十九题'][0] * sum == self.__topics['第十九题'][1]):  # 相乘判断是否为解
            self.__score['第十九题'] = True
        else:
            self.__score['第十九题'] = False

    def full_HomEquationUntie(self, answer):
        """
        判断齐次方程组的解是否正确
        :param mat: 传入的基础解系
        :return:
        """
        self.__ans['第二十题'] = answer  # 把答案先赋值给__ans,然后把正确与否赋值给__score
        sum = answer[0]
        for i in range(len(answer) - 1):  # 将所有的基础解系加起来,验算解
            sum = getAddMat(sum, answer[i + 1])
        if (self.__topics['第二十题'][0] * sum == self.__topics['第二十题'][1]):  # 相乘判断是否为解
            self.__score['第二十题'] = True
        else:
            self.__score['第二十题'] = False

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
            f"13、求向量{self.__topics['第十三题'][0]}和向量{self.__topics['第十三题'][1]}的内积？答案：{self.__ans['第十三题']} 得分：{self.__score['第十三题']}\n" \
            f"14、求矩阵{self.__topics['第十四题']}的阶梯型？答案：{self.__ans['第十四题']} 得分：{self.__score['第十四题']}\n" \
            f"15、求二次型矩阵{self.__topics['第十五题']}的秩？答案：{self.__ans['第十五题']} 得分：{self.__score['第十五题']}\n" \
            f"16、求向量{self.__topics['第十六题'][2]}被向量组{self.__topics['第十六题'][0]}的线性组合？答案：{self.__ans['第十六题']} 得分：{self.__score['第十六题']}\n" \
            f"17、已知对称矩阵{self.__topics['第十七题'][0]}求一可逆矩阵P，使得PAP为对角阵？答案：{self.__ans['第十七题']} 得分：{self.__score['第十七题']}\n" \
            f"18、对向量组{self.__topics['第十八题']}进行正交化。答案：{self.__ans['第十八题']} 得分：{self.__score['第十八题']}\n" \
            f"19、对齐次方程组{self.__topics['第十九题'][0]}={self.__topics['第十九题'][1]}的解。答案：{self.__ans['第十九题']} 得分：{self.__score['第十九题']}\n" \
            f"20、对非齐次方程组{self.__topics['第二十题'][0]}={self.__topics['第二十题'][1]}的解。答案：{self.__ans['第二十题']} 得分：{self.__score['第二十题']}\n"
        return x


if __name__ == '__main__':
    t = TrueOrFalse()
    print(t)
