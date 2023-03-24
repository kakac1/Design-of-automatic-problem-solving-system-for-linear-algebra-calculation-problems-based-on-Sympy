# coding=utf8
# Product Name:PyCharm
# Project Name:project
# @File : algori .py
# author: 柯坤程
# Time：2023/2/15
import operator

import numpy as np
import random
import sympy
from sympy import *

def setReverNum(num=8, *args, **kwargs) -> str:
    """
    随机生成一串长度为num的逆序列（逆序列的每个数字都不一样）
    :param num: 逆序数列的最大值
    :param args:
    :param kwargs:
    :return: 逆序序列

    例如：
    # 测试获取逆序数

    >>> test=setReverNum() #(生成一串逆序列)
    >>> test=setReverNum(9) #(生成一串9位的逆序列)
    >>> test=setReverNum(num=9) #(生成一串9位的逆序列)

    """
    #    返回一个num位(默认8位)的不相同数
    numlist = [str(i) for i in range(1, num + 1)]
    random.shuffle(numlist)  # 随机打乱列表的顺序，不生成新的列表
    rannum = "".join(numlist)  # 将列表组合成str
    return rannum  # 返回一个str的随机数


def getReverNum(Num, *args, **kwargs) -> int:
    """
    传入一串逆序列，返回它的逆序列
    :param Num: 传入一个逆序数列
    :param args:
    :param kwargs:
    :return: 返回逆序数列的逆序数

    例：


    >>> print("57426813的逆序数为",getReverNum('57426813'))
    >>> test=setReverNum()
    >>> print(test)
    >>> print(getReverNum(test))
    >>> print(getReverNum('1537624'))
    """
    Numlist = np.array(list(map(int, Num)))  # 把列表里的str转换成int
    ReverNum = [np.sum(Numlist[:numKey] > numValue) for numKey, numValue in
                enumerate(Numlist)]  # 利用enumerate通过循环迭代自动生成的索引变量。
    return int(np.sum(ReverNum))  # 返回逆序数


def setDetMat(long=4, minVal=-10, maxVal=100, minNum=0, maxNum=10, inver=False, *args, **kwargs) -> Matrix:
    """
    随机产生一个用minNum-maxNum数字构成的long行wid列的行列式随机产生一个用minNum - maxNum数字构成的long行wid列的行列式
    :param long: 行列式的行数和列数
    :param minVal: 行列式最小的值
    :param maxVal: 行列式最大的值
    :param minNum: 行列式内部最小的数
    :param maxNum: 行列式内部最大的数
    :param Inver: 行列式是否可逆，当为False时，行列式可以可逆也可以不可逆，当为True时行列式可逆
    :param args:
    :param kwargs:
    :return: 返回long*long的行列式
    例：

    >>> test = setDetMat() #生成一个行列式
    >>> print(test)
    >>> print(setDetMat(5)) #生成一个5行5列的行列式
    """
    ranMat = Matrix(np.random.randint(minNum, maxNum, (long, long)))
    ans = ranMat.det()  # 先计算计算行列式的值，把行列式的值控制在minVal-maxVal之间
    if not inver:
        while ans < minVal or ans > maxVal:
            ranMat = Matrix(np.random.randint(minNum, maxNum, (long, long)))
            ans = ranMat.det()  # 计算行列式的值
    else:
        while ans < minVal or ans > maxVal or ans == 0:  # 若行列式为零则行列式不可逆，重新产生行列式
            ranMat = Matrix(np.random.randint(minNum, maxNum, (long, long)))
            ans = ranMat.det()  # 计算行列式的值
    return Matrix(ranMat)


def getDetMat(DeterEvaluation, *args, **kwargs) -> int:
    """
    计算行列式的值
    :param DeterEvaluation: 要计算的行列式
    :param args:
    :param kwargs:
    :return: 返回行列式的值

    >>> test=setDetMat()
    >>> test1=Matrix([[-3, -1, 0, 0], [-4, -4, 0, 0], [0, 0, 4, 7], [0, 0, 4, 8]])
    >>> print(getDetMat(test)) #计算行列式的值
    >>> print(getDetMat(test1))

    """
    return int(DeterEvaluation.det())


def setMat(long=4, wid=5, minNum=-5, maxNum=5, *args, **kwargs) -> Matrix:
    """
    随机产生一个用minNum-maxNum数字构成的long行wid列的矩阵。
    为了确保行阶梯矩阵里面不含分数，查看随机产生的矩阵的行阶梯里面有没有分数，
    如果有，将分数取整，在对不含有分数的行阶梯矩阵进行任意次数的行变换得到最后的矩阵
    :param long: 矩阵的行数
    :param wid:  矩阵的列数
    :param minNum: 最简矩阵包含的最小数
    :param maxNum: 最简矩阵里包含的最大数
    :param args:
    :param kwargs:
    :return: 返回一个矩阵

    >>> setMat(2,3) #随机生成一个4行5列的行列式
    >>> setMat(3,2)  #随机生成一个行列式

    """
    ranMat = Matrix(np.random.randint(minNum, maxNum, (long, wid)))  # 随机产生一个用minNum-maxNum数字构成的long行wid列的矩阵
    stepMat = ranMat.rref()[0]  # stepMat是最简矩阵
    for idx, x in np.ndenumerate(stepMat):  # 确保行阶梯的矩阵
        stepMat[idx[0] * stepMat.shape[1] + idx[1]] = int(stepMat[idx[0] * stepMat.shape[1] + idx[1]])  # 对行阶梯矩阵取整
        if x < minNum or x > maxNum:  # 确保矩阵中的数范围是minNum-maxNum之间的数
            stepMat[idx[0] * stepMat.shape[1] + idx[1]] = sympy.Integer(random.randint(minNum, maxNum))  # 将矩阵中的行列式取整
    for i in range(stepMat.shape[0] - 1):  # 将得到的行最简的矩阵进行任意的行变换
        a = list(stepMat.row(0))
        n = random.randint(1, 3)
        b = list(map(lambda x: x * n, stepMat.row(stepMat.shape[0] - 1)))
        c = Matrix([np.sum([a, b], axis=0)])
        stepMat.row_del(0)
        stepMat = stepMat.row_insert(stepMat.shape[0] - 1, c)
    for i in range(stepMat.shape[0] - 1):  # 将得到的行最简的矩阵进行任意的行变换
        a = list(stepMat.row(stepMat.shape[0] - 1))
        n = random.randint(1, 3)
        b = list(map(lambda x: x * n, stepMat.row(0)))
        c = Matrix([np.sum([a, b], axis=0)])
        stepMat.row_del(stepMat.shape[0] - 1)
        stepMat = stepMat.row_insert(0, c)
    return stepMat


def getMiniMat(mat, *args, **kwargs) -> Matrix:
    """
    求行最简矩阵
    :param mat:  要求最简型的矩阵
    :param args:
    :param kwargs:
    :return: 返回矩阵的最简型

    >>> test=setMat()
    >>> getMiniMat(test)
    >>> test1=Matrix([[12, 1, 3, 40, -8], [4, 0, 1, 13, -1], [2, 0, 0, 5, 3], [1, 0, 0, 2, 2]])
    >>> getMiniMat(test1)

    """
    miniMat = mat.rref()[0]
    return miniMat


def getMatInver(mat, *args, **kwargs) -> Matrix:
    """
      要求矩阵的逆
    :param mat:要求矩阵的逆的矩阵
    :param args:
    :param kwargs:
    :return: 逆矩阵

    >>> test=Matrix([[1, 0, 0], [0, 1, 0], [2, 2, 1]])
    >>> getMatInver(test)

    """
    matInv = mat.inv()
    return matInv


def getAddMat(mat1, mat2, *args, **kwargs) -> Matrix:
    """
    计算矩阵的加法
    :param mat1: 要相加的第一个矩阵
    :param mat2: 要相加的第二个矩阵
    :param args:
    :param kwargs:
    :return: 相加后的矩阵

    >>> test1=setMat()
    >>> test2=setMat()
    >>> test3=Matrix([[2, 4, 4, 7, 7], [3, 5, 9, 8, 4], [4, 5, 5, 8, 8], [0, 8, 8, 0, 0], [3, 5, 6, 9, 8]])
    >>> test4=Matrix([[5, 7, 2, 1, 2], [9, 5, 9, 9, 4], [8, 7, 9, 6, 4], [9, 7, 8, 1, 2], [1, 1, 1, 2, 1]])
    >>> getAddMat(test1,test2)
    >>> getAddMat(test3,test4)

    """
    return mat1 + mat2


def getRedMat(mat1, mat2, *args, **kwargs) -> Matrix:
    """
    计算两个相减的数
     :param mat1: 要相减的第一个矩阵
    :param mat2: 要相减的第二个矩阵
    :param args:
    :param kwargs:
    :return: 相减后的矩阵

    >>> test1=setMat()
    >>> test2=setMat()
    >>> test3=Matrix([[8, 4, 7, 4, 1], [3, 9, 1, 6, 3], [3, 5, 6, 4, 3], [1, 4, 3, 3, 4], [9, 3, 1, 3, 1]])
    >>> test4=Matrix([[3, 5, 9, 8, 8], [3, 2, 9, 0, 7], [7, 5, 5, 5, 5], [6, 5, 8, 4, 9], [6, 3, 7, 0, 6]])
    >>> getRedMat(test1,test2)
    >>> getRedMat(test3,test4)

    """
    return mat1 - mat2


def getMulMat(mat1, mat2, *args, **kwargs) -> Matrix:
    """
    计算两个矩阵相乘
    :param mat1: 矩阵1
    :param mat2: 矩阵2
    :param args:
    :param kwargs:
    :return: 相乘后的矩阵

    >>> test1=setMat(3,4)  #随机生成一个三行四列的矩阵
    >>> test2=setMat(4,3)  #随机生成一个四行三列的矩阵
    >>> test3=Matrix([[3, 1, 12, 0], [1, 0, 3, 0], [1, 0, 2, 0]])
    >>> test4=Matrix([[6, 1, 2], [3, 0, 1], [3, 0, 0], [1, 0, 0]])
    >>> getMulMat(test1,test2)
    >>> getMulMat(test3,test4)

    """
    return mat1 * mat2


def setChuMat(long=4, minVal=0, maxVal=10, minNum=-5, maxNum=10, inver=False, *args, **kwargs) -> Matrix:
    """
    生成分块矩阵
    :param long: 分块矩阵的长度
    :param minNum: 分块矩阵包含的最小数
    :param maxNum: 分块矩阵包含的最大数
    :param inver: 是否可逆
    :param args:
    :param kwargs:
    :return: 返回分块矩阵

    >>> setChuMat()

    """
    x = long % 2  # 判断矩阵的行数是否为奇数
    matlong = int(long / 2)
    y = long - matlong
    if x == 0:  # 直接生成两个相同维度分块
        chuMat = diag(setDetMat(matlong, minVal, maxVal, minNum, maxNum, inver),
                      setDetMat(matlong, minVal, maxVal, minNum, maxNum, inver))
    else:  # 若矩阵的维度为基数，则用两个不同维度的可逆矩阵生成
        chuMat = diag(setDetMat(matlong, minVal, maxVal, minNum, maxNum, inver),
                      setDetMat(y, minVal, maxVal, minNum, maxNum, inver))
    return chuMat


def getMatT(mat, *args, **kwargs) -> Matrix:
    """
    计算矩阵的转置
    :param mat:要求转置的矩阵
    :param args:
    :param kwargs:
    :return: 矩阵的转置

    >>> test=Matrix([[4, 2, 0,  0],[9, 5, 0,  0],[0, 0, 3, -1],[0, 0, 2,  0]])
    >>> test1=Matrix([[-2, 6, 0, 0], [-3, 7, 0, 0], [0, 0, 8, 8], [0, 0, 7, 8]])
    >>> getMatT(test)
    >>> getMatT(test1)

    """
    return mat.T


def setSolveEquations(n=3, s=2, minVal=-2, maxVal=2, minNum=-4, maxNum=4, inver=True, returnlist=False, vector=False,
                      *args, **kwargs) -> Tuple:
    """
    生成矩阵方程AX=B
    :param n: 矩阵A的维度，A为可逆矩阵
    :param s: n*s的X矩阵
    :param minVal: A的最大值
    :param maxVal: A的最小值
    :param minNum: A和 B包含的数的最小值
    :param maxNum: A和 B包含的最大值
    :param inver: A矩阵是否可逆
    :param vector: 是否是求向量组
    :param args:
    :param kwargs:
    :return: A、X、B矩阵

    >>> setSolveEquations(3,4) #生成3*3的A矩阵，3*4的X矩阵，3*4的B矩阵

    """
    A = setDetMat(n, minVal, maxVal, minNum, maxNum, inver)  # A为n*n的矩阵
    X = setMat(n, s, minNum, maxNum)  # X为n*s的矩阵
    B = A * X
    if returnlist:
        A = A.tolist()
        X = X.tolist()
        B = B.tolist()
        return [A, X, B]
    return (A, X, B)


def getSolveEquations(mat1, mat2, *args, **kwargs) -> Matrix:
    """
    输入两个矩阵求解方程AX=B
    :param mat1:A为可逆的矩阵
    :param mat2:B为矩阵
    :param args:
    :param kwargs:
    :return:X矩阵

    >>> test=setSolveEquations(3,4) #生成3*3的A矩阵，3*4的X矩阵，3*4的B矩阵
    >>> getSolveEquations(test[0],test[2])
    >>> test1=Matrix([[-4,  0, 1],[ 1, -1, 3],[ 2,  0, 0]])
    >>> test2=Matrix([[-3, -4, -21, -5],[ 3,  1,  11, -1],[ 2,  2,  12,  2]])
    >>> getSolveEquations(test1,test2)  #求解方程组AX=B的X矩阵

    """
    return mat1 ** (-1) * mat2  # 对方程进行求解


def setVector(n=3, m=3, minNum=-4, maxNum=4, rel=True, ram=False, returnlist=False, *args, **kwargs) -> list:
    """
    生成n个m维的向量组
    :param n: 向量的个数
    :param m: 向量的维数
    :param relevant: 向量组是否线性相关
    :param args:
    :param kwargs:
    :return: 向量组（元组）

    >>> setVector(4,4) #生成4个4维的向量
    >>> setVector(3,4)#生成3个4维的向量
    >>> setVector(4,4,rel=True)#生成3个4维的线性相关的向量
    >>> setVector(4,4,ram=True)#生成3个4维的线性相关性随机的向量

    """
    Vectors = []
    if ram == True:
        x = (random.randint(1, 3)) % 2  # 产生0到2的随机数
        if x == 1:
            rel = False
    for i in range(n):  # 依次生成一个向量，依次添加进向量组里
        if returnlist:
            s = setMat(m, 1, minNum, maxNum).tolist()
        else:
            s = setMat(m, 1, minNum, maxNum)
        Vectors.append(s)
    if rel == True:
        if not getRelevant(Vectors):
            Vectors = setVector(n, m, minNum, maxNum, rel, ram, returnlist)
    else:
        if getRelevant(Vectors):
            Vectors = setVector(n, m, minNum, maxNum, rel, ram, returnlist)

    return Vectors


def getRelevant(vector, *args, **kwargs) -> bool:
    """
    判断向量组里的元素是否相关
    :param Vector: 向量组
    :param args:
    :param kwargs:
    :return: 线性相关返回True,线性无关返回False

    >>> test=setVector(3,3,rel=True)#产生3个3维的向量，相关性随机
    >>> test
    >>> getRelevant(test[0])
    >>> test1=setVector(3,3,ram=True)#产生3个3维的向量，相关性随机
    >>> test1
    >>> getRelevant(test1[0])
    >>> test1=setVector(3,3,ram=True)#产生3个3维的向量，相关性随机
    >>> test1
    >>> test3=[Matrix([[3],[1],[1],[1]]), Matrix([[3],[3],[1],[1]]), Matrix([[3],[3],[3],[1]]), Matrix([[6],[3],[1],[1]])]
    >>> getRelevant(test3)

    """
    vec1 = []
    for i in range(len(vector)):  # 将向量组转换成一个矩阵
        vec2 = np.array(vector)[i].reshape(1, -1)
        vec1.append(list(vec2[0]))
    vec1 = Matrix(vec1).T
    r = getmatRank(vec1)  # 计算矩阵的秩
    n = np.array(vec1).shape[0]  # 查看向量的个数
    if r < n:  # 线性相关
        return True
    else:  # 线性无关
        return False


def getmatRank(mat, *args, **kwargs) -> int:  # 求矩阵的秩
    """

    :param mat: 要求秩的矩阵
    :param args:
    :param kwargs:
    :return: 矩阵的秩

    >>> test=setMat()
    >>> getmatRank(test)
    >>> test1=setMat(3,4)
    >>> getmatRank(test1)
    >>> test2=Matrix([[5, 4, 9, 9], [4, 4, 9, 5], [9, 9, 2, 8], [9, 5, 8, 4]])
    >>> getmatRank(test2)

    """
    return int(mat.rank())


def getMatDot(mat1, mat2, *args, **kwargs):
    """
    求两向量的内积
    :param mat1: 向量组1
    :param mat2: 向量组2
    :param args:
    :param kwargs:
    :return: 返回内积

    >>> mat1=setMat(1, 3,5,20)
    >>> mat2=setMat(1, 3,5,20)
    >>> getMatDot(mat1,mat2)
    >>> mat3=Matrix([[5, 5, 17]])
    >>> mat4=Matrix([[5, 8, 6]])
    >>> getMatDot(mat3,mat4)

    """
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    mat2 = np.transpose(mat2)
    return int(mat1.dot(mat2)[0])


def getStepMat(mat, *args, **kwargs) -> Matrix:  # 求矩阵的阶梯型
    """
    注意，这个算法只能求出矩阵的一个阶梯型，矩阵的阶梯型并不是唯一的
    :param mat: 要求阶梯型的矩阵
    :param args:
    :param kwargs:
    :return: 返回阶梯型矩阵

    >>> mat1=Matrix([[18, 1, 3, 71, -70], [6, 0, 1, 23, -23], [3, 0, 0, 10, -10], [1, 0, 0, 3, -3]])
    >>> getStepMat(mat1)

    """
    return mat.rref()[0]


def setSymMat(n=3, minVal=-10, maxVal=100, minNum=0, maxNum=10, inver=True) -> Matrix:
    """
     生成对称矩阵(二次型矩阵)
    :param n:二次型矩阵的维度
    :return:

    >>> setSymMat(4)

    """
    mat = np.array(setDetMat(n, minVal, maxVal, minNum, maxNum))  # 创建一个方阵
    mat = np.triu(mat)  # 保留其上三角部分
    mat += mat.T - np.diag(mat.diagonal())  # 将上三角”拷贝”到下三角部分
    mat = Matrix(mat)
    if inver:
        if mat.det() == 0:
            mat = setSymMat(n, minVal, maxVal, minNum, maxNum, inver)
    return mat


def setlinear_combination(n=4, minVal=0, maxVal=10, minNum=1, maxNum=6, returnlist=False) -> Tuple:
    """
    :param n:求一个向量a被n个向量表示的线性组合
    :return:（A,X,B）向量B能被向量组A表示，向量系数为X

    >>> setlinear_combination(4)

    """
    mat1 = setMat(1, n, minNum, maxNum)  # 生成一个一行四列的矩阵
    mat2 = setDetMat(n, minVal, maxVal, minNum, maxNum)
    mat = []
    for i in range(sqrt(len(mat2))):
        sum = np.array(mat2.row(i)).sum()
        mat.append(sum * mat1[i])
    if returnlist:
        mat1 = mat1.tolist()
        mat2 = mat2.tolist()
        return [mat2, mat1, mat]
    else:
        mat = Matrix(mat)
    return (mat2, mat1, mat)


def setDiagMat(n=3, minNum=-2, maxNun=6) -> Matrix:
    """
    生成对角矩阵
    :param n: 对角矩阵的维度
    :param minNum: 对角矩阵中最小的数
    :param maxNun: 对角矩阵中最大的数
    :return:随机的对角矩阵
    """
    diaList = np.array(setMat(1, n))  # 先生成对角的元素列表
    mat = np.diag(diaList[0])
    return Matrix(mat)


def setQuadratic_Matrix(n=3, inverMinNum=-1, inverMaxNum=3, minNum1=1, maxNum1=3, returnlist=False) -> Tuple:
    """
    根据对角矩阵生成对称矩阵
    :param n: 对称矩阵的维度
    :param inverMinNum:对角矩阵中最小的数
    :param inverMaxNum: 对角矩阵中最大的数
    :param minNum1: 正定矩阵中最小的数
    :param maxNum1: 正定矩阵中最大的数
    :return:返回对称矩阵,可逆矩阵，对角矩阵（可逆矩阵乘以对称矩阵乘以可逆矩阵等于对角矩阵）

    >>> test=setQuadratic_Matrix()
    >>> test[1]*test[0]*test[1]
    >>> test[2]

    """
    diaMat = setDiagMat(n, inverMinNum, inverMaxNum)  # 生成一个对角矩阵
    mat = setSymMat(n, minNum=minNum1, maxNum=maxNum1, inver=True)  # 生成一个正交矩阵(可逆矩阵)
    mat2 = getMatInver(mat).T  # 生成一个正交矩阵
    quaMat = mat2 * diaMat * getMatInver(mat)  # 对称矩阵
    if returnlist:
        diaMat = diaMat.tolist()
        mat = mat.tolist()
        quaMat = quaMat.tolist()
        return [quaMat, mat, diaMat]
    return (quaMat, mat, diaMat)  # 返回对称矩阵,可逆矩阵，对角矩阵（P可逆矩阵乘以A对称矩阵乘以P可逆矩阵等于对角矩阵）


def getOrMat(mat, orthonormal=False, *args, **kwargs) -> list:
    """
    返回正交后的矩阵(斯密斯正交化)
    :param mat:要正交化的向量组
    :param orthonormal:是否执行单位化
    :param args:
    :param kwargs:
    :return: 正交化后的向量组

    >>> test=[Matrix([[1],[1],[1]]), Matrix([[3],[3],[1]]), Matrix([[6],[2],[1]])]
    >>> getOrMat(test)

    """
    return GramSchmidt(mat, orthonormal)


def setHomEquation(long=4, wid=5, minNum=-5, maxNum=5, returnlist=False):
    """
    生成一组有无穷多解的齐次方程组
    :param long: 矩阵的行数
    :param wid:  矩阵的列数
    :param minNum: 最简矩阵包含的最小数
    :param maxNum: 最简矩阵里包含的最大数
    :param returnlist: 是否输出列表形式
    :return:

    >>> setHomEquation(3,4)

    """
    mat = setMat(long, wid, minNum, maxNum)  # 生成一个long*wid的矩阵
    mat2 = Matrix(np.zeros((1, long))).T  # 生成1*wid的零矩阵
    if (mat.rank() < mat.shape[1]):  # 判断齐次方程是否有无穷多解，如果没有则重新生成
        if returnlist:
            mat = mat.tolist()
            mat2 = mat2.tolist()
            return [mat, mat2]
        else:
            return (mat, mat2)
    else:
        setHomEquation(long, wid, minNum, maxNum)


def setNotHomEquation(long=4, wid=6, minNum=-2, maxNum=2, returnlist=False)->Tuple:
    """
    生成随机的无穷多解的增广矩阵增广矩阵
    :param long:增广矩阵的行数
    :param wid: 增光矩阵的列数
    :param minNum: 增广矩阵中最小的数
    :param maxNum: 增广矩阵中最大的数
    :param returnlist: 是否返回列表
    :return: (增广矩阵的系数，增广矩阵的值)

    >>> setNotHomEquation()

    """
    mat = setMat(long, wid, minNum, maxNum)  # 生成一个long*wid的增广矩阵
    mat2 = Matrix(np.array(mat)[:, :mat.shape[1] - 1])  # 截取增广矩阵的系数矩阵
    mat3 = mat.col(mat.shape[1] - 1)  # 截取增广矩阵的结果 and np.sum(np.array(mat3)==0)<long
    if (mat.rank() < mat.shape[1] - 1 and mat.rank() == mat2.rank() and np.sum(
            np.array(mat3) == 0) < long):  # 判断非齐次方程是否有无穷多解，如果没有则重新生成
        if returnlist:
            mat = mat.tolist()
            mat2 = mat2.tolist()
            mat3 = mat3.tolist()
            return [mat2, mat3]
        else:
            return (mat2, mat3)
    else:
        (mat2, mat3) = setNotHomEquation(long, wid, minNum, maxNum, returnlist)
        return (mat2, mat3)


def listStrToMat(listStrmat):
    """
     将str类型的array转换成int类型的Matrix(只限三维以内)
    :param listStrmat:要转换的数组
    :return:int类型的Matrix

    >>> test=[['7','7','7'],['7','7','7'],['7','7','7']]
    >>> listStrToMat(test)
    >>> test2=[[['1','2'],['1/2','1']],[['1','1/989'],['1','2']]]
    >>> listStrToMat(test2)
    >>> test3=[[['1','2/7'],['1','1']],[['1','1'],['1','2']],[['1','2'],['1','1']]]
    >>> listStrToMat(test3)
    >>> test4=['1','2/9']
    >>> listStrToMat(test4)

    """
    metrics = len(np.array(listStrmat).shape)  # 判断要转换数组的维度

    listNum = []
    if metrics == 1:  # 如果数组是一维的话直接转化
        listNum = Matrix([x for x in listStrmat]).T

    elif metrics == 2:  # 如果数组是二维的话，遍历循环两次
        for i in listStrmat:
            number = [x for x in i]
            listNum.append(number)
        listNum = Matrix(listNum)
    elif metrics == 3:  # 如果数组是三维的话，遍历循环三次
        for i in listStrmat:
            liNum = []
            for j in i:
                number = [x for x in j]
                liNum.append(number)
            listNum.append(Matrix(liNum))
    return listNum


def judge_ReverNum(topic, answer) -> bool:
    """
     判断逆序数是否正确1
    :param topic:传入一个逆序列
    :param answer: 传入要判断的答案
    :param inputStr:输入的是字符串数组
    :return:判断结果

    >>> test1=setReverNum()#生成逆序列
    >>> test1
    >>> answer=getReverNum(test1)#得test1的逆序数
    >>> answer
    >>> judge_ReverNum(test1,15)#判断逆序数是否正确
    >>> judge_ReverNum('51736248',11)

    """
    if (answer == getReverNum(topic)):
        return True
    else:
        return False
    return


def judge_DetMat(topic, answer) -> bool:
    """

    判断answer是否为行列式topic的值2
    :param topic:输入一个行列式
    :param answer: 要判断的答案
    :return: 判断结果

    >>> test1=setDetMat(3)#生成随机2维矩阵
    >>> test1
    >>> getDetMat(test1)#获取行列式的值
    >>> judge_DetMat(test1,1)#判断传入的答案是否正确
    >>> test2=Matrix([[8, 6, 3],[7, 3, 7],[6, 3, 3]])
    >>> getDetMat(test2)
    >>> judge_DetMat(test2,39)

    """
    if (answer == getDetMat(topic)):
        return True
    else:
        return False
    return


def judge_MiniMat(topic, answer) -> bool:
    """
    判断answer是否为矩阵topic的最简型3
    :param topic:输入一个矩阵
    :param answer: 要判断的答案
    :return: 判断结果

    >>> test1=setMat(3,4)
    >>> test1
    >>> getMiniMat(test1)
    >>> answer1=Matrix([[1, 0, 0,  3],[0, 1, 0, -1],[0, 0, 1, -1]])
    >>> judge_MiniMat(test1,answer1)
    >>> test2=Matrix([[9, 1, 14, 12],[3, 0,  4,  5],[1, 0,  1,  2]])
    >>> answer2=Matrix([[1, 0, 0,  3],[0, 1, 0, -1],[0, 0, 1, -1]])
    >>> getMiniMat(test2)
    >>> judge_MiniMat(test2,answer2)
    """
    if (answer == getMiniMat(topic)):
        return True
    else:
        return False
    return


def judge_inverMat(topic, answer) -> bool:
    """
    判断输入的answer是否为topic的可逆矩阵4
    :param topic:输入一个矩阵
    :param answer: 要判断的答案
    :return: 判断结果

    >>> test1=setDetMat(3, 1, 10, 0, 3, inver=True)  # 生成一个可逆矩阵4
    >>> test1
    >>> answer1=Matrix([[ 1,  0, 0],[ 0,  1, 0],[-2, -1, 1]])
    >>> judge_inverMat(test1,answer1)
    >>> test2=Matrix([[1, 0, 0],[0, 1, 0],[2, 1, 1]])
    >>> getMatInver(test2)
    >>> answer2=Matrix([[ 1,  0, 0],[ 0,  1, 0],[-2, -1, 1]])
    >>> judge_inverMat(test2,answer2)
    """
    if (answer == getMatInver(topic)):
        return True
    else:
        return False


def judge_MatAdd(topic, answer) -> bool:
    """
     判断输入的answer是否为topic相加后的矩阵5
    :param topic:输入输入两个要相加的矩阵，例如(矩阵1,矩阵2)
    :param answer: 要判断的答案
    :return: 判断结果

    >>> test1=(setDetMat(3), setDetMat(3))
    >>> test1
    >>> getAddMat(test1[0],test1[1])
    >>> answer1=Matrix([[ 5,  2,  4],[12,  3,  5],[12, 10, 13]])
    >>> judge_MatAdd(test1,answer1)
    >>> test2=(Matrix([[5, 5, 6],[8, 4, 3],[2, 4, 2]]), Matrix([[8, 3, 1],[1, 4, 0],[6, 3, 4]]))
    >>> answer2=Matrix([[13, 8, 7],[ 9, 8, 3],[ 8, 7, 6]])
    >>> judge_MatAdd(test2,answer2)
    """
    if (answer == getAddMat(topic[0], topic[1])):
        return True
    else:
        return False


def judge_MatRed(topic, answer) -> bool:
    """
    判断answer是否为topic相减后的矩阵6
    :param topic: 两个相减的矩阵，例如(矩阵1，矩阵2)
    :param answer: 需要判断的答案
    :return:判断的结果

    >>> test1=(setDetMat(3), setDetMat(3))
    >>> test1
    >>> answer1=Matrix([[ 2,  0,  3],[ 3, -1, -2],[-3,  1, -6]])
    >>> judge_MatRed(test1,answer1)
    >>> test2=(Matrix([[4, 7, 4],[4, 5, 2],[1, 6, 3]]), Matrix([[2, 7, 1],[1, 6, 4],[4, 5, 9]]))
    >>> answer2=Matrix([[ 2,  0,  3],[ 3, -1, -2],[-3,  1, -6]])
    >>> judge_MatRed(test2,answer2)
    """
    if (answer == getRedMat(topic[0], topic[1])):
        return True
    else:
        return False


def judge_MatMul(topic, answer) -> bool:
    """
    判断answer是否为topic乘后的矩阵7
    :param topic: 两个相乘的矩阵，例如(矩阵1，矩阵2)
    :param answer: 需要判断的答案
    :return:判断的结果

    >>> test1=(setMat(3, 4), setMat(4, 3))
    >>> test1
    >>> getMulMat(test1[0],test1[1])
    >>> answer1=Matrix([[84, 3, 10],[24, 1,  3],[22, 1,  3]])
    >>> judge_MatMul(test1,answer1)
    >>> test2=(Matrix([[3, 1, 12, 0],[1, 0,  3, 0],[1, 0,  2, 0]]), Matrix([[18, 1, 3],[ 6, 0, 1],[ 2, 0, 0],[ 1, 0, 0]]))
    >>> answer2=Matrix([[84, 3, 10],[24, 1,  3],[22, 1,  3]])
    >>> judge_MatMul(test2,answer2)
    """
    if (answer == getMulMat(topic[0], topic[1])):
        return True
    else:
        return False


def judge_ChuMat_num(topic, answer) -> bool:
    """
    判断answer是否为分块矩阵topic的值8
    :param topic: 分块矩阵
    :param answer: 需要判断的答案
    :return:判断的结果

    >>> test1=setChuMat(inver=True)  # 生成一个分块矩阵8
    >>> test1
    >>> getDetMat(test1)
    >>> judge_ChuMat_num(test1,1)
    >>> test2=Matrix([[ 0, 2,  0,  0],[-3, 4,  0,  0],[ 0, 0, -3, -4],[ 0, 0,  6,  7]])
    >>> answer2=18
    >>> judge_ChuMat_num(test2,18)
    """
    if (answer == getDetMat(topic)):
        return True
    else:
        return False


def judge_ChuMat_T(topic, answer) -> bool:
    """

    判断answer是否为分块矩阵topic的转置9
    :param topic: 一个分块矩阵
    :param answer: 需要判断的答案
    :return:判断的结果

    >>> test1=setChuMat(inver=True)  # 生成一个分块矩阵8
    >>> test1
    >>> answer1=Matrix([[ 1,  6, 0,  0],[-2, -5, 0,  0],[ 0,  0, 7, -5],[ 0,  0, 2, -1]])
    >>> getMatT(test1)
    >>> judge_ChuMat_T(test1,answer1)
    >>> test2=Matrix([[1, -2,  0,  0],[6, -5,  0,  0],[0,  0,  7,  2],[0,  0, -5, -1]])
    >>> answer2=Matrix([[ 1,  6, 0,  0],[-2, -5, 0,  0],[ 0,  0, 7, -5],[ 0,  0, 2, -1]])
    >>> judge_ChuMat_T(test2,answer2)
    """
    if (answer == getMatT(topic)):
        return True
    else:
        return False


def judge_SolveEquations(topic, answer) -> bool:
    """
    判断answer是否为topic相减后的矩阵9(矩阵方程 AX=B求解)10
    :param topic: (矩阵A,矩阵X,矩阵B)
    :param answer: 需要判断的答案
    :return:判断的结果
    >>> test1=setSolveEquations()
    >>> test1
    >>> answer1= Matrix([[2, 1],[2, 0],[1, 0]])
    >>> judge_SolveEquations(test1,answer1)
    >>> test2=(Matrix([[ 2, -2,  2],[-2,  1, -3],[-1,  1, -2]]), Matrix([[4, 1],[2, 0],[1, 0]]), Matrix([[ 6,  2],[-9, -2],[-4, -1]]))
    >>> answer2= Matrix([[4, 1],[2, 0],[1, 0]])
    >>> judge_SolveEquations(test2,answer2)
    """
    if (answer == topic[1]):
        return True
    else:
        return False


def judge_matRel(topic, answer) -> bool:
    """
    判断向量组topic是否线性相关，answer是否判断正确11
    :param topic: 一个向量组
    :param answer: 需要判断的答案
    :return:判断的结果

    >>> test1=setVector(4, 4, ram=True)  # 生成4个4维的线性相关性随机的向量组11
    >>> test1
    >>> getRelevant(test1)
    >>> judge_matRel(test1,True)
    >>> test2=[Matrix([[6],[3],[3],[1]]), Matrix([[3],[1],[1],[1]]), Matrix([[6],[6],[3],[1]]), Matrix([[6],[6],[2],[1]])]
    >>> judge_matRel(test2,False)
    """
    if answer == '是':
        answer = True
    if answer == '否':
        answer = False
    if (answer == getRelevant(topic)):
        return True
    else:
        return False


def judge_RankMat(topic, answer) -> bool:  # 判断秩是否正确
    """
    判断answer是否为矩阵topic的秩12
    :param topic: 一个矩阵
    :param answer: 需要判断的答案
    :return:判断的结果

    >>> test1=setMat(3, 4)  # 生成一个4行5列的矩阵12
    >>> test1
    >>> getmatRank(test1)
    >>> judge_RankMat(test1,2)
    >>> test2=Matrix([[6, 1, 16, 60],[3, 0,  7, 27],[1, 0,  2,  8]])
    >>> getmatRank(test2)
    >>> judge_RankMat(test2,3)
    """
    if (answer == getmatRank(topic)):
        return True
    else:
        return False


def judge_MatDot(topic, answer) -> bool:  # 判断内积是否正确
    """
    判断answer是否为向量组topic的内积13
    :param topic: 两个一维向量，例(一维向量1,一维向量2)
    :param answer: 需要判断的答案
    :return:判断的结果
    >>> test1=(setMat(1, 3, 5, 20), setMat(1, 3, 5, 20))  # 生成两个向量13
    >>> test1
    >>> getMatDot(test1[0], test1[1])
    >>> judge_MatDot(test1,9)
    >>> test2=(Matrix([[5, 20, 12]]), Matrix([[18, 15, 17]]))
    >>> judge_MatDot(test2,594)

    """
    if (answer == getMatDot(topic[0], topic[1])):
        return True
    else:
        return False


def judge_StepMat(topic, answer) -> bool:  # 判断矩阵的阶梯型是否正确，矩阵的阶梯型并不是唯一的，所以通过阶梯型的最简型来判断，矩阵的最简型是唯一的
    """
    判断answer是否为矩阵topic的阶梯型14
    :param topic:一个矩阵
    :param answer: 需要判断的答案
    :return:判断的结果
    >>> test1=setMat(3, 4)  # 生成一个矩阵14
    >>> test1
    >>> getStepMat(test1)
    >>> answer1=Matrix([[1, 0, 0, 1],[0, 1, 0, 0],[0, 0, 1, 0]])
    >>> judge_StepMat(test1,answer1)
    >>> test2= Matrix([[6, 1, 12, 6],[2, 0,  3, 2],[1, 0,  1, 1]])
    >>> answer2=Matrix([[1, 0, 0, 1],[0, 1, 0, 0],[0, 0, 1, 0]])
    >>> judge_StepMat(test2,answer2)
    """
    if (answer == getMiniMat(topic)):
        return True
    else:
        return False


def judge_SymMat(topic, answer) -> bool:
    """
    判断answer是否为二次型矩阵topic的秩15
    :param topic:一个二次型矩阵
    :param answer: 需要判断的答案
    :return:判断的结果
   >>> test1= setSymMat(4)  # 生成一个二次矩阵15
   >>> test1
   >>> getmatRank(test1)
   >>> judge_SymMat(test1,1)
   >>> test2= Matrix([[7, 7, 2, 7],[7, 4, 7, 0],[2, 7, 4, 7],[7, 0, 7, 1]])
   >>> judge_SymMat(test2,4)
    """
    if (answer == getmatRank(topic)):
        return True
    else:
        return False


def judge_linear_combination(topic, answer) -> bool:  # 判断表示的系数是否正确
    """
    判断answer是否为向量组topic[A]的线性组合16
    :param topic:（A,X,B）向量B能被向量组A表示，向量系数为X
    :param answer: 需要判断的答案
    :return:判断的结果

    >>> test1=setlinear_combination()  # 生成（A,X,B）向量B能被向量组A表示，向量系数为X---------16
    >>> test1
    >>> answer1=Matrix([[1, 1, 5, 1]])
    >>> judge_linear_combination(test1,test1)
    >>> test2=(Matrix([[4, 3, 3, 5],[3, 2, 2, 2],[5, 3, 5, 3],[5, 3, 5, 3]]), Matrix([[1, 1, 5, 1]]), Matrix([[15],[ 9],[80],[16]]))
    >>> answer2=Matrix([[1, 1, 5, 1]])
    >>> judge_linear_combination(test2,answer2)
    """
    if (answer == topic[1]):
        return True
    else:
        return False


def judge_diaMat(topic, answer) -> bool:  # 判断PAP是否为对角阵,对称矩阵对角化后的对角元素不按顺序的话是唯一的
    """
    判断可逆矩阵answer是否为topic的答案17
    :param topic:(对称矩阵，可逆矩阵，对角矩阵)
    :param answer: 需要判断的答案
    :return:判断的结果
    >>> test1=setQuadratic_Matrix()  # 已知对称矩阵A，求一可逆矩阵P，使得PAP为对角阵17
    >>> test1
    >>> answer1=Matrix([[2, 1, 1],[1, 1, 2],[1, 2, 2]])
    >>> judge_diaMat(test1,answer1)
    >>> test2= (Matrix([[0,  0,  0],[0,  1, -1],[0, -1,  1]]), Matrix([[2, 1, 2],[1, 1, 1],[2, 1, 1]]), Matrix([[1, 0, 0],[0, 0, 0],[0, 0, 0]]))
    >>> answer2=Matrix([[2, 1, 2],[1, 1, 1],[2, 1, 1]])
    >>> judge_diaMat(test2,answer2)

    """
    answerDia = answer * topic[0] * answer  # 先计算可逆矩阵与对称矩阵与可逆矩阵相乘的结果

    for i in range(len(np.array(answerDia))):  # 先判断answerDia是不是对角矩阵
        for j in range(len(np.array(answerDia)[i])):
            if i != j and np.array(answerDia)[i][j] != 0:
                return False
    diaList = []  # 取出答案的生成的对角矩阵的元素列表
    answerList = []  # 取出题目的答案的对角矩阵的元素
    for i in range(len(np.array(answerDia))):  # 取出答案的生成的对角矩阵的元素
        diaList.append(np.array(answerDia)[i][i])
    for i in range(len(np.array(topic[2]))):  # 取出题目的答案的对角矩阵的元素
        answerList.append(np.array(topic[2])[i][i])
    for i in answerDia:  # 如果题目答案的对角矩阵不在计算后的对角矩阵中，则错误
        if i not in diaList:
            return False
    return True  # 如果符合以上条件则正确


def judge_orVector(topic, answer) -> bool:  # 判断正交后的矩阵是否正确（的进行单位化）,必须按照顺序
    """
    判断向量组topic正交化后的向量组是否为answer18
    :param topic:向量组
    :param answer: 需要判断的答案
    :return:判断的结果
    >>> test1=setVector(3, 3, rel=False)
    >>> test1
    >>> getOrMat(test1)
    >>> test2=[Matrix([[4],[2],[1]]), Matrix([[2],[2],[1]]), Matrix([[6],[3],[1]])]
    >>> answer2=[Matrix([[4],[2],['1']]), Matrix([['-10/21'],[ '16/21'],[ '8/21']]), Matrix([[0],[ '1/5'],['-2/5']])]
    >>> judge_orVector(test2,answer2)
    """
    # list = []
    # for i in range(np.shape(np.array(answer))[1]):  # 把输入的矩阵化成向量组
    #     list.append(Matrix(np.array(answer).col(i)))
    # answer = list
    print(answer)
    print(getOrMat(topic))
    if (answer == getOrMat(topic)):
        return True
    else:
        return False


def judge_HomEquationUntie(topic, answer) -> bool:
    """

    判断answer是否为齐次方程组topic的解19
    :param topic:齐次方程组,例(系数矩阵，系数矩阵的值)
    :param answer: 需要判断的答案
    :return:判断的结果
    >>> test1=(Matrix([[-2], [1], [-1], [0], [1]]),Matrix([[0], [0], [0]]))
    >>> test1
    >>> answer1=[Matrix([-1,1,0,0,0]),Matrix([-1,0,-1,0,1])]
    >>> judge_HomEquationUntie(test1,answer1)
    """

    sum = answer[0]
    for i in range(len(answer) - 1):  # 将所有的基础解系加起来,验算解
        sum = getAddMat(sum, answer[i + 1])
    if (topic[0] * sum == topic[1]):  # 相乘判断是否为解
        return True
    else:
        return False


def judge_HomEquationUntie(topic, answer) -> bool:
    """
    判断answer是否为非齐次方程组topic的值20
    :param topic:非齐次方程组,例(系数矩阵，系数矩阵的值)
    :param answer: 需要判断的答案
    :return:判断的结果

    """

    sum = answer[0]
    for i in range(len(answer) - 1):  # 将所有的基础解系加起来,验算解
        sum = getAddMat(sum, answer[i + 1])
    if (topic[0] * sum == topic[1]):  # 相乘判断是否为解
        return True
    else:
        return False


if __name__ == '__main__':  # 主函数测试
    pass
