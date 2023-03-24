# -*- coding: utf-8 -*-
from trueOrFalse import *
from algori import *

'''
功能：多级菜单
'''


def inputMat(returnvector=False):
    try:
        if not returnvector:
            n = int(input(' 输入二维数组的行数'))
            m = int(input(' 输入二维数组的列数'))
            line = [[0] * m] * n  # 初始化二维数组
            for i in range(n):
                line[i] = input('请输入第' + str(i + 1) + '行二维数组(同行数字用逗号分隔，不同行则用回车换行)').split(
                    ",")  # 输入二维数组，同行数字用逗号分隔，不同行则用回车换行
                line[i] = [j for j in line[i]]  # 将数组中的每一行转换成整型
        else:
            n = int(input(' 输入向量组的个数'))
            m = int(input(' 输入向量组的维度'))
            line = [[0] * m] * n  # 初始化二维数组
            for i in range(n):
                line[i] = input('请输入第' + str(i + 1) + '行向量(同行数字用逗号分隔，不同行则用回车换行)').split(
                    ",")  # 输入二维数组，同行数字用逗号分隔，不同行则用回车换行
                line[i] = [j for j in line[i]]  # 将数组中的每一行转换成整型
        ans = Matrix(line)
        if returnvector:
            ans = ans.T
            list = []
            for i in range(np.shape(np.array(ans))[1]):  # 把输入的矩阵化成向量组
                list.append(Matrix(ans.col(i)))

        ans = list
    except Exception:
        print('输入格式错误请重新输入')
    return ans


def showTopic(topic):
    """
    这是出题系统的三级菜单
    :param topic:
    :return:
    """
    try:
        while True:
            print('=' * 30)
            mc1 = int(input('请输入要作答的题目：(输入88返回一级菜单)'))
            if mc1 == 1:
                ans = int(input('请输入第一题答案：'))
                topic.full_ReverNum(ans)
            elif mc1 == 2:
                ans = int(input('请输入第二题答案：'))
                topic.full_DetMat(ans)
            elif mc1 == 3:
                print('请输入三题答案')
                ans = inputMat()  # ’输入一个最简矩阵‘
                topic.full_MiniMat(ans)
            elif mc1 == 4:
                print('请输入四题答案')
                ans = inputMat()  # ’输入一个可逆矩阵‘
                topic.full_inverMat(ans)
            elif mc1 == 5:
                print('请输入五题答案')
                ans = inputMat()  # '输入一个相加后的矩阵'
                topic.full_MatAdd(ans)
            elif mc1 == 6:
                print('请输入六题答案')
                ans = inputMat()  # 输入一个相减后的矩阵
                topic.full_MatRed(ans)
            elif mc1 == 7:
                print('请输入七题答案')
                ans = inputMat()  # 输入一个相乘后的矩阵
                topic.full_MatMul(ans)
            elif mc1 == 8:
                print('请输入八题答案')
                ans = int(input('请输入第八题答案：'))  # 输入分块矩阵的值
                topic.full_ChuMat_num(ans)
            elif mc1 == 9:
                print('请输入九题答案')
                ans = inputMat()  # 输入一个分块矩阵转置后的矩阵
                topic.full_ChuMat_T(ans)
            elif mc1 == 10:
                print('请输入十题答案')
                ans = inputMat()  # 输入一个X矩阵
                topic.full_SolveEquations(ans)
            elif mc1 == 11:
                print('请输入十一题答案')
                ans = input()  # 输入是相关
                topic.full_matRel(ans)
            elif mc1 == 12:
                print('请输入十二题答案')
                ans = int(input('请输入第十二题答案：'))
                topic.full_RankMat(ans)
            elif mc1 == 13:
                print('请输入十三题答案')
                ans = int(input('请输入第十三题答案：'))
                topic.full_MatDot(ans)
            elif mc1 == 14:
                print('请输入十四题答案')
                ans = inputMat()  # ’输入一个阶梯矩阵‘
                topic.full_StepMat(ans)
            elif mc1 == 15:
                ans = int(input('请输入第十五题答案：'))
                topic.full_SymMat(ans)
            elif mc1 == 16:
                print('请输入十六题答案')
                ans = inputMat()  # 输入表示的系数
                topic.full_linear_combination(ans)
            elif mc1 == 17:
                print(topic.get__topics()['第十七题'])
                print('请输入十七题答案')
                ans = inputMat()  # 输入一个可逆矩阵P，使得PAP为对角阵
                topic.full_diaMat(ans)
            elif mc1 == 18:
                print(topic.get__topics()['第十八题'])
                print('请输入十八题答案')
                ans = inputMat()  # 输入一个正交后的矩阵
                topic.full_orVector(ans)
            elif mc1 == 19:
                print(topic.get__topics()['第十九题'])
                print('请输入十九题答案')
                ans = inputMat(returnvector=True)  # 输入一个正交后的矩阵
                topic.full_HomEquationUntie(ans)
            elif mc1 == 20:
                print(topic.get__topics()['第二十题'])
                print('请输入二十题答案')
                ans = inputMat(returnvector=True)  # 输入一个正交后的矩阵
                topic.full_HomEquationUntie(ans)
            elif mc1 == 88:
                firstMenu(topic)
            print('=' * 30)
    except Exception:
        print('输入格式错误请重新输入')
        showTopic(topic)


def firstMenu(topic):
    """
    这是一个二级菜单
    :param topic:传入题目
    :return:
    ==============================
    1.所有线代题目
    2.题目作答
    3.退出
    ==============================
    """
    try:
        while True:
            print('=' * 30)
            print('1.所有线代题目')
            print('2.题目作答')
            print('3.退出')
            print('=' * 30)
            mc1 = int(input('请输入菜单号：'))
            if mc1 == 1:
                print(topic)
                firstMenu(topic)
            elif mc1 == 2:
                showTopic(topic)
                break
            elif mc1 == 3:
                break
    except Exception:
        print('输入格式错误请重新输入(请输入数字)')
        firstMenu(topic)


def menu():
    """
    这是一个一级菜单，也是整个出题系统的入口
    :return:
    ==============================
    基于Sympy的线性代数计算题自动出题系统设计
    auther:柯坤程
    Time：2023/2/15


    按任意键进入系统...
    按b结束系统...
    ==============================
    """
    topic = TrueOrFalse()  # 实例化生成对象
    while True:
        print('=' * 30)
        print('基于Sympy的线性代数计算题自动出题系统设计')
        print('auther:柯坤程')
        print('Time：2023/2/15\n\n\n')
        mc1 = input('按任意键进入系统...\n按b结束系统...')
        print('=' * 30)
        if mc1 == 'b':
            break
        else:
            firstMenu(topic)  # 调用第一层菜单
        return


menu()
if __name__ == '__main__':
    pass
