# coding=utf-8
import numpy as np
import math

'''     行列演算

    行列Aの逆行列
    A = np.linalg.inv(A)

    行列Aの転置行列
    A = A.T

    行列Aの行列式
    A = np.linalg.det(A)

    行列A,Bの和、差
    A + B
    A - B

    行列A、Bの乗算
    np.matmul(A,B)

'''


# ベイズ識別器クラス


class BayseClassifier:
    def __init__(self, sample, x):
        self.sample = sample
        self.x = x

    u = np.zeros(4)  # 平均ベクトルu
    s = np.zeros((4, 4))  # 共分散行列s

    # ベイズ推定を行うメソッド
    def BayesianInference(self):

        # 平均ベクトルuの推定をする関数
        def averagevector():
            u = np.zeros(4)
            for i in range(4):
                for j in range(self.sample):
                    u[i] += self.x[j][i]
                u[i] /= 50
            return u

        # 共分散行列sの推定をする関数
        def covariancematrix():
            s = np.zeros((4, 4))
            for j in range(self.sample):
                a = np.zeros((4, 4))
                b = np.zeros((4, 4))
                a[0] = self.x[j]
                a = a.T
                b[0] = self.u
                b = b.T
                s += np.matmul((a - b), (a - b).T)
            s /= self.sample - 1
            return s

        # 平均ベクトルの推定
        self.u = averagevector()
        # 共分散行列の推定
        self.s = covariancematrix()

    # 距離dを求めるメソッド
    def d(self, vector):

        # 距離dの計算
        det_s = np.linalg.det(self.s)  # 共分散行列の行列式
        d = np.matmul((vector - self.u).T, np.linalg.inv(self.s))
        d = np.matmul(d, (vector - self.u))
        d /= 2
        d += math.log(math.sqrt(det_s))

        return float(d)


# 任意のファイルからベクトルを入力する関数
def input_vector(filename):
    file = open(filename, 'r')
    x = np.zeros((4, 50), float)

    string = file.readline()
    listitem = string.split('\t')
    x[0] = list(map(float, listitem))

    string = file.readline()
    listitem = string.split('\t')
    x[1] = list(map(float, listitem))

    string = file.readline()
    listitem = string.split('\t')
    x[2] = list(map(float, listitem))

    string = file.readline()
    listitem = string.split('\t')
    x[3] = list(map(float, listitem))

    x = x.T
    file.close()
    return x


# 実行部

mixing_mat = np.zeros((3,3))    #混合行列

# ファイルからベクトルを入力
setosa = input_vector("iris setosa.txt")
versicolor = input_vector("iris versicolor.txt")
viriginica = input_vector("iris virginica.txt")

iris = [0 for i in range(3)]
# サンプルirissetosaをベイズ識別器に入力
iris[0] = BayseClassifier(len(setosa)-25, setosa)
# サンプルirisversicolorをベイズ識別器に入力
iris[1] = BayseClassifier(len(versicolor)-25, versicolor)
# サンプルirisviriginicaをベイズ識別器に入力
iris[2] = BayseClassifier(len(viriginica)-25, viriginica)

# 各クラスの共分散行列と平均ベクトルの推定

for i in range(3):
    iris[i].BayesianInference()

# テストサンプルに対する識別
e = 0.0  # 誤識別率

# 分割代入法でテスト
# 距離を求めて間違って識別した回数を混合行列にカウント
for i in range(25):
    testvector = setosa[i+25]
    ds = [iris[0].d(testvector), iris[1].d(testvector), iris[2].d(testvector)]

    if min(ds) == ds[0]:
        mixing_mat[0][0] += 1

    elif min(ds) == ds[1]:
        mixing_mat[1][0] += 1
    elif min(ds) == ds[2]:
        mixing_mat[2][0] += 1

for i in range(25):
    testvector = versicolor[i+25]
    ds = [iris[0].d(testvector), iris[1].d(testvector), iris[2].d(testvector)]

    if (min(ds) == ds[0]):
        mixing_mat[0][1] += 1

    elif min(ds) == ds[1]:
        mixing_mat[1][1] += 1

    elif min(ds) == ds[2]:

        mixing_mat[2][1] += 1

for i in range(25):
    testvector = viriginica[i+25]
    ds = [iris[0].d(testvector), iris[1].d(testvector), iris[2].d(testvector)]

    if min(ds) == ds[0]:
        mixing_mat[0][2] += 1

    elif min(ds) == ds[1]:
        mixing_mat[1][2] += 1

    elif min(ds) == ds[2]:
        mixing_mat[2][2] += 1

print(mixing_mat)
for i in range(3):
    e += mixing_mat[i][i]
e /= 75
e = 1 - e
print(e*100)