import numpy as np
import matplotlib.pyplot as plt
import control.matlab as con
import control as c
import sympy as sp
import time
from datetime import timedelta


def nyquis(W):
    plt.grid()
    c.nyquist_plot(W)
    plt.show()


def pzmap(W):
    con.pzmap(W)
    plt.title("Расположение нулей на комплексной плоскости")
    plt.grid(True)
    plt.show()


def mihalych(W1):
    P2 = list(W1.den[0])
    a = P2[0]  # коэфициенты
    plt.figure(1)
    w = sp.symbols("w", real=True)  # Годограф Михайлова
    z = 0
    for i in range(0, len(a), 1):
        z = z + a[i] * (1j * w) ** ((len(a) - 1) - i)
    z = sp.factor(z)
    zR = sp.re(z)
    zIm = sp.im(z)
    x = [zR.subs({w: q}) for q in np.arange(0, 100, 0.01)]
    y = [zIm.subs({w: q}) for q in np.arange(0, 100, 0.01)]
    plt.title("Годограф Михайлова")
    plt.axis([-100, 30, -150, 15])
    plt.plot(x, y)
    plt.grid()
    plt.show()


""" Функция, необходимая для Гурьвица """


def matrica_from_spisok(z, k):  # k - kolichestvo strok=stolbcov
    matr = ''
    z = z
    k = k
    h = 1
    for i in range(0, len(z), 1):
        if ((h * k) - i) == 0 and i != 0 and i != len(z) - 1:  # последний?
            matr = matr + ';' + str(z[i]) + ' '
            h = h + 1
        else:
            if i == len(z) - 1:  # проверка на самый посл элемент
                matr = matr + str(z[i])
            else:
                matr = matr + str(z[i]) + ' '
    Mat = np.matrix(matr)
    return Mat


""" Функция, необходимая для Гурьвица """


def matrica_spisok(z1, z2, k):
    """ к - количество столбцов и строк в матрице, ранг матрицы
    z1 - список четных коэф
    z2 - список нечетных коэф
    """
    z = []
    h1 = 0
    h2 = 0
    for g in range(0, k, 1):
        index_nach_nech = h1
        index_nach_ch = h2
        index_konec_nech = k - len(z1) - h1
        index_konec_ch = k - len(z2) - h2
        if g % 2 == 0:  # no четный
            for i in range(0, index_nach_nech, 1):
                z.append("0")
            for i in range(0, len(z1), 1):
                z.append(z1[i])
            for i in range(0, index_konec_nech, 1):
                z.append("0")
            h1 = h1 + 1
        else:
            for i in range(0, index_nach_ch, 1):
                z.append("0")
            for i in range(0, len(z2), 1):
                z.append(z2[i])
            for i in range(0, index_konec_ch, 1):
                z.append("0")
            h2 = h2 + 1
    return z


""" Критерий Гурвица. Все корни полинома Δ(s) имеют отрицательные вещественные части то-
гда и только тогда, когда все n главных миноров матрицы А (определителей Гурвица) поло-
жительны. """


def gurych(W1):
    T = True
    P2 = (W1.den[0])
    a = P2[0]  # коэфициенты
    z1 = []
    z2 = []
    for i in range(0, len(a), 1):
        if (i % 2 != 0):
            z1.append(str(a[i]))
        else:
            z2.append(str(a[i]))

    # print(z1) #нечетные коэф
    # print(z2) #четные коэф
    kol_str_stol = len(a) - 1  # количество строк и столбцов
    z = matrica_spisok(z1, z2, kol_str_stol)  # создаем матрицу n порядка
    Mat_gl = matrica_from_spisok(z, kol_str_stol)
    Del_mat = np.linalg.det(Mat_gl)

    """ необходимо рассмотреть определители главных миноров матрицы"""
    Del_minors = []
    for i in range(1, kol_str_stol, 1):
        Mat__minor_gl = Mat_gl[:i, :i]
        Del_mat_minor = np.linalg.det(Mat__minor_gl)
        Del_minors.append(Del_mat_minor)

    """Проверка определителей главных миноров"""
    for i in Del_minors:
        if i < 0:
            T = False
    print("Определитель главной матрицы = " + str(Del_mat))
    print("Определители миноров = " + " ".join(str(value) for value in Del_minors), sep=', ')
    if (T):
        print("Определитель главной матрицы = " + str(Del_mat))
        print("Определители миноров = " + " ".join(str(value) for value in Del_minors), sep=', ')
    return T


def pereh(W, t):
    plt.figure(1)  # Вывод графиков в отдельном окне
    y1, t = con.step(W, t)
    lines = [y1]
    lines[0] = plt.plot(t, y1, "r")
    plt.legend(lines[0], ['h(t) для 1'], loc='best', fontsize=10)
    plt.title('Переходная характеристика', fontsize=10)
    plt.ylabel('h')
    plt.xlabel('t, c')
    plt.grid()
    plt.show()


t = np.linspace(0, stop=10000, num=10000)

G1 = con.tf(1, 1)  # обратная связь
G2 = con.tf([2, 0], [3, 4])  # generator
G3 = con.tf(1, [1, 0])  # turbine
G4 = con.tf(1, [3, 1])  # device
G5 = G2 * G3 * G4
G_raz = G2 * G3 * G4 * G1  # W разомкнутой САУ
G_zam = G5/(1+G5)
print(G_zam)

nyquis(G_raz)
plt.show()
plt.figure()
# plt.plot()
plt.title('Частотная характеристика', fontsize=10, y=2.2)
mag1, phase1, omega1 = con.bode(G_raz)
plt.plot()
plt.show()
# pzmap(G_zam)
# mihalych(G_zam)
# gurych(G_zam)
pereh(G_zam, t)
G = True
k1 = 0.3
while G:
    G1 = con.tf(k1, 1)  # обратная связь
    G5 = G2 * G3 * G4
    G_zam = G5.feedback(other=G1, sign=-1)
    G_raz = G5*G1
    if (gurych(G_zam)):
        print("Система устойчива при k1 = " + str(k1))
        mag1, phase1, omega1 = con.bode(G_raz)
        plt.show()
        pereh(G_zam, t)
        # nyquis(G_raz)
        # pzmap(G_zam)
        # mihalych(G_zam)
        nyquis(G_raz)

        G = False
    else:
        k1 = k1 - 0.001
        print("k1 = ", k1)
