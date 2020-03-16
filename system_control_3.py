import numpy as np
import matplotlib.pyplot as plt
import control.matlab as con
import control as c
import sympy as sp
import time
from datetime import timedelta
from numba import jit
import math
import scipy.integrate as integrate
import time as tttt

# @jit
def nyquis(W):
    plt.grid()
    c.nyquist_plot(W)
    plt.show()


def pzmap(W, T):
    # f = con.pzmap(W).
    g = np.array(W.pole())  # возвращает список всех полюсов - корней знаменателя
    # plt.title("Расположение полюсов на комплексной плоскости")
    # plt.grid(True)
    # print("Корни = ", g)
    plt.close()
    D = True
    j = 0
    G = False
    kol = 0
    a = np.array(np.zeros((1000, 2)))
    # print(g)
    while D and j < len(g):
        r = g[j].real
        # print(r)
        if float(r) <= 0.000000:
            # print("Левый корень = " + str(r))
            D = True
            if float(r) == 0.0:
                G = True
            else:
                a[kol, 0] = np.abs(r)
                a[kol, 1] = np.abs(g[j].imag)
                kol += 1
        else:
            D = False
            # print("Первый правый  корень = " + str(r))
        j = j + 1
    # if D and G:
    #     print("Система на границе устойчивости, есть нулевые корни")
    # elif D:
    #     print("Система устойчива, все корни левые")
    # else:
    #     print("Cистема не устойчива, есть хотя бы один правый корень")
    a = a[:kol, :]
    print(a)
    # plt.title("Расположение полюсов на комплексной плоскости")
    # plt.grid(True)
    # plt.show()
    if T:
        return D
    else:
        return a


# @jit
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


#
# @jit
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


# @jit
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
    # if (T):
    #     print("Определитель главной матрицы = " + str(Del_mat))
    #     print("Определители миноров = " + " ".join(str(value) for value in Del_minors), sep=', ')
    # else:
    # print("Система неустойчива")
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


def PID(Kp,Ki,Kd):
    p = con.tf(Kp, 1) + con.tf([0, Ki], [1, 0]) + con.tf([Kd, 0], 1)
    return p


def ust(spisok_y1):
    state = float(spisok_y1[len(spisok_y1) - 1])
    if (state > 0.95 and state < 1.05):
        Valueofstate = 1
    else:
        Valueofstate = 0

    return Valueofstate


def transition_time_of_proсess_andGraph(y1, state, t1, Graph):
    c = 0
    u = 0
    T = True
    y = list(y1)
    y.reverse()
    t = list(t1)
    t.reverse()
    line1 = 1.05 * state
    line2 = 0.95 * state
    while (T and c < len(y)):
        if (float(y[c]) < line1) and (float(y[c]) > line2):
            c += 1
        else:
            T = False
    transition_time = t[c]
    if (not T) and Graph:
        # y.reverse()
        line11 = [line1 for i in range(len(y1))]
        line22 = [line2 for i in range(len(y1))]
        line3 = np.arange(0, y[c], 0.001)
        c = np.ones((1, len(line3))) * t[c]
        spisok_time = c[0]
        lines = [y, line11, line22, line3]
        lines[0], lines[1], lines[2], lines[3] = plt.plot(t, y, t, line11, t, line22, spisok_time, line3)
        plt.legend(lines, ['h(t) для 1'], loc='best', fontsize=10)
        plt.title('Переходная характеристика', fontsize=10)
        plt.ylabel('h')
        plt.xlabel('t, c')
        plt.axis([0, 100, 0, 1.5])
        plt.grid()
        plt.show()
    return transition_time


def search_order_numbers(grad):
    bool = True
    first = 10
    number_order = 0
    grag_int = int(grad)
    while bool:
        relation = grag_int / first
        if (np.abs(relation) > 1):
            number_order += 1
            first = first * 10
        else:
            bool = False

    return number_order + 1


def Pole(W):
    con.pzmap(W)
    plt.plot()
    plt.grid(True)
    plt.show()
    print('Оценка по распределению корней:')
    Pol = con.pole(W)
    # print(Pol)
    P = []
    """ Показатель колебательности характеризует склонность системы к
    колебаниям: чем выше М, тем менее качественна система при прочих
    равных условиях. Считается допустимым, если 1,1 < М < 1,5.    """

    degreeovershoot_M = []
    for i in Pol:
        k = complex(i)
        if k.real != 0:
            P.append(k.real)
            m = k.imag / k.real
            degreeovershoot_M.append(m)
    a_min = max(P)
    t_reg = abs(1 / a_min)
    overshooting = math.exp(math.pi / max(degreeovershoot_M))
    psi = 1 - math.exp((-2) * math.pi / max(degreeovershoot_M))
    print("a_min= ", a_min)
    print("Время пп: ", t_reg, " c")
    print("Степень колебательности: ", max(degreeovershoot_M))
    print("Перерегулирование: ", overshooting)
    print("Степень затухания: ", psi)


def Freq(W):
    print("Оценка по АЧХ и ФЧХ:")
    # t = np.linspace(0, stop=100, num=1000)
    mag, phase, omega = con.bode(W, dB=False)
    plt.plot()
    # plt.close()
    plt.show()
    mag_max = max(mag)
    M = (mag_max / mag[0])
    print("Показатель коллебательности М: " + str(M))
    t = True
    n = list(mag).index(mag_max)
    while n < len(mag) and t:
        if ((mag[n] > mag[0] - 0.01) and (mag[n] < mag[0] + 0.01)) or mag[n] < mag[0]:
            wc = omega[n]
            phase_res = phase[n]
            print("Время регулирования: " + str(2 * math.pi / wc))
            """ В хорошо демпфированных системах запас устойчивости по амплитуде колеблется 
            в пределах от 6 до 20 дБ, а запас по фазе от 30 до 60"""
            print("Запас по фазе: " + str(180 - abs(phase_res)))
            t = False
        n = n + 1

    t_magn = True
    n = 0
    while n < len(phase) and t_magn:
        if -179.7 > phase[n] > -180.7:
            index_phase_180 = list(phase).index(phase[n])
            print("Запас по амплитуде: " + str(round(mag[0] - mag[index_phase_180], 3)))
            t_magn = False
        n = n + 1
    """запас по амплитуде может быть равен бесконечности, если фазовая характеристика не 
    пересекает линию −180°"""
    if t_magn:
        print("Запас по амплитуде равен бесконечности,  фазовая характеристика не пересекает линию −180°")


def intergalnaya_Otsenka(W):
    a = 0
    b = 100
    n = 1000
    h = (b - a) / n
    t = np.linspace(a, b, num=n)
    y, x = con.step(W, t)  # х-время ПП
    func = 0
    x0 = x[0]
    y0 = y[0]
    # нахождение площади методом трапеций
    for i in range(0, len(x), 1):
        xi = x[i]
        y1 = y[i]
        func += abs(1 * (xi - x0) - 0.5 * (y1 + y0) * (xi - x0))
        x0 = xi
        y0 = y1
    # print(func)
    print("Интегральная оценка за ", b, " c = ", func)


def for_iter(coef_regul, main_condition, middle_condition, delta_overshoot, delta_processtime, name):
    alfa = []
    for i in range(0, 50):
        ran = np.random.random()
        if i < 2:
            alfa.append(ran * 0.00001)
        elif i < 4:
            alfa.append(ran * 0.001)
        elif i < 5:
            alfa.append(ran * 0.01)
        elif i < 10:
            alfa.append(ran * 1)
    # print(alfa)
    grad_overshoot = delta_overshoot + 0.0001
    grad_processtime = delta_processtime + 0.0001
    print("Градиент для максимума=", grad_overshoot, " для " + name)
    print("Градиент для времени=", grad_processtime, " для " + name)
    number_of_grad_overshoot = search_order_numbers(grad_overshoot)
    number_of_grad_processtime = search_order_numbers(grad_processtime)
    alfa_spisok = []
    # print("HI  " + name)
    if main_condition and (not middle_condition) and (name == "P" or name == "D"):
        if number_of_grad_overshoot > len(alfa):
            if (grad_overshoot>0):
                alfa_forovershot = alfa[len(alfa) - 1] * (grad_overshoot / (np.abs(grad_overshoot)))*(np.abs(grad_overshoot))
            else:
                alfa_forovershot = - alfa[len(alfa) - 1] * (grad_overshoot / (np.abs(grad_overshoot))) * (
                    np.abs(grad_overshoot))
        else:
            if (grad_overshoot>0):
                alfa_forovershot = alfa[number_of_grad_overshoot] * grad_overshoot / (np.abs(grad_overshoot))*(np.abs(grad_overshoot))
            else:
                alfa_forovershot = - alfa[number_of_grad_overshoot] * grad_overshoot / (np.abs(grad_overshoot)) * (
                    np.abs(grad_overshoot))
        the_best_alfa = alfa_forovershot
        # print("Best11111 = " + str(the_best_alfa) + "  " + name)

    elif not main_condition and (middle_condition) and (name == "P" or name == "D"):
        if number_of_grad_processtime > len(alfa):
            # if (grad_overshoot > 0):
            #     alfa_forprocesstime =  alfa[len(alfa) - 1] * (grad_processtime / (np.abs(grad_processtime)))*(np.abs(grad_processtime))
            # else:
            alfa_forprocesstime = alfa[len(alfa) - 1] * (grad_processtime / (np.abs(grad_processtime))) * (
                    np.abs(grad_processtime))
        else:
            # if grad_processtime > 0:
            #     alfa_forprocesstime = alfa[number_of_grad_processtime] * grad_processtime / (np.abs(grad_processtime))*(np.abs(grad_processtime))
            # else:
            alfa_forprocesstime = alfa[number_of_grad_processtime] * grad_processtime / (
                    np.abs(grad_processtime)) * (np.abs(grad_processtime))
        the_best_alfa = alfa_forprocesstime
        # print("number_of_grad_processtime " + str(number_of_grad_processtime) + "  " + name)
        # print("grad_processtime " + str(grad_processtime) + "  " + name)
        # print("Best22222 = " + str(the_best_alfa) + "  " + name)
    elif (not main_condition and (middle_condition)) or (name == "I"):
        the_best_alfa = 0.0001
    else:
        the_best_alfa = alfa[number_of_grad_processtime] * grad_processtime / (np.abs(grad_processtime))*(np.abs(grad_processtime))
        # print("grad_processtime " + str(grad_processtime) + "  " + name)
        # print("Best3333333 number_of_grad_processtime= " + str(number_of_grad_processtime) + "  " + name)
    # print("Best = " + str(the_best_alfa) + "  " + name)
    return the_best_alfa


t = np.linspace(0, stop=100, num=100)
treg = 14  # Время регулирования
overshoot = 1.22

G0 = con.tf(1, 1)  # обратная связь
G1 = con.tf(1, [2, 1])  # generator
G2 = con.tf(1, [5, 1])  # turbine
G3 = con.tf(21, [5, 1])  # device

J = True
p_for_iter = np.zeros((1, 2))
i_for_iter = np.zeros((1, 2))
d_for_iter = np.zeros((1, 2))
M_for_iter = 0
T_for_iter = 0
M_for = np.zeros((1, 2))
T_for = np.zeros((1, 2))
p = 0.06
i = 0.006
d = 0.6
# FlagP = True
# FlagI = False
# FlagD = False
iter1 = 1
start_t= tttt.time()

while J:
    overshoot_condition = False
    processtime_condition = False
    value_condition = False
    p_for_iter[0, 1] = p
    i_for_iter[0, 1] = i
    d_for_iter[0, 1] = d
    delta_p = p_for_iter[0, 1] - p_for_iter[0, 0] + 0.00000000000001
    delta_i = i_for_iter[0, 1] - p_for_iter[0, 0] + 0.00000000000001
    delta_d = d_for_iter[0, 1] - p_for_iter[0, 0] + 0.00000000000001
    p_for_iter[0, 0] = p_for_iter[0, 1]
    i_for_iter[0, 0] = i_for_iter[0, 1]
    d_for_iter[0, 0] = d_for_iter[0, 1]
    # p_ofpid = P(p)
    # i_ofpid = I(i)
    # d_ofpid = D(d)
    Greg = PID(p,i,d)
    G5 = G1 * G2 * G3 * Greg
    G_zam = G5 / (1 + G5)
    y1, t = con.step(G_zam,t)
    Value_state = ust(y1)
    maximom_of_func = max(y1)
    time = transition_time_of_proсess_andGraph(y1, Value_state, t, False)

    M_for_iter = maximom_of_func
    T_for_iter = time
    deltaM = overshoot - M_for_iter
    deltaT = treg - T_for_iter

    if Value_state == 1:
        # Roots_of_the_equation = pzmap(G_zam, False)
        # Alfa = np.array(Roots_of_the_equation[:, 0])
        # Alfa_min = min(Alfa)
        if maximom_of_func < overshoot:
            overshoot_condition = True
        if time < treg:
            processtime_condition = True
        if Value_state < 1.1 and Value_state > 0.95:
            value_condition = True

        if (overshoot_condition and processtime_condition and value_condition):
            end_t = tttt.time()
            transition_time_of_proсess_andGraph(y1, Value_state, t, True)
            print('Время подбора коэф = ', end_t - start_t)
            J = False

    # iter1 += 1
    # if iter1 % 3 == 1:
    k1 = for_iter(delta_p, overshoot_condition, processtime_condition, deltaM, deltaT, " P")
    if p + k1 > 0:
        p = p + k1
    else:
        p = p+abs(k1)

    # if iter1 % 3 == 2:
    k2 = for_iter(delta_i, overshoot_condition, processtime_condition, deltaM, deltaT, " I")
    if i + k2 > 0:
        i = i + k2
    else:
        i = i+abs(k2)

    # if iter1 % 3 == 0:
    k3 = for_iter(delta_d, processtime_condition, processtime_condition, deltaM, deltaT, " D")
    if d + k3 > 0:
        d = d + k3
    else:
        d = d + abs(k3)
    print("Установившееся значение: = ", Value_state)
    print("Максимальное значение: = ", maximom_of_func)
    print("Время переходного процесса: = ", time)
    print("P = ", p, "I = ", i, "D = ", d)

print(G_zam)
Pole(G_zam)
Freq(G_zam)
intergalnaya_Otsenka(G_zam)


