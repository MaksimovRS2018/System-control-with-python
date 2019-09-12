import numpy as np
import matplotlib.pyplot as plt
import control.matlab as con
import math


# import Proverka #не нужно

class Lab1():

    def BAGUVIX(GG1, name_gg1, GG2, name_gg2, t):  # функция для построения графиков характеристик
        topic = {
            'G1': 'безынерционного звена',
            'G2': 'апериодического звена',
            'G3': 'интегрирующего звена',
            'G4': 'реального диф. звена',
            'G5': 'идеального диф. звена',
        }  # словарь для графика
        if name_gg1 in topic:  # определяем какой именно строим, для графика
            k1 = topic[name_gg1]

        plt.figure(1)  # Вывод графиков в отдельном окне
        y1, t1 = con.step(GG1, t)
        y2, t2 = con.step(GG2, t)
        lines = [y1, y2]
        # plt.subplot(1, 1, 1)  # 1цифра - количество строк в графике, 2 -тьиьтиколичество графиков в строке, 3 -номер графика
        lines[0], lines[1] = plt.plot(y1, 'r', y2, 'g')
        plt.legend(lines, ['W для 1', 'W для 2'], loc='best', ncol=2, fontsize=10)
        plt.title('Переходная функция' + '\n для ' + k1, fontsize=10)
        plt.ylabel('W')
        plt.xlabel('t, c')
        plt.grid()

        plt.figure(2)
        y2, t2 = con.impulse(GG2, t)
        y1, t1 = con.impulse(GG1, t)
        lines[0], lines[1] = plt.plot(y1, 'r', y2, 'g')
        plt.legend(lines, ['W для 1', 'W для 2'], loc='best', ncol=2, fontsize=10)
        plt.title('Импульсная функция' + '\n для ' + k1, fontsize=10)
        plt.ylabel('h')
        plt.xlabel('t, c')
        plt.grid()

        plt.figure(3)
        mag1, phase1, omega1 = con.bode(GG1, dB=False)
        # plt.plot()
        plt.title('Частотные характеристики' + "\n для " + k1, fontsize=10, y=2.2)
        mag1, phase1, omega1 = con.bode(GG2, dB=False)
        plt.plot()
        plt.title('Частотные характеристики' + "\n для " + k1, fontsize=10, y=2.2)
        plt.show()

    # Для задания своих чисел через консоль
    # F = Proverka
    # td = F.Kek('Введите длительность процесса в секундах= ').Prov()
    # k1 = F.Kek('Введите коэф. к для безынерцинного звена = ').Prov1()
    # k2 = F.Kek('Введите коэф. к для апериодического звена = ').Prov1()
    # T2 = F.Kek('Введите коэф. T для апериодического звена = ').Prov1()
    # k3 = F.Kek('Введите коэф. к для интегрального звена = ').Prov1()
    # T3 = int(input('Введите коэф. T для интегрального звена = '))
    # k4 = F.Kek('Введите коэф. к для реал дифф звена = ').Prov1()
    # T4 = F.Kek('Введите коэф. T для реал дифф звена = ').Prov1()

    r = 2  # для изменения
    td = 300
    t = np.linspace(1, 2, td)  # задаем значения времени t от 1 до td с шагом 2 в мс
    k1 = 2  # коэф. к для безынерцинного звена 1
    k2 = 1  # коэф. к для апериодического звена 1
    T2 = 2  # коэф. T для апериодического звена1
    k3 = 3  # коэф. к для интегрального звена 1
    T3 = 1  # коэф. T для интегрального звена 1
    k4 = 4  # коэф. к для реал дифф звена1
    T4 = 2  # коэф. T для реал дифф звена 1
    # k5 = 1
    # T5 = 10 ** -4 #малое число, иначе не получится

    G11 = con.tf(k1, 1)  # Безынерционное звено 1
    G21 = con.tf(k2, [T2, 1])  # Апериодическое звено звено 1
    G31 = con.tf(k3, [T3, 0])  # Интегрирующее звено 1
    G41 = con.tf([k4, 0], [T4, 1])  # Реальное диф. звено звено 1
    # G51 = con.tf([k5, 0], [T5, 1])  # Идеальное диф. звено звено 1

    k1 = k1 * r  # коэф. к для безынерцинного звена 2
    k2 = k2 * r  # коэф. к для апериодического звена 2
    T2 = T2 / r  # коэф. T для апериодического звена 2
    k3 = k3 * r  # коэф. к для интегрального звена 2
    T3 = 1  # коэф. T для интегрального звена 2
    k4 = k4 * r  # коэф. к для реал дифф звена 2
    T4 = T4 / r  # коэф. T для реал дифф звена 2
    # k5 = k5 * r

    G12 = con.tf(k1, 1)  # Безынерционное звено 2
    G22 = con.tf(k2, [T2, 1])  # Апериодическое звено звено 2
    G32 = con.tf(k3, [T3, 0])  # Интегрирующее звено 2
    G42 = con.tf([k4, 0], [T4, 1])  # Реальное диф. звено звено 2
    # G52 = con.tf([k5, 0], [T5, 1])  # Идеальное диф. звено звено 1

    Go1 = BAGUVIX(G11, 'G1', G12, 'G12', t)  # Безынерционные звенья
    Go2 = BAGUVIX(G21, 'G2', G22, 'G22', t)  # Апериодические звенья
    Go3 = BAGUVIX(G31, 'G3', G32, 'G32', t)  # Интегрирующие звенья
    Go4 = BAGUVIX(G41, 'G4', G42, 'G42', t)  # Реальное диф. звенья
    # Go5 = BAGUVIX(G51, 'G5', G52, 'G52', t) #Идеальное диф. звенья
