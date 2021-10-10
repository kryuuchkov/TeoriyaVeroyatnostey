#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import math

from scipy.stats import norm


# In[2]:


"""

Исходные выборки загружаются из файла "МС_ЛР_4.xlsx":

- лист "D1_2021" - выборка для Задания 1

- лист "D3_2021" - выборка для Заданий 2-6

Результатом работы программы является файл "МС_ЛР_4_таблицы.xlsx":

- лист "1" с таблицей в раздел Анализ результатов и выводы для Задания 1

- лист "2.1" с таблицей в раздел Результаты расчетов для Задания 2

- лист "2.2" с таблицей в раздел Анализ результатов и выводы для Задания 2

- лист "3.1" с таблицей в раздел Результаты расчетов для Задания 3

- лист "3.2" с таблицей в раздел Анализ результатов и выводы для Задания 3

- лист "4.1" с таблицей в раздел Анализ результатов и выводы для Задания 4(1)

- лист "4.2.1" с таблицей в раздел Результаты расчетов для Задания 4(2)

- лист "4.2.2" с таблицей в раздел Анализ результатов и выводы для Задания 4(2)

- лист "4.3.1" с таблицей в раздел Результаты расчетов для Задания 4(3)

- лист "4.3.2" с таблицей в раздел Анализ результатов и выводы для Задания 4(3)

- лист "5.1" с таблицей в раздел Результаты расчетов для Задания 5

- лист "5.2" с таблицей в раздел Анализ результатов и выводы для Задания 5

- лист "6.1" с таблицей в раздел Анализ результатов и выводы для Задания 6(1)

- лист "6.2.1" с таблицей в раздел Результаты расчетов для Задания 6(2)

- лист "6.2.2" с таблицей в раздел Анализ результатов и выводы для Задания 6(2)

"""


# In[9]:


def Student_params(arr_1, arr_2):
    
    """
    Расчет параметров и выборочного значения критерия для проверки

    гипотезы о равенстве математических ожиданий (распределение Стьюдента)

    Аргументы:

    arr_1, arr_2 (numpy.ndarray): выборки

    Возвращает:

    x_, y_ (float): выборочные средние

    x_2, y_2 (float): выборочные несмещенные дисперсии

    s_x, s_y (float)

    T (float): выборочное значение критерия

    """

    N = len(arr_1)

    M = len(arr_2)

    x_ = np.mean(arr_1)

    y_ = np.mean(arr_2)

    x_2 = np.mean(arr_1 ** 2)

    y_2 = np.mean(arr_2 ** 2)

    s_x = N / (N - 1) * (x_2 - x_ ** 2)

    s_y = M / (M - 1) * (y_2 - y_ ** 2)
    
    T = (x_ - y_) / math.sqrt(s_x * (N - 1) + s_y * (M - 1)) * math.sqrt((N * M * (N + M - 2)) / (N + M))

    return np.round([x_, y_, x_2, y_2, s_x, s_y, T], 5)


# In[10]:


def Fisher_Snedecor_mean_params(arr):

    """Расчет параметров и выборочного значения критерия для проверки

    гипотезы о равенстве математических ожиданий (распределение Фишера-Снедекора)
    
    Аргументы:
    
    arr (numpy.ndarray): массив выборок

    Возвращает:

    s_1 (float): общая сумма квадратов отклонений

    s_2 (float): факторная сумма квадратов отклонений

    s_3 (float): остаточная сумма квадратов отклонений

    s_2_2, s_3_2 (float)

    k_1, k_2 (float): число степеней свободы

    F (float): выборочное значение критерия

    """

    m = arr.shape[0]

    N = arr.shape[1]

    u_ = np.mean(arr)

    u_j = np.mean(arr, axis=1)

    s_1 = np.sum((arr - u_) ** 2)

    s_2 = N * np.sum((u_j - u_) ** 2)

    s_3 = s_1 - s_2

    k_1 = m - 1

    k_2 = m * (N - 1)

    s_2_2 = s_2 / k_1

    s_3_2 = s_3 / k_2

    F = (s_2_2) / (s_3_2)

    return np.round([s_1, s_2, s_3, s_2_2, s_3_2, k_1, k_2, F], 5)


# In[11]:


def Fisher_Snedecor_variance_params(arr_1, arr_2):

    """Расчет параметров и выборочного значения критерия для проверки

    гипотезы о равенстве дисперсий (распределение Стьюдента)

    Аргументы:

    arr_1, arr_2 (numpy.ndarray): выборки

    Возвращает:

    s_x, s_y (float)

    k_1, k_2 (float): число степеней свободы
    
    F (float): выборочное значение критерия

    """

    N = len(arr_1)

    M = len(arr_2)

    x_ = np.mean(arr_1)

    y_ = np.mean(arr_2)

    x_2 = np.mean(arr_1 ** 2)

    y_2 = np.mean(arr_2 ** 2)

    s_x = N / (N - 1) * (x_2 - x_ ** 2)

    s_y = M / (M - 1) * (y_2 - y_ ** 2)

    if s_x >= s_y:

        s_1 = s_x

        s_2 = s_y

        k_1 = N - 1

        k_2 = M - 1
    
    else:

        s_1 = s_y

        s_2 = s_x

        k_1 = M - 1

        k_2 = N - 1

        F = s_1 / s_2

    return np.round([s_1, s_2, k_1, k_2, F], 5)


# In[13]:


def main():

    #загрузка начальных данных

    a = 1.70

    sigma = 2.24

    alpha = 0.05

    arr_1 = np.array(pd.read_excel(r"C:\Users\User\Desktop\фуфа\МС_ЛР_4.xlsx", sheet_name="D1_2021", header=None))

    arr_2 = np.array(pd.read_excel(r"C:\Users\User\Desktop\фуфа\МС_ЛР_4.xlsx", sheet_name="D3_2021", header=None))

    arr_2 = arr_2.transpose()

    cases = [(1, 2), (1, 3), (2, 3)]
    
    N = len(arr_1)

    m = len(arr_2)

    NM = [len(arr_2[i]) for i in range(m)]

    #Задание1

    mu_1 = round(np.mean(arr_1), 5)

    C = round(a + (sigma / math.sqrt(N)) * norm.ppf(1 - alpha), 5)

    hypothesis = "H_0" if mu_1 <= C else "H_1"

    df_1 = pd.DataFrame([[mu_1, alpha, C, hypothesis]], columns=["mu_1", "alpha", "C_alpha", "Вывод"])

    #Задание2

    T = [Student_params(arr_2[i - 1], arr_2[j - 1]) for i, j in cases]

    t_alpha = np.round([t.ppf(1 - alpha / 2, NM[i - 1] + NM[j - 1] - 2) for i, j in cases], 5)

    hypothesis = ["ВЕРНА" if abs(T[i][-1]) <= t_alpha[i] else "НЕВЕРНА" for i in range(len(cases))]

    df_2_1 = pd.DataFrame(T, columns=["x_", "y_", "x_^2", "y_^2", "S_x^2", "S_y^2", "T_NM"], index=cases)
    
    df_2_2 = pd.DataFrame([[abs(T[i][-1]), t_alpha[i], hypothesis[i]] for i in range(len(cases))], columns=["|T_NM|", "t_кр(N+M-2)", "Вывод"], index=cases)

    #Задание3

    F = Fisher_Snedecor_mean_params(arr_2)

    F_alpha = round(f.ppf(1 - alpha, F[-3], F[-2]), 5)

    hypothesis = "ВЕРНА" if F[-1] <= F_alpha else "НЕВЕРНА"

    df_3_1 = pd.DataFrame([F], columns=["S_общ", "S_факт", "S_ост", "S_факт^2", "S_ост^2", "k_1", "k_2", "F_Nm"])

    df_3_2 = pd.DataFrame([[F[-1], alpha, F_alpha, hypothesis]], columns=["F_Nm", "alpha", "F_кр(k_1, k_2)", "Вывод"])

    #Задание4_1

    pval_1 = round(f_oneway(arr_2[0], arr_2[1], arr_2[2])[1], 5)

    hypothesis = "ВЕРНА" if pval_1 >= alpha else "НЕВЕРНА"

    df_4_1 = pd.DataFrame([[pval_1, alpha, hypothesis]], columns=["pval[anova]", "alpha", "Вывод"])
    
    #Задание4_2

    pval_2 = np.round([ttest_ind(arr_2[i - 1], arr_2[j - 1])[1] for i, j in cases], 5)

    hypothesis = ["ВЕРНА" if pval_2[i] >= alpha else "НЕВЕРНА" for i in range(len(cases))]

    df_4_2_1 = pd.DataFrame(pval_2, columns=["pval[student]"], index=cases)

    df_4_2_2 = pd.DataFrame([[pval_2[i], alpha, hypothesis[i]] for i in range(len(cases))], columns=["pval[student]", "alpha", "Вывод"], index=cases)

    #Задание4_3

    pval_3 = np.round([ttest_ind(arr_2[i - 1], arr_2[j - 1],equal_var=False)[1] for i, j in cases], 5)

    hypothesis = ["ВЕРНА" if pval_3[i] >= alpha else "НЕВЕРНА" for i in range(len(cases))]

    df_4_3_1 = pd.DataFrame(pval_3, columns=["pval[welch]"], index=cases)

    df_4_3_2 = pd.DataFrame([[pval_3[i], alpha, hypothesis[i]] for i in range(len(cases))], columns=["pval[welch]", "alpha", "Вывод"], index=cases)

    #Задание 5

    F = [Fisher_Snedecor_variance_params(arr_2[i - 1], arr_2[j - 1]) for i, j in cases]

    F_alpha = np.round([f.ppf(1 - alpha / 2, F[i][-3], F[i][-2]) for i in range(len(cases))], 5)

    hypothesis = ["ВЕРНА" if abs(F[i][-1]) <= F_alpha[i] else "НЕВЕРНА" for i in range(len(cases))]

    df_5_1 = pd.DataFrame(F, columns=["S_1^2", "S_2^2", "k_1", "k_2", "F_NM"], index=cases)

    df_5_2 = pd.DataFrame([[abs(F[i][-1]), F_alpha[i], hypothesis[i]] for i in range(len(cases))], columns=["F_NM", "F_кр(k_1, k_2)", "Вывод"], index=cases)

    #Задание6_1

    pval_1 = round(bartlett(arr_2[0], arr_2[1], arr_2[2])[1], 5)

    hypothesis = "ВЕРНА" if pval_1 >= alpha else "НЕВЕРНА"

    df_6_1 = pd.DataFrame([[pval_1, alpha, hypothesis]], columns=["pval[bartlett]", "alpha", "Вывод"])

    #Задание6_2

    pval_2 = np.round([levene(arr_2[i - 1], arr_2[j - 1])[1] for i, j in cases], 5)

    hypothesis = ["ВЕРНА" if pval_2[i] >= alpha else "НЕВЕРНА" for i in range(len(cases))]

    df_6_2_1 = pd.DataFrame(pval_2, columns=["pval[levene]"], index=cases)

    df_6_2_2 = pd.DataFrame([[pval_2[i], alpha, hypothesis[i]] for i in range(len(cases))], columns=["pval[levene]", "alpha", "Вывод"], index=cases)
    
    #сохранение таблиц

    with pd.ExcelWriter("МС_ЛР_4_че.xlsx") as writer:

        df_1.to_excel(writer, sheet_name="1", index=False)

        df_2_1.to_excel(writer, sheet_name="2.1")

        df_2_2.to_excel(writer, sheet_name="2.2")

        df_3_1.to_excel(writer, sheet_name="3.1", index=False)

        df_3_2.to_excel(writer, sheet_name="3.2", index=False)

        df_4_1.to_excel(writer, sheet_name="4.1", index=False)

        df_4_2_1.to_excel(writer, sheet_name="4.2.1")

        df_4_2_2.to_excel(writer, sheet_name="4.2.2")

        df_4_3_1.to_excel(writer, sheet_name="4.3.1")

        df_4_3_2.to_excel(writer, sheet_name="4.3.2")

        df_5_1.to_excel(writer, sheet_name="5.1")

        df_5_2.to_excel(writer, sheet_name="5.2")

        df_6_1.to_excel(writer, sheet_name="6.1", index=False)

        df_6_2_1.to_excel(writer, sheet_name="6.2.1")

        df_6_2_2.to_excel(writer, sheet_name="6.2.2")


# In[17]:


if __name__ == '__main__':
    
    try:

        main()

    except KeyboardInterrupt:

        print()


# In[ ]:




