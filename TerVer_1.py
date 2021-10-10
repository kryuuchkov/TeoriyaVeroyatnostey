#!/usr/bin/env python
# coding: utf-8

# In[17]:


from scipy.stats import binom, geom, poisson
import matplotlib.pyplot as plt
import numpy as np
from math import factorial, exp, sqrt, log


# In[18]:


#исходные данные
n = 21
p = 0.27
lamb = 1.28
N = 200
random_state = 42


# In[19]:


r_binom = binom.rvs(n, p, size = N, random_state = random_state)
r_geom = geom.rvs(p, size = N, random_state = random_state)
r_geom = r_geom - 1
r_poisson = poisson.rvs(lamb, size = N, random_state = random_state)
print('Выборка 200 псевдослучайных чисел:\n')
print('- распределенных по биномиальному закону с параметрами n = {} и p = {}\n'.format(n, p))
print(r_binom)
print('\n- распределенных по геометрическому закону с параметром p = {}\n'.format(p))
print(r_geom)
print('\n- распределенных по закону Пуассона с параметром lambda = {}\n'.format(lamb))
print(r_poisson)


# In[20]:


r_binom.sort()
r_geom.sort()
r_poisson.sort()
print('Упорядоченная выборка 200 псевдослучайных чисел:\n')
print('- распределенных по биномиальному закону с параметрами n = {} и p = {}\n'.format(n, p))
print(r_binom)
print('\n- распределенных по геометрическому закону с параметром p = {}\n'.format(p))
print(r_geom)
print('\n- распределенных по закону Пуассона с параметром lambda = {}\n'.format(lamb))
print(r_poisson)


# In[21]:


#статистический ряд
def stat_series(r):
 x = np.array([r[0]])
 n = np.array([0])
 w = np.array([0.0])
 s = np.array([0.0])
 m = 0
 N = len(r)
    
 for i in range(len(r)):
    if (x[m] == r[i]):
     n[m] += 1
    else:
     x = np.append(x, r[i])
     n = np.append(n, 1)
     w = np.append(w, 0)
     w[m] = n[m] / N
     s[m] += w[m]
     s = np.append(s, s[m])
     m += 1

 w[m] = n[m] / N
 s[m] += w[m]
 return x, n, w, s


# In[22]:


x_binom, n_binom, w_binom, s_binom = stat_series(r_binom)
x_geom, n_geom, w_geom, s_geom = stat_series(r_geom)
x_poisson, n_poisson, w_poisson, s_poisson = stat_series(r_poisson)
print('Статистические ряды:\n')
print('- биномиальное распределение')
print('x: {}\nn: {}\nw: {}\ns: {}\n'.format(x_binom, n_binom, w_binom, s_binom))
print('- геометрическое распределение\n')
print('x: {}\nn: {}\nw: {}\ns: {}\n'.format(x_geom, n_geom, w_geom, s_geom))
print('- распределение Пуассона\n')
print('x: {}\nn: {}\nw: {}\ns: {}\n'.format(x_poisson, n_poisson, w_poisson, s_poisson))


# In[23]:


#теоретические вероятности
def binom_prob(n, p, x):
    res = np.zeros(len(x))
    for i in range(len(x)):
     res[i] = round(factorial(n) / (factorial(x[i]) * factorial(n - x[i])) * p ** x[i] * (1 - p) ** (n - x[i]), 5)
    return res

def geom_prob(p, x):
    res = np.zeros(len(x))
    for i in range(len(x)):
      res[i] = round(p * (1 - p) ** x[i], 5)
    return res

def poisson_prob(lamb, x):
    res = np.zeros(len(x))
    k = exp(-lamb)
    for i in range(len(x)):
     res[i] = round(lamb ** x[i] / factorial(x[i]) * k, 5)
    return res


# In[24]:


#теоретические вероятности
def binom_prob(n, p, x):
 res = np.zeros(len(x))
 for i in range(len(x)):
  res[i] = round(factorial(n) / (factorial(x[i]) * factorial(n - x[i])) * p ** x[i] * (1 - p) ** (n - x[i]), 5)
 return res

def geom_prob(p, x):
 res = np.zeros(len(x))
 for i in range(len(x)):
  res[i] = round(p * (1 - p) ** x[i], 5)
 return res

def poisson_prob(lamb, x):
 res = np.zeros(len(x))
 k = exp(-lamb)
 for i in range(len(x)):
  res[i] = round(lamb ** x[i] / factorial(x[i]) * k, 5)
 return res


# In[25]:


#полигон относительных частот и теоретических вероятностей
def poligon_x_w(x, w):
 x_poligon = np.arange(np.max(x) + 1)
 w_poligon = np.zeros(np.max(x) + 1)
 i = 0
 for j in range(len(w_poligon)):
  if x[i] == j:
   w_poligon[j] = w[i]
   i += 1
  else:
   w_poligon[j] = 0
 return x_poligon, w_poligon
def poligon_plot(ax, x, w, p):
 ax.plot(x, w, label = 'Относительная частота')
 ax.plot(x, p, 'r', label = 'Теоретическая вероятность')
 ax.set_xticks(np.arange(len(w)))
 ax.set_yticks(np.arange(-0.1, np.max(w) + 0.1, 0.1))
 ax.legend()
 ax.grid(True, linestyle = '--')


# In[34]:


fig, ax = plt.subplots(1, 3, figsize = (27, 9))
ax[0].set_title("Полигон относительных частот \n и теоретических вероятностей \n биномиального распределения")
ax[1].set_title("Полигон относительных частот \n и теоретических вероятностей \n геометрического распределения")
ax[2].set_title("Полигон относительных частот \n и теоретических вероятностей \n распределения Пуассона")
X_binom, W_binom = poligon_x_w(x_binom, w_binom)
P_binom = binom_prob(n, p, X_binom)
poligon_plot(ax[0], X_binom, W_binom, P_binom)
X_geom, W_geom = poligon_x_w(x_geom, w_geom)
P_geom = geom_prob(p, X_geom)
poligon_plot(ax[1], X_geom, W_geom, P_geom)
X_poisson, W_poisson = poligon_x_w(x_poisson, w_poisson)
P_poisson = poisson_prob(lamb, X_poisson)
poligon_plot(ax[2], X_poisson, W_poisson, P_poisson)
print('Относительные частоты и теоретические вероятности:\n')
print('- биномиальное распределение\n')
abs_ = np.round(np.abs(W_binom - P_binom), 5)
print('X: {}\nW: {}\nP: {}\n|W - P|: {}\nmax|W - P| = {}\nSum(W) = {}\nSum(P) = {}\n '.format(X_binom, W_binom, P_binom, abs_, np.max(abs_), np.sum(W_binom), round(np.sum(P_binom), 5)))
print('- геометрическое распределение', end = '\n\n')
abs_ = np.round(np.abs(W_geom - P_geom), 5)
print('X: {}\nW: {}\nP: {}\n|W - P|: {}\nmax|W - P| = {}\nSum(W) = {}\nSum(P) = {}\n '.format(X_geom, W_geom, P_geom, abs_, np.max(abs_), np.sum(W_geom), round(np.sum(P_geom), 5)))
print('- распределение Пуассона', end = '\n\n')
abs_ = np.round(np.abs(W_poisson - P_poisson), 5)
print('X: {}\nW: {}\nP: {}\n|W - P|: {}\nmax|W - P| = {}\nSum(W) = {}\nSum(P) = {}\n '.format(X_poisson, W_poisson, P_poisson, abs_, np.max(abs_),
np.sum(W_poisson), round(np.sum(P_poisson), 5)))


# In[27]:


#график эмпирической функции распределения
def F_emp_plot(ax, x, s):
 ax.set_xticks(np.arange(0, x[len(x) - 1] + 2))
 ax.set_yticks(np.arange(-0.1, 1.2, 0.1))
 ax.arrow(0, 0, x[0], 0, color = 'b', length_includes_head = True, head_width = 0.01, head_length = 0.1)
 for i in range(len(s) - 1):
  ax.arrow(x[i], s[i], x[i + 1] - x[i], 0, color = 'b',
  length_includes_head = True, head_width = 0.01, head_length = 0.1)
  ax.arrow(x[len(x) - 1], s[len(s) - 1], 1, 0, color = 'b',
  length_includes_head = True, head_width = 0.01, head_length = 0.1)
  ax.grid(True, linestyle = '--')


# In[28]:


fig, ax = plt.subplots(1, 3, figsize = (27, 9))
ax[0].set_title("График эмпирической функции \n биномиального распределения")
ax[1].set_title("График эмпирической функции \n геометрического распределения")
ax[2].set_title("График эмпирической функции \n распределения Пуассона")
F_emp_plot(ax[0], x_binom, s_binom)
F_emp_plot(ax[1], x_geom, s_geom)
F_emp_plot(ax[2], x_poisson, s_poisson)


# In[29]:


# выборочный k-ый момент
def mu_(x, w, k):
 return round(np.sum((x ** k) * w), 5)
# выборочные характеристики
def emp_properties(x, n, w, s):
    x_ = round(np.sum(x * w), 5) # выборочное среднее
    D_B = round(np.sum(((x - x_) ** 2) * w), 5) # выборочная дисперсия
    sigma_ = round(np.sqrt(D_B), 5) # выборочное среднее квадратическоеотклонение
 # выборочная мода
    idx = np.argmax(n)
    l = np.size(idx)
    if l == 1:
      Mo_ = x[idx]
    elif n_idx == idx[0] - idx[l - 1]:
      Mo_ = 1 / 2 * (idx[0] + idx[l - 1])
    else:
      Mo_ = 'не существует'
 # выборочная медиана
    i = 0
    while (s[i] < 0.5):
      i += 1
    if s[i] == 0.5:
      Me_ = 1 / 2 * (idx[i] + idx[i + 1])
    else:
      Me_ = x[i]
    gamma_1_ = round((mu_(x, w, 3) - 3 * mu_(x, w, 2) * x_ + 2 * (x_ ** 3)) / (sigma_ ** 3), 5) # выборочный коэффициент асимметрии
    gamma_2_ = round((mu_(x, w, 4) - 4 * mu_(x, w, 3) * x_ + 6 * mu_(x, w,2) * (x_ ** 2) - 3 * (x_ ** 4)) / (sigma_ ** 4) - 3, 5) # выборочный коэффициент эксцесса
    return x_, D_B, sigma_, Mo_, Me_, gamma_1_, gamma_2_


# In[30]:


# теоретические характеристики
def binom_properties(n, p):
 M = n * p # математическое ожидание
 D = round(M * (1 - p), 5) # дисперсия
 sigma = round(sqrt(D), 5) # среднее квадратическое отклонение
 # мода
 k = (n + 1) * p
 if k.is_integer():
  Mo = k - 1 / 2
 else:
  Mo = int(k)
  Me = round(M) # медиана
  gamma_1 = round((1 - 2 * p) / sigma, 5) # коэффициент асимметрии
  gamma_2 = round((1 - 6 * p * (1 - p)) / D, 5) # коэффициент эксцесса
 return M, D, sigma, Mo, Me, gamma_1, gamma_2
def geom_properties(p):
 M = round((1 - p) / p, 5) # математическое ожидание
 D = round((1 - p) / p ** 2, 5) # дисперсия
 sigma = round(sqrt(1 - p) / p, 5) # среднее квадратическое отклонение
 Mo = 0 # мода
 # медиана
 k = log(2) / log(1 - p)
 if k.is_integer():
  Me = - k - 1 / 2
 else:
  Me = int(-k)
  gamma_1 = round((2 - p) / sqrt(1 - p), 5) # коэффициент асимметрии
  gamma_2 = round(6 + p ** 2 / (1 - p), 5) # коэффициент эксцесса
 return M, D, sigma, Mo, Me, gamma_1, gamma_2
def poisson_properties(lamb):
 M = lamb # математическое ожидание
 D = lamb # дисперсия
 sigma = round(sqrt(lamb), 5) # среднее квадратическое отклонение
 Mo = int(lamb) # мода
 Me = int(lamb + 1 / 3 - 0.002 / lamb) # медиана
 gamma_1 = round(1 / sqrt(lamb), 5) # коэффициент асимметрии
 gamma_2 = round(1/ lamb, 5) # коэффициент эксцесса
 return M, D, sigma, Mo, Me, gamma_1, gamma_2


# In[31]:


# относительное отклонение
def relative_diff(a, b):
 if b == 0:
  return '-'
 else:
  return round(abs(a - b) / abs(b), 5)


# In[32]:


print('Характеристики (экспериментальное значение / теоретическое значение (абсолютное отклонение, относительное отклонение)):\n')
print('- биномиальное распределение:')
x_, D_B, sigma_, Mo_, Me_, gamma_1_, gamma_2_ = emp_properties(x_binom,n_binom, w_binom, s_binom)
M, D, sigma, Mo, Me, gamma_1, gamma_2 = binom_properties(n, p)
print(' - выборочное среднее: {} / {} ({}, {})\n - выборочная дисперсия: {} / {} ({}, {})\n - выборочное среднее квадратическое отклонение: {} / {} ({}, {})\n - выборочная мода: {} / {} ({}, {})\n - выборочная медиана: {} / {} ({}, {})\n - выборочный коэффициент ассиметрии: {} / {} ({}, {})\n - выборочный коэффициент эксцесса: {} / {} ({}, {})\n'.format(x_, M, round(abs(x_ - M), 5), relative_diff(x_, M), D_B, D, round(abs(D_B - D), 5), relative_diff(D_B, D),sigma_, sigma, round(abs(sigma_ - sigma), 5),
relative_diff(sigma_, sigma),
 Mo_, Mo, round(abs(Mo_ - Mo), 5), relative_diff(Mo_, Mo),
 Me_, Me, round(abs(Me_ - Me), 5), relative_diff(Me_, Me),
 gamma_1_, gamma_1, round(abs(gamma_1_ - gamma_1), 5),
relative_diff(gamma_1_, gamma_1),
 gamma_2_, gamma_2, round(abs(gamma_2_ - gamma_2), 5),
relative_diff(gamma_2_, gamma_2)))
print('- геометрическое распределение:')
x_, D_B, sigma_, Mo_, Me_, gamma_1_, gamma_2_ = emp_properties(x_geom,
n_geom, w_geom, s_geom)
M, D, sigma, Mo, Me, gamma_1, gamma_2 = geom_properties(p)
print(' - выборочное среднее: {} / {} ({}, {})\n - выборочная дисперсия: {} / {} ({}, {})\n - выборочное среднее квадратическое отклонение: {} / {} ({}, {})\n - выборочная мода: {} / {} ({}, {})\n - выборочная медиана: {} / {} ({}, {})\n - выборочный коэффициент ассиметрии: {} / {} ({}, {})\n - выборочный коэффициент эксцесса: {} / {} ({}, {})\n'.format(x_, M, round(abs(x_ - M), 5), relative_diff(x_, M),
 D_B, D, round(abs(D_B - D), 5), relative_diff(D_B, D),
 sigma_, sigma, round(abs(sigma_ - sigma), 5),
relative_diff(sigma_, sigma),
 Mo_, Mo, round(abs(Mo_ - Mo), 5), relative_diff(Mo_, Mo),
 Me_, Me, round(abs(Me_ - Me), 5), relative_diff(Me_, Me),
 gamma_1_, gamma_1, round(abs(gamma_1_ - gamma_1), 5),
relative_diff(gamma_1_, gamma_1),
 gamma_2_, gamma_2, round(abs(gamma_2_ - gamma_2), 5),
relative_diff(gamma_2_, gamma_2)))
print('- распределение Пуассона:')
x_, D_B, sigma_, Mo_, Me_, gamma_1_, gamma_2_ = emp_properties(x_poisson,
n_poisson, w_poisson, s_poisson)
M, D, sigma, Mo, Me, gamma_1, gamma_2 = poisson_properties(lamb)
print(' - выборочное среднее: {} / {} ({}, {})\n - выборочная дисперсия: {} / {} ({}, {})\n - выборочное среднее квадратическое отклонение: {} / {} ({}, {})\n - выборочная мода: {} / {} ({}, {})\n - выборочная медиана: {} / {} ({}, {})\n - выборочный коэффициент ассиметрии: {} / {} ({}, {})\n - выборочный коэффициент эксцесса: {} / {} ({}, {})\n'.format(x_, M, round(abs(x_ - M), 5), relative_diff(x_, M),
 D_B, D, round(abs(D_B - D), 5), relative_diff(D_B, D),
 sigma_, sigma, round(abs(sigma_ - sigma), 5),
relative_diff(sigma_, sigma),
 Mo_, Mo, round(abs(Mo_ - Mo), 5), relative_diff(Mo_, Mo),
 Me_, Me, round(abs(Me_ - Me), 5), relative_diff(Me_, Me),
 gamma_1_, gamma_1, round(abs(gamma_1_ - gamma_1), 5),
relative_diff(gamma_1_, gamma_1),
 gamma_2_, gamma_2, round(abs(gamma_2_ - gamma_2), 5),
relative_diff(gamma_2_, gamma_2)))


# In[ ]:




