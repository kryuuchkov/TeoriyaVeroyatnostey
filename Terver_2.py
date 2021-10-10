#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


# In[9]:


#исходные данные
mu = 1.4
sigma = 1.14
lamb = 2.14
a = 0.7
b = 6.7
N = 200
seed=22


# In[11]:


rng = np.random.default_rng(seed)
r_normal = rng.normal(mu, sigma, N).round(5)
r_exp = rng.exponential(1 / lamb, N).round(5)
r_uniform = rng.uniform(a, b, N).round(5)
print('Выборка 200 псевдослучайных чисел:\n')
print('- распределенных по нормальному закону с параметрами mu = {} и sigma = {}\n'.format(mu, sigma))
print(r_normal)
print('\n- распределенных по показательному закону с параметром lambda = {}\n'.format(lamb))
print(r_exp)
print('\n- распределенных равномерно на отрезке [{}, {}]\n'.format(a, b))
print(r_uniform)


# In[12]:


r_normal.sort()
r_exp.sort()
r_uniform.sort()
print('Упорядоченная выборка 200 псевдослучайных чисел:\n')
print('- распределенных по нормальному закону с параметрами mu = {} и sigma = {}\n'.format(mu, sigma))
print(r_normal)
print('\n- распределенных по показательному закону с параметром lambda = {}\n'.format(lamb))
print(r_exp)
print('\n- распределенных равномерно на отрезке [{}, {}]\n'.format(a, b))
print(r_uniform)


# In[13]:


#интервальный ряд и ассоциированный статистический ряд
def stat_series(r, low, high):
   N = len(r)
   m = 1 + int(math.log2(N))
   h = round((high - low) / m, 5)
   a = []
   x = []
   n = []
   w = []

   i = 0
   j = low
   for _ in range(m):
     t = round(j + h, 5)
     k = 0
     while (i < N) and (r[i] <= t):
       i += 1
       k += 1
     if i == N:
       a.append((j, high))
       x.append(round((j + high) / 2, 5))
     else:
       a.append((j, t))
       x.append(round((j + t) / 2, 5))
     n.append(k)
     w.append(k / N)
     j = t
   return a, x, n, w, h


# In[14]:


a_normal, x_normal, n_normal, w_normal, h_normal = stat_series(r_normal,r_normal[0], r_normal[N - 1])
a_exp, x_exp, n_exp, w_exp, h_exp = stat_series(r_exp, 0, r_exp[N - 1])
a_uniform, x_uniform, n_uniform, w_uniform, h_uniform = stat_series(r_uniform, a, b)


# In[15]:


print('Интервальный и ассоциированный статистический ряды для нормального распределения')
pd.DataFrame([x_normal, n_normal, w_normal], columns=a_normal, index=['x*', 'n', 'w'], )


# In[16]:


print('Интервальный и ассоциированный статистический ряды для показательного распределения')
pd.DataFrame([x_exp, n_exp, w_exp], columns=a_exp, index=['x*', 'n', 'w'], )


# In[17]:


print('Интервальный и ассоциированный статистический ряды для равномерного распределения')
pd.DataFrame([x_uniform, n_uniform, w_uniform], columns=a_uniform, index=['x*', 'n', 'w'], ) 


# In[18]:


#теоретические вероятности
from scipy.stats import norm
def normal_prob(mu, sigma, x):
 return np.array([norm.cdf(x[i][1], loc=mu, scale=sigma) - norm.cdf(x[i][0], loc=mu, scale=sigma) for i in range(len(x))]).round(5)
def exp_prob(lamb, x):
 return np.array([- (math.exp(- lamb * x[i][1]) - math.exp(- lamb * x[i][0])) for i in range(len(x))]).round(5)
def uniform_prob(a, b, x):
 return np.array([(x[i][1] - x[i][0]) / (b - a) for i in
range(len(x))]).round(5)


# In[19]:


p_normal = normal_prob(mu, sigma, a_normal)
abs_diff_prob = np.abs(w_normal - p_normal)
print('Наибольшее абсолютное отклонение: {}'.format(np.max(abs_diff_prob)))
pd.DataFrame([w_normal, p_normal, abs_diff_prob], columns=a_normal, index=['Частота', 'Теоретическая вероятность', 'Абсолютное отклонение'])


# In[20]:


p_exp = exp_prob(lamb, a_exp)
abs_diff_prob = np.abs(w_exp - p_exp)
print('Наибольшее абсолютное отклонение: {}'.format(np.max(abs_diff_prob)))
pd.DataFrame([w_exp, p_exp, abs_diff_prob], columns=a_exp, index=['Частота', 'Теоретическая вероятность', 'Абсолютное отклонение'])


# In[21]:


p_uniform = uniform_prob(a, b, a_uniform)
abs_diff_prob = np.abs(w_uniform - p_uniform)
print('Наибольшее абсолютное отклонение: {}'.format(np.max(abs_diff_prob)))
pd.DataFrame([w_uniform, p_uniform, abs_diff_prob], columns=a_uniform, index=['Частота', 'Теоретическая вероятность', 'Абсолютное отклонение'])


# In[22]:


#гистограмма относительных частот и теоретических вероятностей
def poligon_plot(ax, x, w, h):
 w = np.array(w)
 ax.bar(x, w / h, h, edgecolor='black')
 ax.set_xticks(np.arange(round(x[0] - 1), round(x[len(x) - 1] + 1)))
 ax.set_yticks(np.arange(0, np.max(w / h) + 0.1, 0.1))


# In[23]:


fig, ax = plt.subplots(1, 3, figsize = (27, 9))
ax[0].set_title("Гистограмма относительных частот\nнормального распределения")
ax[1].set_title("Гистограмма относительных частот\nпоказательного распределения")
ax[2].set_title("Гистограмма относительных частот\nравномерного распределения")
poligon_plot(ax[0], x_normal, w_normal, h_normal)
poligon_plot(ax[1], x_exp, w_exp, h_exp)
poligon_plot(ax[2], x_uniform, w_uniform, h_uniform)


# In[24]:


#график эмпирической функции распределения
def F_emp_plot(ax, x):
 ax.set_xticks(np.arange(round(x[0] - 1), round(x[len(x) - 1] + 1)))
 ax.set_yticks(np.arange(-0.1, 1.2, 0.1))
 ax.step(x, [i / N for i in range(1, N + 1)])


# In[25]:


fig, ax = plt.subplots(1, 3, figsize = (27, 9))
ax[0].set_title("График эмпирической функции \n нормального распределения")
ax[1].set_title("График эмпирической функции \n показательного распределения")
ax[2].set_title("График эмпирической функции \n равномерного распределения на отрезке")
F_emp_plot(ax[0], r_normal)
F_emp_plot(ax[1], r_exp)
F_emp_plot(ax[2], r_uniform)


# In[28]:


# выборочный k-ый момент
def mu_(x, w, k):
     return round(np.sum((x ** k) * w), 5)
    
# выборочные характеристики
def emp_properties(a, x, n, w, h):
     a = np.array(a)
     x = np.array(x)
     n = np.array(n)
     w = np.array(w)
     x_ = round(np.sum(x * w), 5) # выборочное среднее
     D_B = round(np.sum(((x - x_) ** 2) * w) - h ** 2 / 12, 5) # выборочная дисперсия
     sigma_ = round(np.sqrt(D_B), 5) # выборочное среднее квадратическоеотклонение
 # выборочная мода
     idx = np.argmax(n)
     n_idx = np.size(idx)
     if n_idx == 1:
         if idx == 0:
             Mo_ = a[idx][0] + h * (w[idx]) / (2 * w[idx] - w[idx + 1])
         elif idx == len(w) - 1:
             Mo_ = a[idx][0] + h * (w[idx] - w[idx - 1]) / (2 * w[idx] - w[idx - 1])
         else:
             Mo_ = a[idx][0] + h * (w[idx] - w[idx - 1]) / (2 * w[idx] - w[idx - 1] - w[idx + 1])
         Mo_ = round(Mo_, 5)
     elif n_idx == idx[0] - idx[n_idx - 1] + 1:
         if idx[0] == 0:
             Mo_ = a[idx[0]][0] + h * (w[idx[0]]) / (2 * w[idx[0]] - w[idx[n_idx - 1] + 1])
         elif idx[n_idx - 1] == len(w) - 1:
             Mo_ = a[idx[0]][0] + h * (w[idx[0]] - w[idx[0] - 1]) / (2 * w[idx[0]] - w[idx[0] - 1])
         else:
             Mo_ = a[idx[0]][0] + h * (w[idx[0]] - w[idx[0] - 1]) / (2 * w[idx[0]] - w[idx[0] - 1] - w[idx[n_idx - 1] + 1])
         Mo_ = round(Mo_, 5)
     else:
         Mo_ = 'не существует'
     # выборочная медиана
     i = 0
     s = w[0]
     while (s < 0.5):
         i += 1
         s += w[i]
     if s == 0.5:
         Me_ = a[i][1]
     else:
         Me_ = round(a[i][0] + h / w[i] * (0.5 - (s - w[i])), 5)
     gamma_1_ = round((mu_(x, w, 3) - 3 * mu_(x, w, 2) * x_ + 2 * (x_ ** 3)) / (sigma_ ** 3), 5) # выборочный коэффициент асимметрии
     gamma_2_ = round((mu_(x, w, 4) - 4 * mu_(x, w, 3) * x_ + 6 * mu_(x, w, 2) * (x_ ** 2) - 3 * (x_ ** 4)) / (sigma_ ** 4) - 3, 5) # выборочный коэффициент эксцесса
     return x_, D_B, sigma_, Mo_, Me_, gamma_1_, gamma_2_


# In[29]:


# теоретические характеристики
def normal_properties(mu, sigma):
 M = mu # математическое ожидание
 D = round(sigma ** 2, 5) # дисперсия
 Mo = mu # мода
 Me = mu # медиана
 gamma_1 = 0 # коэффициент асимметрии
 gamma_2 = 0 # коэффициент эксцесса
 return M, D, sigma, Mo, Me, gamma_1, gamma_2
def exp_properties(lamb):
 M = round(1 / lamb, 5) # математическое ожидание
 D = round(1 / lamb ** 2, 5) # дисперсия
 sigma = round(1 / lamb, 5) # среднее квадратическое отклонение
 Mo = 0 # мода
 Me = round(math.log(2) / lamb, 5) # медиана
 gamma_1 = 2 # коэффициент асимметрии
 gamma_2 = 6 # коэффициент эксцесса
 return M, D, sigma, Mo, Me, gamma_1, gamma_2
def uniform_properties(a, b):
 M = (a + b) / 2 # математическое ожидание
 D = round((b - a) ** 2 / 12, 5) # дисперсия
 sigma = (a + b) / 2 # среднее квадратическое отклонение
 Mo = (a + b) / 2 # мода
 Me = (a + b) / 2 # медиана
 gamma_1 = 0 # коэффициент асимметрии
 gamma_2 = - 6 / 5 # коэффициент эксцесса
 return M, D, sigma, Mo, Me, gamma_1, gamma_2


# In[31]:


# относительное отклонение
def relative_diff(a, b):
 res = []
 for i in range(len(a)):
     if b[i] == 0:
         res.append('-')
     else:
         res.append(round(abs(a[i] - b[i]) / abs(b[i]), 5))
     return res
 


# In[33]:


print('Характеристики нормального распределения')
emp = emp_properties(a_normal, x_normal, n_normal, w_normal, h_normal)
teor = normal_properties(mu, sigma)
abs_diff = np.abs(np.array(emp) - np.array(teor))
rel_diff = relative_diff(np.array(emp), np.array(teor))
pd.DataFrame([emp, teor, abs_diff, rel_diff],
 columns=['Выборочное среднее', 'Выборочная дисперсия', 'Выборочное среднее квадратическое отклонение','Выборочная мода', 'Выборочная медиана', 'Выборочный коэффициентассиметрии', 'Выборочный коэффициент эксцесса'],
 index=['Экспериментальное значение', 'Теоретическое значение', 'Абсолютное отклонение', 'Относительное отклонение'])


# In[34]:


print('Характеристики показательного распределения')
emp = emp_properties(a_exp, x_exp, n_exp, w_exp, h_exp)
teor = exp_properties(lamb)
abs_diff = np.abs(np.array(emp) - np.array(teor))
rel_diff = relative_diff(np.array(emp), np.array(teor))
pd.DataFrame([emp, teor, abs_diff, rel_diff],
 columns=['Выборочное среднее', 'Выборочная дисперсия','Выборочное среднее квадратическое отклонение','Выборочная мода', 'Выборочная медиана', 'Выборочный коэффициентассиметрии', 'Выборочный коэффициент эксцесса'],
 index=['Экспериментальное значение', 'Теоретическоезначение', 'Абсолютное отклонение', 'Относительное отклонение'])


# In[35]:


print('Характеристики равномерного распределения на отрезке')
emp = emp_properties(a_uniform, x_uniform, n_uniform, w_uniform,
h_uniform)
teor = uniform_properties(a, b)
abs_diff = np.abs(np.array(emp) - np.array(teor))
rel_diff = relative_diff(np.array(emp), np.array(teor))
pd.DataFrame([emp, teor, abs_diff, rel_diff],
 columns=['Выборочное среднее', 'Выборочная дисперсия', 'Выборочное среднее квадратическое отклонение','Выборочная мода', 'Выборочная медиана', 'Выборочный коэффициентассиметрии', 'Выборочный коэффициент эксцесса'],
 index=['Экспериментальное значение', 'Теоретическоезначение', 'Абсолютное отклонение', 'Относительное отклонение'])


# In[ ]:




