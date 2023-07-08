#!/usr/bin/env python
# coding: utf-8

# # Ecuación de Onda con condiciones Dirichlet y Neumann

# Importamos las librerías necesarias

# In[1]:


import numpy as np
import scipy as sp
from scipy.sparse import diags
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Preparamos la función de animación para poder mostrar el resultado

# In[2]:


from matplotlib import rc
rc('animation', html='jshtml')
def plotAnimation(time, space, solution):
  fig, ax = plt.subplots()
  ax.set_xlabel('x')
  ax.set_ylabel('tª')
  plotLine, = ax.plot(space, np.zeros(len(space))*np.NaN, 'r-')
  plotTitle = ax.set_title("t=0")
  ax.set_xlim(np.min(space)-0.1, np.max(space)+0.1)
  ax.set_ylim(-1, np.max(solution[0])+0.1)

  def animate(t):
      pp = solution[t]
      plotLine.set_ydata(pp)
      plotTitle.set_text(f"t = {time[t]:.1f}")
      return [plotLine,plotTitle]


  anim = animation.FuncAnimation(fig, func=animate, frames=np.arange(0, len(solution)-1, 1), blit=True)
  return anim


# ## Implementación del método numérico

# Definimos la función usando matrices, aunque luego impondré las condiciones a mano:

# In[11]:


def Wave(f, df, Np, Mp, condition, t0=0.0, tn=20.0, x0=0.0, xn=10.0):
  N=Np
  M=Mp
  a=t0
  L=tn
#Generamos los nodos
  k = (L-a)/N
  h = (xn - x0)/M
  nodost = [a+k*i for i in np.arange(N)]
  nodosx = [x0+h*i for i in np.arange(M)]

#Preparamos la matriz tridiagonal para la derivada segunda en espacio
  diagonals = [-2*np.ones(M), np.ones(M-1), np.ones(M-1)]
  mat = diags(diagonals, (0, 1, -1)).toarray()
#Aplicamos la condición inicial a los nodos
  inicial = np.array([f(x) for x in nodosx])
#Calculamos el primer valor a utilizar en el cálculo usando la derivada
  anterior = np.array([f(x) + k*df(x) for x in nodosx])
#Iteramos para calcular las soluciones en cada tiempo a partir de las dos anteriores e imponemos condiciones
  result=[anterior, inicial]
  for i in np.arange(2, N+1):
    actual = 2*result[i-1] - result[i-2] + ((k/h)**2)*mat@result[i-1]
    #Condiciones tipo Neumann
    if(condition=="Neumann"):
        actual[0] = actual[1]
        actual[M-1] = actual[M-2]
    else:
    #Condiciones tipo Dirichlet
        actual[0] = actual[M-1] = 0
    result.append(actual)
  return nodost, nodosx, result


# ## Onda centrada

# ### Dirichlet

# In[12]:


def onda(x):
  return np.sin(x*np.pi/10)
def derivada(x):
  return np.cos(x*np.pi/10)*np.pi/10


# In[13]:


nodost, nodosx, result = Wave(onda, derivada, 500, 100, "Dirichlet")
plotAnimation(nodost, nodosx, result)


#  ### Neumann

# In[14]:


nodost, nodosx, result = Wave(onda, derivada, 500, 100, "Neumann")
plotAnimation(nodost, nodosx, result)


# ## Perturbación a la izquierda

# He elegido esta condición inicial (a base de ir refinando para ponerla a la izquierda, quizá es algo complicada), a diferencia de la anterior se introduce cierta oscilación extraña, como no parece inestable y no ocurría en el otro caso, puede tratarse de ruido numérico procedente de la aproximación de la condición inicial y su complejidad.

# In[20]:


def onda(x):
  return 7 * np.exp(-1*x**2)*np.sin(x*np.pi/10)
def derivada(x):
  return (-1/10)*np.exp(-1*x**2)*(140*x*np.sin(x*np.pi/10) -np.cos(x*np.pi/10)*np.pi*7)


# In[21]:


nodost, nodosx, result = Wave(onda, derivada, 500, 100, "Dirichlet", tn=40)
plotAnimation(nodost, nodosx, result)


# In[23]:


nodost, nodosx, result = Wave(onda, derivada, 500, 100, "Neumann")
plotAnimation(nodost, nodosx, result)

