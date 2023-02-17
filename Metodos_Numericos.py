import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

x, y = sy.symbols("x y")

def Metodo_Euler_Generalizado(ecuacion_f, A, N, y0, t0):
	H = A / N
	T = np.zeros(N+1)
	Y = np.zeros(N+1)
	Y[0] = y0
	T[0] = t0
	eval_funcion  = sy.lambdify([x, y], ecuacion_f)

	for i in range(N):
		T[i+1] = T[i] + H
		Y[i+1] = Y[i] + H*eval_funcion(T[i], Y[i])
	return T,Y

def Metodo_Euler_3_Orden_Generalizado(ecuacion_f, A, N, y0, t0):
	H = A / N
	T = np.zeros(N+1)
	Y = np.zeros(N+1)
	Y[0] = y0
	T[0] = t0
	parcial_f_respecto_x    = sy.diff(ecuacion_f, x)
	parcial_f_respecto_y    = sy.diff(ecuacion_f, y)
	eval_funcion            = sy.lambdify([x, y], ecuacion_f)
	eval_funcion_parcial_x  = sy.lambdify([x, y], parcial_f_respecto_x)
	eval_funcion_parcial_y  = sy.lambdify([x, y], parcial_f_respecto_y)

	for i in range(N):
		T[i+1]  = T[i] + H
		Y[i+1]  = Y[i] + H*eval_funcion(T[i], Y[i])
		Y[i+1] += ((H**2)/2)*(eval_funcion_parcial_x(T[i], Y[i]))
		Y[i+1] += ((H**2)/2)*(eval_funcion(T[i], Y[i])*eval_funcion_parcial_y(T[i], Y[i]))
	return T,Y


def Metodo_Euler_Modificado_Generalizado(ecuacion_f, A, N, y0, t0):
	H = A / N
	T = np.zeros(N+1)
	Y = np.zeros(N+1)
	Y[0] = y0
	T[0] = t0
	
	eval_funcion = sy.lambdify([x, y], ecuacion_f)
	
	for i in range(N):
		T[i+1]  = T[i] + H
		Y[i+1]  = Y[i] + (H/2)*(eval_funcion(T[i], Y[i]))
		Y[i+1] += (H/2)*eval_funcion(T[i] + H, Y[i] + H*eval_funcion(T[i], Y[i])) 
	return T,Y

def RungeKutta(ecuacion_f, A, N, y0, t0):
	H = A / N
	T = np.zeros(N+1)
	Y = np.zeros(N+1)
	Y[0] = y0
	T[0] = t0
	eval_funcion = sy.lambdify([x, y], ecuacion_f)
	
	for i in range(N):
		LK1 = eval_funcion(T[i],Y[i])
		LK2 = eval_funcion(T[i] + (1/2)*H, Y[i] + (1/2)*H*LK1)
		LK3 = eval_funcion(T[i] + (1/2)*H, Y[i] + (1/2)*H*LK2)
		LK4 = eval_funcion(T[i] + H, Y[i] + H*LK3)
		T[i+1]  = T[i] + H
		Y[i+1]  = Y[i] + (H/6)*(LK1 + 2*LK2 + 2*LK3 + LK4)
	return T,Y

