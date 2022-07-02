# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 14:51:38 2021

@author: alexa
"""

from execucao_sem_poda import Execucao
import matplotlib.pyplot as grafico
import statistics as st

execs=0

Vexecs=[]

VE=[]
Velapsed=[]
VItr=[]

VE_medio=[]
Velapsed_medio=[]
VItr_medio=[]

for i in range(100):
    print('tentativa número ', i)
    teste=Execucao()
    resultado=teste.Exec()
    execs+=1
    Vexecs.append(execs)
    VE.append(resultado[0])
    Velapsed.append(resultado[1])
    VItr.append(resultado[2])
    
    VE_medio.append(resultado[3])
    Velapsed_medio.append(resultado[4])
    VItr_medio.append(resultado[5])

grafico.xlabel('Tentativas')
grafico.ylabel('Coeficiente de correlação nas primeiras localizações')
grafico.plot(Vexecs, VE, marker='o')
grafico.grid(True)       
grafico.show()

grafico.xlabel('Tentativas')
grafico.ylabel('Elapsed de localização nas primeiras localizações')
grafico.plot(Vexecs, Velapsed, marker='o')
grafico.grid(True)
grafico.show()

grafico.xlabel('Tentativas')
grafico.ylabel('Número de iterações realizadas nas primeiras localizações')
grafico.plot(Vexecs, VItr, marker='o')
grafico.grid(True)
grafico.show()

grafico.xlabel('Tentativas')
grafico.ylabel('Coeficiente de correlação na média das localizações')
grafico.plot(Vexecs, VE_medio, marker='o')
grafico.grid(True)       
grafico.show()

grafico.xlabel('Tentativas')
grafico.ylabel('Elapsed de localização na média das localizações')
grafico.plot(Vexecs, Velapsed_medio, marker='o')
grafico.grid(True)
grafico.show()

grafico.xlabel('Tentativas')
grafico.ylabel('Número de iterações realizadas na média das localizações')
grafico.plot(Vexecs, VItr_medio, marker='o')
grafico.grid(True)
grafico.show()

print('Coeficiente, primeiras localizações: ',round(st.mean(VE),3), 'desvio padrão: ',round(st.pstdev(VE),3))
print('Elapsed, primeiras localizações: ',round(st.mean(Velapsed),3), 'desvio padrão: ',round(st.pstdev(Velapsed),3))
print('Iterações, primeiras localizações: ',round(st.mean(VItr),3), 'desvio padrão: ',round(st.pstdev(VItr),3))
print('Coeficiente, média das localizações: ',round(st.mean(VE_medio),3), 'desvio padrão: ',round(st.pstdev(VE_medio),3))
print('Elapsed, média das localizações: ',round(st.mean(Velapsed_medio),3), 'desvio padrão: ',round(st.pstdev(Velapsed_medio),3))
print('Iterações, média das localizações: ',round(st.mean(VItr_medio),3), 'desvio padrão: ',round(st.pstdev(VItr_medio),3))


