# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 16:47:22 2021

@author: alexa
"""
from __future__ import division
import sys
import numpy as np
import random as random
import cv2 as cv



class Rede_neural:

    def __init__(self):
        self.modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        self.configFile = "models/deploy.prototxt"
        self.net = cv.dnn.readNetFromCaffe(self.configFile, self.modelFile)  
        self.source = 0
        self.conf_threshold = 0.7
        if len(sys.argv) > 1:
            self.source = sys.argv[1]
            
    def detectFaceOpenCVDnn(self, frame):
 
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

        self.net.setInput(blob)
        detections = self.net.forward()
        bboxes = []
        

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
            
            return frameOpencvDnn, bboxes
class PSO:
    def __init__(self, W, ROI, template, limiar):
        self.q=template.shape[0]
        self.p=template.shape[1]
        self.n=ROI.shape[0]
        self.m=ROI.shape[1]
        M=self.m-(self.p)-10
        N=self.n-(self.q)-10
        if M < 1:
            M=1
        if N < 1:
            N=1
        self.W=W
        self.ROI=ROI
        self.limiar=limiar
        self.template=template
        self.particulas=[np.random.randint(0, M, W),np.random.randint(0, N, W)]
        self.Vit=[]
        for _ in range(len(self.particulas)):
            self.Vit.append(np.random.randint(0,10,W))
        self.particula_historic=[[0] * W,[0] * W]
        self.Ehistoric=[0] * W
        self.E=[]
    def __Avaliar(self, particula):
        e=0
        T1=0
        I1=0
        x=particula[0]
        y=particula[1]				
        for c in (range(self.p-1)):
          for t in (range(self.q-1)):
            A=y+t
            B=x+c
            if A >= (self.n-1):
                A=self.n-1
            if B >= (self.m-1):
                B=self.m-1
            I1 = I1 + self.ROI[(A)][(B)]
            T1 = T1 + self.template[t][c]
        I2=(1/(self.p*self.q))*I1
        T2=(1/(self.p*self.q))*T1
        N=0
        D1=0
        D2=0
        for c in (range(self.p-1)):
          for t in (range(self.q-1)):
            A=y+t
            B=x+c
            if A >= (self.n-1):
                A=self.n-1
            if B >= (self.m-1):
                B=self.m-1
            f_I = (self.ROI[(A)][(B)] - I2)
            f_T = (self.template[t][c] - T2)
 #           print('f_I= ', f_I, 'f_T= ', f_T, 'N= ', N)
            N = N + (f_I * f_T)       
 #           print('N= ', N)
            D1 = D1 + (f_I*f_I)
            D2 = D2 + (f_T*f_T)
        e = (N)/((D1*D2)**(1/2))
        return e
    def __popula_E(self, limiar):
        self.E=[0]*self.W
        
        for i in range(self.W):
            if self.E[i]<=limiar:
                particula=[]
                for j in range(len(self.particulas)):
                    particula.append(self.particulas[j][i])
                self.E[i]=self.__Avaliar(particula)
            else:
                break
        return self.E 
    
    def __popula_E_historic(self):       
        for i in range(self.W):
            if self.E[i] > self.Ehistoric[i]:
                self.Ehistoric[i] = self.E[i]
                for j in range(len(self.particulas)):
                    self.particula_historic[j][i]=self.particulas[j][i]
                    
    def calculaNCC(self):
        self.__popula_E(self.limiar)
        self.__popula_E_historic()
        Etemp=self.Ehistoric[0]
        for i in range(self.W):
            if self.Ehistoric[i]>Etemp:
                Etemp=self.Ehistoric[i]
        return Etemp
    
    def RetornaMelhorParticula(self):
        MelhorPartic=[]
        Etemp=self.Ehistoric[0]
        for j in range(len(self.particula_historic)):
            MelhorPartic.append(self.particula_historic[j][0])
        for i in range(self.W):
            if self.Ehistoric[i]>Etemp:
                Etemp=self.Ehistoric[i]
                MelhorPartic=[]
                for j in range(len(self.particula_historic)):
                    MelhorPartic.append(self.particula_historic[j][i])
        return MelhorPartic
    
    def __Pibest(self,j,i):
        return self.particula_historic[j][i]
    
    def __Pgbest(self,j):
        temp=self.particulas[j][0]
        Etemp=self.E[0]
        for i in range(self.W):
            if self.E[i]>Etemp:
                temp=self.particulas[j][i]
                Etemp=self.E[i]
        return temp

    def evolui_particulas(self,Itr, Itrmax, wmax=1, wmin=0.7, Vmax=10 ):
        c1=0.5	
        c2=0.2     
        w=(wmax)-(((wmax-wmin)/Itrmax)*Itr)
        for i in range(self.W):
            particula=0
            for j in range(len(self.particulas)):
                r1=random.random()
                r2=random.random()
                Pibet=self.__Pibest(j,i)
                Pgbet=self.__Pgbest(j)
                particula=self.particulas[j][i]
                self.Vit[j][i]=self.Vit[j][i]*w+c1*r1*(Pibet-particula)+r2*c2*(Pgbet-particula)
                if self.Vit[j][i] > Vmax:
                    self.Vit[j][i] = Vmax
                self.particulas[j][i]=self.particulas[j][i]+self.Vit[j][i]
 