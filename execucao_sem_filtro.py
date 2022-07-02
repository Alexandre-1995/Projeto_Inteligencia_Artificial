# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 16:57:20 2021

@author: alexa
"""

from algoritmos_com_poda import PSO
from algoritmos_com_poda import Rede_neural
import cv2 as cv
import time
import statistics as st

path='./'

class Execucao:

    def __init__(self):
        
        self.localizacao=0
        self.Itrmax = 50
        self.limiar=0.75
        self.Itr=0
        self.marcador="Not Found"
        self.xpso1=0
        self.xpso2=0
        self.ypso1=0
        self.ypso2=0
        self.VItr=[]
        self.VE=[]
        self.Velapsed=[]
        self.execs=0
        self.Vexecs=[]
        self.frame_count = 0
        self.tt_opencvDnn = 0
        
    def Exec(self):
#-------------------------------------------------------------------------------------------------------------------
        self.rede=Rede_neural()
        cap = cv.VideoCapture("video.mp4")
        hasFrame, frame = cap.read()
        
#-------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------    
        while (1):
            hasFrame, frame = cap.read()
            if not hasFrame:
                break
            self.frame_count += 1
    
            outOpencvDnn, bboxes = self.rede.detectFaceOpenCVDnn(frame)
            y1=bboxes[0][0]
            x1=bboxes[0][1]
            y2=bboxes[0][2]
            x2=bboxes[0][3]          
            img=outOpencvDnn[x1:x2, y1:y2]
            cv.imshow('img',img)
#-------------------------------------------------------------------------------------------------------------------
            template = cv.imread('template_esquerda_2.jpg',cv.IMREAD_GRAYSCALE)
            if self.localizacao==0:
                meio=int((y2-y1)/2)
                centro=y1+meio
                centrox=x1+int(3*(x2-x1)/4)
                img21=outOpencvDnn[x1:centrox, centro:y2]   

                ROI = cv.cvtColor(img21, cv.COLOR_BGR2GRAY)
#                ROI=cv.medianBlur(ROI,9)
#                template=cv.medianBlur(template,9)
                populacao=PSO(5, ROI, template, self.limiar)
                yanterior=0
            else:
                print(self.localizacao+1,'ª localização')
                ROI = img[self.ypso1-7:self.ypso2+7, self.xpso1-7:self.xpso2+7]
                yanterior=self.ypso1-7
                ROI = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)
#                ROI=cv.medianBlur(ROI,9)
#                template=cv.medianBlur(template,9)
                populacao=PSO(5, ROI, template, self.limiar)
            cv.imshow('ROI',ROI)
            q=template.shape[0]
            p=template.shape[1]
    
            self.execs+=1
            self.Vexecs.append(self.execs)
            start = time.time()
            while (self.Itr<self.Itrmax and self.marcador=="Not Found"):
                self.Itr+=1       
                E=populacao.calculaNCC()        
                if E < self.limiar:
                    populacao.evolui_particulas(self.Itr, self.Itrmax)
                else:
                    print('encontrado, E= ', E, 'Interação, ', self.Itr)
                    posicao=populacao.RetornaMelhorParticula()
                    print(posicao)
                    vermelho = (0, 0, 255)
                    self.xpso1=posicao[0]+meio
                    self.ypso1=posicao[1]+yanterior
                    self.xpso2=posicao[0]+p+meio
                    self.ypso2=self.ypso1+q
                    novo_rosto=cv.rectangle(img, ((self.xpso1),  (self.ypso1)), ((self.xpso2), (self.ypso2)), vermelho)
                    self.marcador="Found"
                    self.localizacao += 1
                    cv.imshow('deteccao', novo_rosto)
            self.VE.append(E)
            self.VItr.append(self.Itr)
            done = time.time()
            elapsed = done - start
            self.Velapsed.append(elapsed)       
            
            self.Itr=0
            self.marcador="Not Found"
    


#-------------------------------------------------------------------------------------------------------------------    
            if self.frame_count == 1:
                self.tt_opencvDnn = 0

            k = cv.waitKey(10)
            if k == 27:
                break
                cv.destroyAllWindows()
        

        primeiro_coef = self.VE.pop(0)
        primeiro_elapsed=self.Velapsed.pop(0)
        Primeiro_N_Itr=self.VItr.pop(0)
        média_coef=st.mean(self.VE)
        média_elapsed=st.mean(self.Velapsed)
        média_n_itr=st.mean(self.VItr)
        return(primeiro_coef,primeiro_elapsed,Primeiro_N_Itr,média_coef,média_elapsed,média_n_itr)










