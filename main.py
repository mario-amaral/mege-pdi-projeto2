# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:56:53 2022

Projeto 2 de Processamento Digital de Imagem (MEGE 2022-23)

@author: Mário Amaral
"""

import numpy as np
from imageio.v3 import imread
from matplotlib import pyplot as plt
from skimage.morphology import rectangle, disk, dilation, erosion
from skimage.segmentation import watershed
from scipy import ndimage

#%% 1. Obtenção da imagem binária do leito do rio (riverbed)

"""
Definem-se as duas funções para obter reconstrução geodésica e reconstrução 
geodésica dual, utilizadas mais abaixo
"""

def reconst(mask, marker):
    a = 1
    ee = rectangle(3,3)
    while a!=0:
        D = dilation(marker,ee)
        R = np.minimum(mask.astype(float), D.astype(float))
        a = np.count_nonzero(marker != R)
        marker = np.copy(R)
    return R


def reconst_dual(mask, marker):
    a = 1
    ee = rectangle(3,3)
    while a!=0:
        E = erosion(marker,ee)
        R = np.maximum(mask.astype(float), E.astype(float))
        a = np.count_nonzero(marker != R)
        marker = np.copy(R)
    return R


Img = imread('lsat01.tif')
Img_red = Img[:,:,0] #canal vermelho é o que melhor isola a forma do rio

"""
Para destacar o leito do rio e tornar mais fácil a limiarização aplica-se uma
reconstrução geodésica dual com base numa região "escura" do rio (posição 352,81)
"""
x = 352
y = 81
h = np.ones(Img_red.shape)*255
h[x, y] = Img_red[x, y]

Img_red_rec = reconst_dual(Img_red, h)

"""
Aplica-se uma erosão para remover conjuntos de pixels isolados que dariam origem
a "ilhas" não desejadas na segmentação
"""

Img_red_rec = erosion(Img_red_rec, rectangle(3,3))

"""
A análise do histograma permite identificar a gama de frequencias correspondentes
à zona escura do leito do rio

"""
hist, _ = np.histogram(Img_red_rec, bins=256, range=(0, 256)) 

plt.figure(figsize = (16,9))
plt.suptitle("1. Obtenção da imagem binária do leito do rio (riverbed)")
plt.subplot(131); plt.plot(hist,'gray'); plt.title('1. Histograma lsta01.tif (canal R) ')

t_lower = 10 #limiar inferior
t_upper = 20 #limiar superior

riverbed_bin = np.logical_and(Img_red_rec > t_lower, Img_red_rec < t_upper)

distance = ndimage.distance_transform_edt(riverbed_bin)

"""
Determinada a função distância, podemos usar as reconstrução dual para obter
os mínimos regionais que basicamente definem as cinco zonas de terra em torno
do rio que pretendemos segmentar
"""
h = distance + 1
ee = disk(1)

min_reg = (reconst_dual(distance.astype(float), h) - distance) > 0

"""
Obtemos os marcadores para cada uma das zonas
"""
markers, _  = ndimage.label(min_reg) 

"""
A transformação de watershed segmenta as zonas de acordo com os marcados dos
mínimos regionais. Por erosão podemos obter as linhas de separação das zonas que
ficam equidistantes entre as "zonas de terra", produzindo a linha média do rio
"""

Ws = watershed(distance, markers)
Ws_lines = (Ws - erosion(Ws, ee)) > 0

plt.subplot(132); plt.imshow(Img); plt.title('Imagem lsat01.tif')
plt.subplot(133); plt.imshow(Ws_lines*255 + Img_red, 'gray'); plt.title('linha média do rio')

#%% 2. Limiarização da imagem ik02_1.tif pelo método de Otsu

Img = imread('ik02_1.tif')
Img_red = Img[:, :, 0]

hist, bin_edges = np.histogram(Img_red, bins=256, range=(0, 256))

"""
Com base no histgrama do canal R da imagem, vamos percorrer cada valor de 
frequência t para determinar a respetiva variância var2 = sigma ao quadrado. 
Os valores de var2 são registados num array. No loop temos.

hist_e/hist_d: histograma à esquerda/direita de t
w_e/w_d: peso à esquerda/direita de t
mean_e/mean_d: média à esquerda/direita de t
var2_e/var2_d: variância à esquerda/direita de t
"""

var2 = np.zeros(hist.shape)

for t in range(len(hist)):

    hist_e = hist[0:t] 
    hist_d = hist[t:len(hist)] 
    
    w_e = np.sum(hist_e)/np.sum(hist) 
    w_d = np.sum(hist_d)/np.sum(hist) 

    if np.sum(hist_e) == 0:  #é necessário criar uma exceção para a divisão por zero
        mean_e = 0. 
    else: mean_e = np.sum([i * p for i, p in enumerate(hist_e)])/np.sum(hist_e)
    
    mean_d = np.sum([i * p for i, p in enumerate(hist_d, t)])/np.sum(hist_d)
    
    if np.sum(hist_e) == 0: #é necessário criar uma exceção para a divisão por zero
        var2_e = 0. 
    else: var2_e = np.sum([(i - mean_e)**2 * p for i, p in enumerate(hist_e)])/np.sum(hist_e)
        
    var2_d = np.sum([(i - mean_d)**2 * p for i, p in enumerate(hist_d, t)])/np.sum(hist_d)
    
    var2[t] = w_e * var2_e + w_d * var2_d 

"""
t_otsu: threshold de otsu dado pela posição no array do valor mínimo de var2
"""
t_otsu = np.argmin(var2) 

Img_otsu = Img_red > t_otsu

plt.figure(figsize = (16,9))
plt.suptitle("2. Limiarização da imagem ik02_1.tif pelo método de Otsu")
plt.subplot(141); plt.imshow(Img); plt.title('Imagem ik02.tif')
plt.subplot(142); plt.plot(hist); plt.title('Histograma ik02.tif (Canal R)')
plt.subplot(143); plt.plot(var2); plt.title('variância')
plt.subplot(143); plt.plot(t_otsu, var2[t_otsu], 'ro'); plt.annotate('t_otsu = ' + str(t_otsu), (t_otsu + 10, var2[t_otsu]), color = 'red')
plt.annotate('min', xy=(t_otsu, 1), xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))
plt.subplot(144); plt.imshow(Img_otsu, 'gray'); plt.title('threshold Otsu')

#%% 3. Limiarização da imagem ik02_1.tif pelo método da máxima distância

"""
O código para determinação do valor de threshold que corresponde à distância
máxima entre histograma e uma linha reta que une os respetivos pontos máximo e 
mínimo é definida numa função get_tdmax
Os dois pontos de definição da reta são:
    x1, y1 = pico máximo do histograma
    x2, y2 = primeiro valor do histograma com valor zero
"""

def get_tdmax(hist):

    xmax = np.where(hist == np.max(hist))[0][0]
    ymax = hist[xmax]
    
    x1 = xmax
    y1 = ymax
    
    x2 = np.where(hist == 0)[0][0] 
    y2 = hist[x2]
    
    a1 = (y2-y1)/(x2-x1)
    b1 = y1 - a1 * x1
    
    a2 = -1 / a1
    b2 = np.zeros((256)); x = np.copy(b2); y = np.copy(b2); d = np.copy(b2)
    
    for i in range(xmax, 256):
        b2[i] = hist[i] - a2 * i
        x[i] = (b2[i] - b1) / (a1 - a2)
        y[i] = a2 * (b2[i] - b1) /(a1 - a2) + b2[i]
        d[i] = np.sqrt(((x[i] - i)**2 + (y[i] - hist[i])**2))
    
    return np.argmax(d)


Img = imread('ik02_1.tif')
Img_r = Img[:, :, 0]
Img_g = Img[:, :, 1]
Img_b = Img[:, :, 2]

"""
Por tentativas sucessivas, cheguei à conclusão que a melhor forma de isolar
as casas é começando por subtrair as bandas vermelha (r) e azul (b).
Temos de ter o cuidado de trabalhar com floats na subtração para evitar os 
erros de arredondamento ao trabalhar com int
"""

Img_d = np.abs(Img_r.astype(float) - Img_b.astype(float))

"""
O histograma com que vamos trabalhar será então o histograma das diferenças 
entre estas bandas
"""

hist_d, _ = np.histogram(Img_d, bins=256, range=(0, 256)) 

tdmax = get_tdmax(hist_d) #threshold dado pela distância máxima

"""
Da limiarização pelo tdmax, obtemos uma imagem binária limiarizada da diferença
entre as bandas: Img_d_t 
"""

Img_d_t = Img_d > tdmax

"""
Aplicamos esta imagem limiarizada Img_d_t como 'máscara' binária nas três
bandas rgb da imagem original e assim obteremos uma limiarização em 24-bit que
conterá uma representação dos telhados das casas
"""

Img_t = np.uint8(np.zeros(Img.shape))
Img_t[:, :, 0] = Img_r * Img_d_t
Img_t[:, :, 1] = Img_g * Img_d_t
Img_t[:, :, 2] = Img_b * Img_d_t

plt.figure(figsize = (16,9))
plt.suptitle("3. Representação telhados na imagem ik02_1.tif pelo método da máx distância")

plt.subplot(131); plt.imshow(Img); plt.title('ik02_1.tif')
plt.subplot(132); plt.plot(hist_d, color = 'black'); plt.title('hist diferença entre bandas r e b')
plt.subplot(132); plt.plot(tdmax, hist_d[tdmax], 'ro'); plt.annotate('tdmax = ' + str(tdmax), (tdmax + 10, hist_d[tdmax]), color = 'red')
plt.subplot(133); plt.imshow(Img_t); plt.title('casas')
