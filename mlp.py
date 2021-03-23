import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model #Classe que permite criar o objeto que vai representar a Rede Neural
from keras.layers import Input, Dense, Dropout, Flatten #Classe que permite criar as camadas da rede neural
from keras.optimizers import SGD #Classe que permite trabalhar com o Otmizador alterando seus hiperparâmetros
from keras.utils import to_categorical #Função que permite transformar as saídas de 0, 1, 2 para [0,0,0], [0,1,0], [0,0,1]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score #Funções para calcular a acurácia do modelo
from sklearn.manifold import TSNE

sca = MinMaxScaler()
tsne = TSNE(n_components=2)

dados = pd.read_csv('C:/Users/brunn/Google Drive/bases/classificacao/reuters.csv')
dados = dados[dados['y'] < 10]
X = sca.fit_transform(dados.drop(['y'], axis=1).values)
y = dados['y'].values
c = np.size(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=dados['y'].values)

camada_entrada = Input((X.shape[1],))

camada1 = Dense(units=64, activation='relu')(camada_entrada)
camada2 = Dense(units=32, activation='relu')(camada1)
camada_saida = Dense(units=c, activation='softmax')(camada2)

mlp = Model(camada_entrada, camada_saida)
mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mlp.summary()

relatorio = mlp.fit(X_train, to_categorical(y_train), batch_size=256, epochs=100,shuffle=True,validation_data=(X_test, to_categorical(y_test)))
y_pred = np.argmax(mlp.predict(X_test),1)

TX = tsne.fit_transform(X_test)

plt.rcParams['figure.figsize'] = (10,5)
fig, axs = plt.subplots(1, 2)
fig.subplots_adjust(left=0.07, bottom=0.135, right=0.98, top=0.93, wspace=0.22, hspace=0.45)
cores = ['blue', 'red', 'green', 'yellow', 'brown', 'orange', 'purple', 'pink', 'black', 'black', 'pink', 'purple', 'gray']

axs[0].set_title('Base Real')
axs[1].set_title('Base Classificada')

axs[0].scatter(TX[:,0], TX[:,1], c=[cores[i] for i in y_test.astype(np.int64)], s=10)
axs[1].scatter(TX[:,0], TX[:,1], c=[cores[i] for i in y_pred.astype(np.int64)], s=10)

plt.show()