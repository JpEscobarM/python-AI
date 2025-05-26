# Importaçoes iniciais
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import plotly.express as px
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

#PREPARAÇÃO DOS DADOS:

bike = pd.read_csv('bike_sharing_classificacao.csv')



X = bike.drop(columns=['demand_level'])
y = bike['demand_level'] #TARGET

X_dataFrame = pd.DataFrame(X, columns= ['temp', 'atemp', 'humidity',	'windspeed',	'hour',	'workingday', 'season'])

#====

#ANALISE EXPLORATORIA
#
# column_dict = {
#     'temp': 'Temperatura',
#     'atemp': 'Sensação Térmica',
#     'humidity': 'Umidade',
#     'windspeed': 'Velocidade do Vento',
#     'hour': 'Hora do Dia',
#     'workingday': 'Dia de Trabalho',
#     'season': 'Temporada'
# }
#
#
# print(bike.head())
#
# print(bike.info())
#
# print(bike.describe())
#
# plt.figure(figsize=(14, 10))
# bike.hist(bins=20, figsize=(14, 10), color='skyblue', edgecolor='black')
# plt.suptitle("Distribuição das Variáveis Numéricas", fontsize=16)
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(6,4))
# sb.countplot(x='demand_level', data=bike)
# plt.title('Distribuição das Classes - demand_level')
# plt.xlabel('Nível de Demanda')
# plt.ylabel('Contagem')
# plt.show()
#
# plt.figure(figsize=(10, 8))
# sb.heatmap(bike.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Mapa de Correlação entre Variáveis Numéricas", fontsize=14)
# plt.show()
#
# for col in X_dataFrame.columns:
#     fig = px.box(
#         bike,
#         x="demand_level",
#         y=col,
#         color="demand_level",
#         title=f"Boxplot Interativo: {column_dict[col]} por Nível de Demanda",
#     )
#     fig.show()
#
#     fig = px.scatter_3d(bike,
#                         x='temp', y='humidity', z='windspeed',
#                         color='demand_level',
#                         title="Gráfico Interativo 3D - Temperatura, Umidade e Vento",
#                         labels={"temp": "Temperatura", "humidity": "Umidade", "windspeed": "Vento"},
#                         opacity=0.7)
#     fig.show()

#====

#PREPARAÇÂO DOS DADOS

# Temperatura e sensação térmica trazem a mesma informação, portanto pode-se retirar uma das columas
X_dataFrame = X.drop(columns=['atemp'])

#dados desbalanceados portanto precisa de SMOTE pra poder treinar
print(y.value_counts())

dados_balanceados = SMOTE(k_neighbors=3, random_state=42)
X_, y_ = dados_balanceados.fit_resample(X_dataFrame.values, y)

uniformiza = MinMaxScaler()
uniformiza.fit(X_)
X_ = uniformiza.fit_transform(X_)

print(X_.shape)

print(y_.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, random_state=42, stratify = y_)

#====

#ARVORE DE DECISAO
#USA GRIDSEARCH PARA ENCONTRAR OS MELHORES PARAMETROS PRA ARVORE
tree_param = {'criterion':['entropy','gini'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],'min_samples_leaf':[1,2,3,4,5]}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), tree_param, cv=5)
grid.fit(X_, y_)
best_params = grid.best_params_
print("Melhores Parâmetros: ")
print(best_params)

#ARVORE DE DECISAO COM HOLDOUT:
tree = DecisionTreeClassifier(random_state=42, **best_params)
y_pred = tree.fit(X_train, y_train).predict(X_test)

print("Classification Report - Decision Tree (Hold-Out):\n")
print(classification_report(y_test, y_pred))

matrizConfusaoTree = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
labels = ['0 - Baixa', '1 - Média', '2 - Alta'] #demandas

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Nomes das colunas e classes
feature_names = ['temp', 'humidity', 'windspeed', 'hour', 'workingday', 'season']
class_names = ['Baixa', 'Média', 'Alta']

# Visualização da árvore
plt.figure(figsize=(15, 10))
plot_tree(tree,
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True,
          fontsize=12,
          proportion=True)
plt.title("Árvore de Decisão com SMOTE + Treino Balanceado")
plt.show()

sb.heatmap(matrizConfusaoTree, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.xlabel("Previsão")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Árvore de Decisão (Hold-Out)")
plt.show()

print("Acurácia: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

#====

#ARVORE DE DECISAO COM CROSS VALIDATION:

best_params = grid.best_params_
model_final = DecisionTreeClassifier(random_state=42, **best_params)

y_pred = cross_val_predict(model_final, X_, y_, cv=5)
model_final.fit(X_,y_)

print("Classification Report - Decision Tree (Cross-Validation):\n")
print(classification_report(y_, y_pred))

matrizConfusaoTree = confusion_matrix(y_, y_pred)
print(confusion_matrix(y_, y_pred))
labels = ['0 - Baixa', '1 - Média', '2 - Alta'] #demandas

sb.heatmap(matrizConfusaoTree, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.xlabel("Previsão")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Árvore de Decisão")
plt.show()

print("Acurácia: {:.2f}%".format(accuracy_score(y_, y_pred) * 100))

#=====

#K-NEAREST NEIGHBOURS( KNN)

knn_param = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski']}
grid = GridSearchCV(KNeighborsClassifier(), knn_param, cv=5)
grid.fit(X_, y_)
best_params = grid.best_params_
print("Melhores parâmetros:", best_params)

#KNN COM CROSS VALIDATION:
model_final = KNeighborsClassifier(**best_params)
model_final.fit(X_, y_)
y_pred = cross_val_predict(model_final, X_, y_, cv=5)

print("Classification Report - KNN (Cross-Validation):\n")
print(classification_report(y_, y_pred))

matrizConfusao = confusion_matrix(y_, y_pred)
print(confusion_matrix(y_, y_pred))
labels = ['0 - Baixa', '1 - Média', '2 - Alta'] #demandas

sb.heatmap(matrizConfusao, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.xlabel("Previsão")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Árvore de Decisão")
plt.show()

print("Acurácia: {:.2f}%".format(accuracy_score(y_, y_pred) * 100))

#====

#KNN COM HOLDOUT:

model_final = KNeighborsClassifier(**best_params).fit(X_train, y_train)
y_pred = model_final.predict(X_test)

print("Classification Report - KNN (Cross-Validation):\n")
print(classification_report(y_test, y_pred))

matrizConfusao = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
labels = ['0 - Baixa', '1 - Média', '2 - Alta'] #demandas

sb.heatmap(matrizConfusao, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.xlabel("Previsão")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Árvore de Decisão")
plt.show()

print("Acurácia: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

#====

#INPUT PARA TESTE COM O MELHOR RESULTADO DOS ANTERIORES:
#DECISION TREE COM HOLDOUT

best_params = {'criterion': 'entropy', 'max_depth': 8, 'min_samples_leaf': 1}

tree = DecisionTreeClassifier(random_state=42, **best_params)
tree.fit(X_train, y_train)


lista=[]
temperatura = float(input('Digite a temperatura:'))
lista.append(temperatura)
umidade = float(input('Digite a umidade:'))
lista.append(umidade)
ventoVelocidade = float(input('Digite a velocidade do vento:'))
lista.append(ventoVelocidade)
hora = float(input('Digite a hora:'))
lista.append(hora)
diaUtil = float(input('Digite o dia util [0] - Final de semana  | [1] - Dia Util : '))
lista.append(diaUtil)
estacao = float(input("Digite a estação: "))
lista.append(estacao)


lista_np = np.array(lista).reshape(1,-1)

entrada = uniformiza.transform(lista_np)

previsaoDemanda = tree.predict(entrada)
print("Demanda prevista: ", previsaoDemanda)
print("[0] - Baixa")
print("[1] - Media")
print("[2] - Alta")
