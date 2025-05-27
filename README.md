# Projeto Machine Learning

[![NPM](https://img.shields.io/npm/l/react)]((https://github.com/JpEscobarM/python-AI/blob/main/LICENSE)) 

## Sobre o projeto

Projeto criado a para fins de estudo de algoritimos de machine learning como Decision Tree e k-Nearest Neighbors (KNN). 
## Processo

### Analise exploratória

Após carregamento dos dados, deve-se fazer uma análise inicial das informações contidas no dataset, para assim compreender e visualizar, superficialmente, com que tipo de dado estamos a tratar.
Neste processo, oito colunas serão analisadas, sendo elas: temp, atemp, humidity, windspeed, hour, workingday, season e demand_level. A variável demand_level é o nosso alvo, ela pode assumir 3 valores distintos: 0 (baixa), 1 (média), 2 (alta). 
Portanto, inicialmente, fizemos a contagem de quantas situações temos para cada valor da variável alvo. Conforme a Figura 1, fica claro que existem muito mais instâncias contendo o valor 0 do que os demais, criando um desbalanceamento no dataset. Com essa informação em mente, posteriormente teremos que utilizar algum método para corrigir esse desbalanceamento e nivelar os casos.

