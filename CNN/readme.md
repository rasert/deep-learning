# Análise de Projeto: Construindo uma CNN para Classificação de Imagens com Keras

## Objetivo

Este documento resume o processo iterativo de construção e otimização de uma Rede Neural Convolucional (CNN) usando Keras/TensorFlow. O objetivo era classificar corretamente 10 tipos de peças de roupa do dataset **Fashion-MNIST**, partindo de um modelo simples e aplicando técnicas sucessivas para melhorar seu desempenho e robustez.

## 1. O Ponto de Partida: Um Modelo Simples e o Problema do Overfitting

Começamos com uma arquitetura de CNN básica, composta por uma única camada de convolução para extração de características e uma camada densa para classificação.

**Arquitetura Inicial:**

* `Conv2D` (32 filtros)
* `MaxPooling2D`
* `Flatten`
* `Dense` (100 neurônios, ReLU)
* `Dense` (10 neurônios, Softmax)

### Diagnóstico: Overfitting Claro

Após o treinamento, a plotagem das curvas de acurácia e perda revelou um problema clássico: **overfitting**.

* **Acurácia:** A acurácia do treino continuou subindo, enquanto a da validação estagnou e começou a cair.
* **Perda (Erro):** O erro do treino diminuiu, mas o erro da validação começou a aumentar.

Isso indicava que o modelo estava "decorando" as imagens de treino em vez de aprender a generalizar as características que definem cada peça de roupa.

## 2. Iteração e Melhoria: Combatendo o Overfitting

Para resolver o problema, aplicamos uma série de técnicas, analisando o resultado a cada passo.

### Passo 1: Regularização com Dropout

* **Técnica:** Inserimos uma camada `Dropout(0.1)` após a camada `Flatten`. O dropout "desliga" aleatoriamente uma porcentagem de neurônios durante o treino, forçando a rede a aprender de forma mais distribuída e menos dependente de neurônios específicos.
* **Resultado:** O overfitting diminuiu significativamente. As curvas de treino e validação ficaram mais próximas, mostrando que o modelo estava generalizando melhor.

### Passo 2: Regularização de Pesos (L2)

* **Técnica:** Adicionamos um regularizador `L2` à camada `Dense` principal. Essa técnica adiciona uma penalidade à função de perda baseada no quadrado dos valores dos pesos da rede. Isso incentiva o modelo a manter seus pesos pequenos, resultando em um modelo mais simples e menos propenso a se ajustar ao ruído dos dados.
* **Resultado:** As curvas de validação se tornaram ainda mais estáveis e próximas das de treino. Embora a acurácia final tenha tido uma ligeira queda (de ~91% para ~89%), obtivemos um modelo comprovadamente mais robusto e confiável.

### Passo 3: Aprofundando a Rede (Hierarquia de Características)

* **Técnica:** Adicionamos um segundo bloco de `Conv2D -> MaxPooling2D`. A intuição é que a primeira camada aprende características simples (linhas, curvas), e a segunda aprende a combinar essas características em padrões mais complexos (bolsos, golas, texturas).
* **Resultado:** A acurácia voltou a subir (para ~90%) e o alinhamento das curvas melhorou. A rede mais profunda foi capaz de aprender padrões mais significativos.

### Passo 4: Refinamento da Arquitetura com Pooling Global

* **Técnica:** Substituímos o último `MaxPooling2D` e a camada `Flatten` por uma única camada `GlobalAveragePooling2D`. Em vez de "achatar" todos os pixels, essa camada calcula a média de cada mapa de características, gerando um vetor muito menor. Isso atua como uma forte forma de regularização, pois reduz drasticamente o número de parâmetros.
* **Resultado:** **O overfitting foi praticamente eliminado.** As curvas de treino e validação ficaram quase perfeitamente alinhadas. O resultado é um modelo altamente robusto que generaliza de forma excelente, mesmo que a acurácia final tenha se mantido em ~90%.

## 3. Avaliação Final: A Matriz de Confusão

Para uma análise qualitativa, plotamos a **Matriz de Confusão**. Ela nos permitiu ver não apenas *se* o modelo errou, mas *como* ele errou.

* **Diagnóstico:** A matriz mostrou que o modelo teve mais dificuldade em distinguir classes visualmente semelhantes, como "Camiseta" (T-shirt/top) vs. "Camisa" (Shirt) e "Pulôver" vs. "Casaco" (Coat), o que é um comportamento esperado e nos dá insights valiosos sobre suas limitações.

## Conclusão

Este projeto demonstrou um fluxo de trabalho completo para construir e refinar uma CNN. As principais lições foram:

1.  **A importância da visualização:** Gráficos de perda/acurácia são essenciais para diagnosticar problemas como o overfitting.
2.  **O poder da regularização:** Técnicas como `Dropout` e `Regularização L2` são ferramentas fundamentais para criar modelos que generalizam bem.
3.  **Arquitetura importa:** Aprofundar a rede e usar camadas modernas como `GlobalAveragePooling2D` pode levar a modelos mais robustos e eficientes.
4.  **Análise qualitativa:** A Matriz de Confusão oferece insights que uma simples métrica de acurácia não consegue fornecer.

O resultado final foi um modelo com ~90% de acurácia, mas, mais importante, um modelo robusto e com baixo overfitting, pronto para ser aplicado em novos dados com confiança.
```
