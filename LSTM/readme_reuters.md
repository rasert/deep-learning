# Classificação de Notícias Usando LSTM

## 1. Objetivo do Projeto

O objetivo deste projeto foi desenvolver e otimizar um modelo de Deep Learning para a tarefa de **classificação de texto multiclasse**. Utilizamos o dataset clássico **Reuters**, que consiste em aproximadamente 11.000 notícias curtas de agências de notícias, distribuídas em **46 tópicos distintos**.

A arquitetura de rede neural escolhida para esta tarefa foi a **LSTM (Long Short-Term Memory)**, devido à sua capacidade de processar dados sequenciais e capturar o contexto em textos.

## 2. Desafios e Metodologia de Otimização

O desenvolvimento de um modelo eficaz não é um processo linear. Durante este projeto, enfrentamos desafios clássicos de treinamento de redes neurais. A seguir, detalho a jornada iterativa para encontrar o modelo ideal, começando pelas abordagens mais simples.

### Passo 1: Modelos Iniciais e a Descoberta da Complexidade

Nossa exploração começou com modelos básicos para estabelecer uma linha de base de performance.

- **Abordagem 1: LSTM Unidirecional Simples**
  - **Arquitetura:** Uma única camada `LSTM` seguida por um classificador denso.
  - **Resultado:** Acurácia de teste de **~60%**.
  - **Problema Identificado: Gargalo de Informação e Falta de Contexto**
    - A LSTM unidirecional processa o texto apenas em uma direção (do início ao fim).
    - O modelo utilizava apenas a saída do último passo da sequência, criando um "gargalo" que perdia informações importantes do início e do meio do texto.
    - O baixo desempenho motivou a busca por arquiteturas mais robustas que pudessem capturar melhor o contexto.

### Passo 2: O Modelo de Alta Capacidade e o Problema do Overfitting

Inspirados por exemplos de alta performance, implementamos um modelo significativamente mais complexo.

- **Arquitetura:**
  - 2 camadas `Bidirectional(LSTM)` empilhadas, para capturar o contexto em ambas as direções.
  - Uma camada `GlobalMaxPooling1D` para extrair as features mais relevantes de toda a sequência.
  - Um classificador denso (`Dense`) grande com 3 camadas (1024, 512, 256 neurônios).
- **Resultado:** Atingimos uma acurácia de teste de **~76%**.
- **Problema Identificado: Overfitting Severo**
  - O modelo, por ser muito complexo para o tamanho do dataset, começou a "decorar" os dados de treino. Isso ficou evidente nos gráficos de treinamento: a perda de treino caía continuamente, enquanto a perda de validação começava a **aumentar** após algumas épocas.

### Passo 3: Primeira Tentativa de Correção - Regularização Agressiva

Para combater o overfitting, a hipótese foi aplicar "freios" fortes no modelo.

- **Ajustes:**
  - Mantivemos a arquitetura complexa.
  - Aumentamos o `Dropout` para `0.5` e reduzimos a taxa de aprendizado (`learning_rate`).
  - Implementamos `EarlyStopping` para parar o treino quando a performance de validação parasse de melhorar.
- **Resultado:** A acurácia caiu para **~68%**.
- **Problema Identificado: Underfitting**
  - As técnicas de regularização foram tão fortes que "sufocaram" o modelo, impedindo-o de aprender os padrões necessários.

### Passo 4: Simplificando a Arquitetura - Em Busca do Equilíbrio

A próxima hipótese foi que a arquitetura ainda era complexa demais para ser regularizada eficazmente. Simplificamos o modelo.

- **Arquitetura Simplificada:**
  - Apenas **1 camada** `Bidirectional(LSTM)`.
  - Um classificador denso menor com 2 camadas (512, 256 neurônios).
  - `Dropout` moderado (`0.4`).
- **Resultado:** Atingimos uma acurácia estável de **~72%**.
- **Análise:** Este foi um grande avanço. O modelo se mostrou muito mais estável, com um bom equilíbrio entre aprendizado e generalização.

### Passo 5: O Modelo Final - O Melhor dos Dois Mundos

Com base nos aprendizados, a estratégia final foi refinar a arquitetura simplificada com uma combinação de técnicas de regularização.

- **Arquitetura Final (O Ponto Ideal):**
  - **1 camada `Bidirectional(LSTM)`** com `dropout` interno de `0.5`.
  - Um classificador denso médio (512, 256 neurônios).
  - `Dropout` forte (`0.5`) entre as camadas densas.
  - **Regularização L2** (`kernel_regularizer`) com um valor pequeno (`0.0001`) em todas as camadas principais para penalizar pesos muito grandes.
  - **`EarlyStopping`** com `patience=3` para capturar o modelo em seu pico de performance.

- **Resultado Final:** Atingimos uma acurácia de teste de **~74%**.
  - O modelo demonstrou o melhor equilíbrio: a perda de validação se manteve estável por mais tempo e a diferença entre as métricas de treino e validação foi a menor entre os modelos de alta performance.

## 3. Conclusão e Próximos Passos

Este projeto demonstrou na prática o processo iterativo de otimização de uma rede neural, partindo de um modelo simples com performance limitada, passando por um modelo complexo com overfitting, até chegar a uma arquitetura enxuta e bem regularizada. A principal lição foi que **encontrar o equilíbrio entre a complexidade do modelo e as técnicas de regularização é a chave para o sucesso**.

O modelo final, com sua arquitetura `Bi-LSTM` única e regularização combinada, provou ser o mais robusto e eficaz, alcançando uma acurácia confiável de **74%**.