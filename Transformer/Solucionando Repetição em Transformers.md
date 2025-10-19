

# **Do Paradoxo à Produção: Um Mergulho Profundo na Solução da Degeneração de Texto em Modelos Transformer**

## **Introdução**

Um dos paradoxos mais desconcertantes no desenvolvimento de modelos de linguagem generativos é o cenário em que as métricas de treinamento e validação indicam um sucesso retumbante, enquanto o desempenho em inferência resulta em um fracasso catastrófico. O problema apresentado — um modelo Transformer com arquitetura encoder-decoder que exibe curvas de aprendizado exemplares, com acurácia crescente e perda decrescente (Imagem 2), mas que, ao ser testado, produz sequências de palavras repetitivas e sem sentido (Imagem 1\) — encapsula perfeitamente este desafio. Este fenômeno, amplamente documentado na literatura de Processamento de Linguagem Natural (PLN), é conhecido como **degeneração de texto neural**.

A degeneração de texto neural manifesta-se como saídas que são insípidas, incoerentes e, mais notavelmente, presas em laços de repetição.1 Este não é um problema trivial ou raro; é uma falha observada até mesmo em modelos de última geração, como o GPT-2, especialmente quando são empregados métodos de decodificação mais simples ou ingênuos.4 As saídas como "if if if if if" ou "when when when when when" não são meros artefatos aleatórios, mas sim sintomas de problemas sistêmicos profundos que residem na interação entre o objetivo de treinamento do modelo, sua arquitetura, os dados com os quais ele aprende e o algoritmo usado para gerar texto.

O objetivo deste relatório é fornecer uma análise técnica abrangente e um guia prescritivo para diagnosticar e resolver este problema. A análise irá além das soluções superficiais, mergulhando nas causas fundamentais que criam a desconexão entre o desempenho de treinamento e a falha em inferência. A estrutura do relatório seguirá uma progressão lógica: primeiro, uma desconstrução do paradoxo para entender por que as métricas de treinamento podem ser enganosas; segundo, uma análise anatômica das múltiplas causas da repetição; terceiro, um conjunto detalhado de estratégias de mitigação aplicáveis durante a inferência; e, finalmente, intervenções mais fundamentais que podem ser implementadas durante o próprio treinamento do modelo para construir uma robustez intrínseca. Ao final, o leitor estará equipado com um entendimento profundo do problema e um roteiro claro para transformar um modelo paradoxal em um sistema de geração de texto robusto e confiável.

## **Seção 1: Desconstruindo o Paradoxo: Por Que Alta Acurácia Não Garante Geração Coerente**

A aparente contradição entre as métricas de treinamento positivas e a geração de texto de baixa qualidade reside em uma distinção fundamental: o que é medido durante o treinamento não é o mesmo que é executado durante a inferência. As curvas de aprendizado saudáveis, como as apresentadas na Imagem 2, são um reflexo do sucesso do modelo em uma tarefa muito específica, mas limitada.

### **A Ilusão da Acurácia**

A acurácia e a função de perda (loss), como a entropia cruzada (cross-entropy), comumente usadas no treinamento de modelos de linguagem, avaliam a capacidade do modelo de prever o *próximo token individual* em uma sequência, dado um *contexto de referência (ground-truth)*.3 Este paradigma de treinamento é conhecido como Estimação de Máxima Verossimilhança (Maximum Likelihood Estimation \- MLE). Essencialmente, o modelo é recompensado por atribuir a maior probabilidade possível ao token correto em cada passo, com base em um histórico perfeito e fornecido pelo humano. Os gráficos indicam que o modelo aprendeu a fazer isso com uma precisão de aproximadamente 80% no conjunto de validação.

No entanto, este objetivo de otimização local não avalia, nem pode avaliar, a coerência global, a fluidez ou a diversidade de uma sequência completa gerada pelo próprio modelo.8 A qualidade de uma frase ou parágrafo não é simplesmente a soma das probabilidades de suas palavras individuais; é uma propriedade emergente da sequência como um todo. Um modelo pode ser excelente em prever a próxima palavra mais provável em um contexto gramaticalmente perfeito, mas falhar miseravelmente quando o contexto é ligeiramente imperfeito — especialmente quando esse contexto é gerado por ele mesmo.

### **A Natureza Cumulativa da Geração Autorregressiva**

O processo de geração de texto em modelos como o Transformer é **autorregressivo**. Isso significa que a saída do passo de tempo $t$ (a palavra ou token gerado) torna-se parte da entrada para o passo de tempo $t+1$.10 Este mecanismo cria um sistema de feedback fechado, onde o modelo depende de suas próprias previsões anteriores para construir o futuro.

Este processo sequencial é inerentemente frágil e propenso ao que é conhecido como **acumulação de erros**.13 Um único token subótimo gerado no início de uma sequência pode corromper todo o contexto subsequente. Isso empurra o modelo para fora da distribuição de dados na qual foi treinado, forçando-o a fazer previsões com base em sequências que nunca viu antes. Como o modelo não foi treinado para se recuperar de seus próprios erros, um pequeno desvio pode iniciar uma cascata de previsões cada vez mais degradadas, culminando em um colapso total da coerência e, frequentemente, em um estado de repetição de baixa energia.

A falha, portanto, não é um problema de previsão estática, mas sim um problema de sistema dinâmico. A alta acurácia por token do modelo é medida em um estado de "malha aberta" (dado um contexto perfeito e fixo). A inferência, por outro lado, opera em um estado de "malha fechada", onde as saídas do modelo criam um feedback que pode levar o sistema a estados degenerados e estáveis, como os laços de repetição. A alta acurácia de treinamento é uma condição necessária, mas flagrantemente insuficiente, para uma boa geração de texto.

## **Seção 2: A Anatomia da Repetição: Descobrindo as Causas Raiz**

A degeneração de texto e os laços de repetição não surgem de uma única falha, mas de uma confluência de fatores inter-relacionados. Uma análise aprofundada revela quatro causas principais: as limitações dos algoritmos de decodificação, um mecanismo de auto-reforço catastrófico, a discrepância entre treinamento e inferência, e os padrões inerentes nos próprios dados de treinamento.

### **2.1 As Armadilhas da Decodificação Baseada em Maximização**

As estratégias de decodificação mais intuitivas, que tentam encontrar a sequência de texto mais provável de acordo com o modelo, são frequentemente as principais culpadas pela degeneração.

* **Busca Gulosa (Greedy Search):** Este é o método de decodificação mais simples e, muitas vezes, o padrão em implementações. Em cada passo, ele seleciona deterministicamente o único token com a maior probabilidade.11 Essa abordagem míope, que otimiza a escolha localmente em cada passo, é extremamente propensa a ficar presa em laços de palavras ou frases de alta frequência. Palavras comuns como "if", "when", "the" podem facilmente ter a maior probabilidade em muitos contextos, levando a saídas repetitivas como as vistas na Imagem 1\.4 A falta de visão de futuro impede que a busca gulosa escolha um token inicial ligeiramente menos provável que poderia levar a uma sequência geral muito melhor.  
* **Busca por Feixe (Beam Search):** Uma melhoria em relação à busca gulosa, a busca por feixe mantém um conjunto de $k$ hipóteses (os "feixes") mais prováveis em cada passo, explorando múltiplos caminhos simultaneamente.6 No entanto, ela ainda otimiza fundamentalmente a probabilidade geral da sequência. A pesquisa demonstra que, mesmo com feixes largos, a busca por feixe tende a produzir texto que é insípido, repetitivo e carente de diversidade, especialmente em tarefas de geração de final aberto.3 A tendência de favorecer sequências seguras e de alta probabilidade muitas vezes a leva a convergir para as mesmas armadilhas repetitivas da busca gulosa. Além disso, os detalhes de implementação da busca por feixe podem impactar drasticamente seu desempenho, adicionando outra camada de complexidade.19

### **2.2 A Catástrofe do Auto-Reforço**

Uma vez que uma repetição começa, a própria arquitetura e os padrões aprendidos pelo modelo podem amplificá-la, criando um ciclo vicioso.

* **O Loop de Feedback Positivo:** A probabilidade de gerar um token ou frase repetitiva *aumenta* a cada repetição já presente no contexto.4 O modelo aprende um atalho: se o contexto é repetitivo, copiar é uma ação de alta probabilidade.5 Cada vez que o modelo gera a palavra "if", por exemplo, o contexto para a próxima previsão contém mais uma instância de "if", tornando a probabilidade de gerar "if" novamente ainda maior.  
* **O "Problema do Alto Influxo" (High Inflow Problem):** Este conceito teórico oferece uma explicação mais profunda. A repetição é, em parte, uma propriedade da própria linguagem. Existem muitas palavras diferentes que podem prever a mesma palavra subsequente (uma palavra de "alto influxo") com alta probabilidade. Isso torna fácil para o modelo "cair" em um laço em torno dessa palavra de alto influxo, da qual é difícil escapar.2  
* **O Papel do Mecanismo de Atenção:** O mecanismo de auto-atenção, a base da arquitetura Transformer, relaciona todos os tokens de entrada entre si. Quando a sequência de entrada contém repetições, os escores de atenção podem reforçar esses padrões, criando um sinal forte para o decodificador continuar o laço.10

### **2.3 A Discrepância Treinamento-Inferência (Viés de Exposição)**

Uma das causas mais fundamentais da fragilidade do modelo durante a inferência é uma discrepância no modo como ele é treinado versus como ele é usado.

* **Forçamento do Professor (Teacher Forcing):** Durante o treinamento, para estabilizar e acelerar o processo, os modelos autorregressivos são tipicamente treinados usando uma técnica chamada "teacher forcing". Nela, para prever o token no passo $t+1$, o modelo sempre recebe o token de referência correto do passo $t$ como entrada, independentemente de sua própria previsão anterior.7  
* **O Viés (Exposure Bias):** Isso cria uma discrepância crítica: durante o treinamento, o modelo é exposto *apenas* a sequências perfeitas, geradas por humanos. Ele nunca aprende com seus próprios erros. Na inferência, no entanto, ele deve gerar texto condicionado às suas próprias saídas, que podem ser imperfeitas.7  
* **Consequências:** Como o modelo nunca foi treinado para se recuperar de seus próprios erros, a acumulação de erros durante a inferência torna-se inevitável.13 Um pequeno erro pode levar o modelo a um estado (uma sequência de contexto) que tem probabilidade zero sob a distribuição de dados de treinamento, tornando suas previsões subsequentes altamente não confiáveis e propensas ao colapso em modos degenerados como a repetição.

### **2.4 O Eco dos Dados**

Finalmente, a causa da degeneração de texto não é puramente algorítmica ou arquitetural; é também um problema de dados. O modelo Transformer é um motor de correspondência de padrões extremamente sofisticado. Ele aprende a imitar as propriedades estatísticas do corpus de treinamento.

Pesquisas mostram uma forte correlação entre a presença de repetições nos dados de treinamento e a tendência do modelo de gerar texto repetitivo.1 Isso implica que o modelo não está "inventando" o comportamento de laço do nada; ele está aprendendo que a repetição é uma característica válida e até mesmo desejável da linguagem que ele deve emular. Se um corpus de treinamento contém frases repetidas, clichês ou outras formas de redundância, o modelo irá internalizar esses padrões. Portanto, uma causa crítica e muitas vezes negligenciada da degeneração é o próprio corpus de treinamento. Isso reformula o problema de apenas "consertar a saída do modelo" para também "consertar a entrada do modelo", o que representa uma abordagem mais fundamental e potencialmente mais eficaz.

## **Seção 3: Um Conjunto de Ferramentas para Geração Coerente: Estratégias em Tempo de Inferência**

Para combater diretamente a degeneração de texto no momento da geração, uma série de estratégias de decodificação sofisticadas foram desenvolvidas. Essas técnicas substituem os métodos determinísticos e propensos a falhas por abordagens estocásticas e baseadas em penalidades que promovem a diversidade e a coerência.

### **3.1 Indo Além do Determinismo: O Poder da Amostragem**

A alternativa primária aos métodos determinísticos como a busca gulosa é a decodificação estocástica, ou amostragem.14 Em vez de sempre escolher a palavra mais provável, a amostragem introduz aleatoriedade, permitindo que o modelo explore uma gama mais ampla de possibilidades.

* **Escalonamento de Temperatura (Temperature Scaling):** A temperatura ($T$) é um parâmetro que reescala os logits (as saídas brutas do modelo antes da função softmax) para controlar a aleatoriedade da amostragem.  
  * Um valor de $T \> 1$ "achata" a distribuição de probabilidade, tornando as palavras menos prováveis mais prováveis de serem escolhidas. Isso aumenta a diversidade e a criatividade, mas pode levar à incoerência.  
  * Um valor de $T \< 1$ "aguça" a distribuição, aumentando a probabilidade das palavras mais prováveis. Isso torna a saída mais focada e determinística, semelhante à busca gulosa.  
  * T=1 corresponde à amostragem padrão da distribuição original do modelo.  
    A temperatura é um controle crucial para ajustar o equilíbrio entre coerência e criatividade.12  
* **Amostragem Top-K (Top-K Sampling):** Em vez de considerar todo o vocabulário, este método trunca a distribuição de probabilidade para incluir apenas os $K$ tokens mais prováveis. A massa de probabilidade é então redistribuída entre esses $K$ tokens antes que a amostragem seja realizada.12  
  * **Prós:** É uma técnica simples, rápida e eficaz para evitar a amostragem da "cauda não confiável" da distribuição, que consiste em dezenas de milhares de tokens com probabilidades muito baixas que podem levar a resultados sem sentido.8  
  * **Contras:** O valor fixo de $K$ é inflexível. Para distribuições de probabilidade "pontudas" (onde o modelo está muito confiante), um $K$ grande pode incluir tokens ruins. Para distribuições "chatas" (onde o modelo está incerto), um $K$ pequeno pode cortar opções criativas e viáveis.28  
* **Amostragem por Núcleo (Nucleus Sampling ou Top-p):** Esta é uma abordagem mais adaptativa. Em vez de um número fixo de tokens, ela seleciona o menor conjunto de tokens cuja probabilidade cumulativa excede um limiar $p$ (por exemplo, $p=0.92$). A amostragem é então realizada a partir deste "núcleo" de tokens.8  
  * **Prós:** O tamanho do conjunto de amostragem se ajusta dinamicamente com base na confiança do modelo. Se o modelo está confiante, o núcleo será pequeno; se está incerto, o núcleo será maior. Isso a torna mais robusta e geralmente superior à amostragem Top-K.8  
  * **Contras:** É ligeiramente mais intensiva computacionalmente do que a Top-K, pois requer ordenação e uma operação de soma cumulativa.18

### **3.2 Impondo Diversidade: Mecanismos Baseados em Penalidade**

Esses são métodos heurísticos que podem ser aplicados sobre outras estratégias de decodificação para desencorajar explicitamente a repetição.

* **Penalidade de Repetição (Repetition Penalty):** Este parâmetro (tipicamente um valor $\> 1.0$) reduz dinamicamente os escores de logit dos tokens que já apareceram no contexto gerado. Isso torna menos provável que eles sejam amostrados novamente, forçando o modelo a escolher palavras novas.26  
* **Bloqueio de N-gramas (N-gram Blocking):** Esta é uma restrição rígida que impede que qualquer n-grama (uma sequência de $n$ palavras) seja repetido. Isso é feito definindo a probabilidade de qualquer token que completaria um n-grama já visto como zero.28 É uma maneira muito direta e eficaz de quebrar laços de repetição de frases curtas.

### **3.3 Análise Comparativa e Recomendações Práticas**

A escolha da estratégia de decodificação correta depende da tarefa. Para geração criativa e de final aberto (como chatbots ou escrita de histórias), uma combinação de Amostragem por Núcleo (Top-p) com uma temperatura moderada (por exemplo, $T$ entre 0.7 e 1.0) e uma leve penalidade de repetição (por exemplo, 1.1) é frequentemente o estado da arte. Para tarefas que exigem maior precisão factual, como tradução ou sumarização, uma temperatura mais baixa ou até mesmo a Busca por Feixe com bloqueio de n-gramas pode ser preferível.

A tabela a seguir resume e compara as principais estratégias de decodificação.

| Estratégia | Mecanismo | Custo Computacional | Diversidade da Saída | Risco de Repetição | Casos de Uso Primários |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Busca Gulosa** | Seleciona o token de maior probabilidade a cada passo. | Muito Baixo | Muito Baixa | Muito Alto | Saídas curtas e factuais; depuração. |
| **Busca por Feixe** | Mantém as $k$ melhores hipóteses a cada passo. | Médio | Baixa | Alto | Tradução, sumarização (tarefas direcionadas). |
| **Amostragem Top-K** | Amostra a partir dos $K$ tokens mais prováveis. | Baixo | Média | Médio-Baixo | Geração criativa controlada. |
| **Amostragem por Núcleo (Top-p)** | Amostra do menor conjunto de tokens com probabilidade cumulativa \> $p$. | Baixo-Médio | Alta | Baixo | Geração de final aberto, criativa; diálogo. |

## **Seção 4: Construindo Modelos Robustos: Intervenções em Tempo de Treinamento**

Embora as estratégias de inferência sejam cruciais para gerenciar a saída de um modelo, as soluções mais fundamentais envolvem aprimorar o próprio modelo durante o treinamento. Essas intervenções visam criar modelos inerentemente mais robustos, menos propensos à degeneração e mais bem alinhados com a tarefa de geração de sequências coerentes.

### **4.1 Mitigando o Viés de Exposição com Amostragem Programada**

Para resolver diretamente a discrepância entre treinamento e inferência causada pelo "teacher forcing", pode-se usar a amostragem programada (scheduled sampling).

* **Mecanismo:** É uma estratégia de aprendizado curricular que gradualmente faz a ponte entre o "teacher forcing" e a geração em inferência. Durante o treinamento, com uma certa probabilidade, o modelo é alimentado com sua *própria* previsão anterior em vez do token de referência.20  
* **Implementação:** A probabilidade de usar o token de referência geralmente decai ao longo do treinamento. No início, o modelo depende quase inteiramente do "professor", mas, à medida que se torna mais competente, é gradualmente "desmamado" e forçado a lidar com suas próprias previsões e erros potenciais.20  
* **Ressalvas:** Embora empiricamente bem-sucedida, a amostragem programada pode introduzir um objetivo de treinamento inconsistente. Além disso, pode piorar o desempenho se o prefixo gerado pelo modelo estiver correto, um fenômeno conhecido como "esquecimento catastrófico", onde o modelo se torna excessivamente dependente da sequência de entrada em detrimento do contexto gerado.23

### **4.2 Regularização e Funções de Perda Avançadas**

Modificar a forma como o modelo aprende, por meio de regularização e funções de perda mais inteligentes, pode prevenir algumas das causas raiz da degeneração.

* **Suavização de Rótulos (Label Smoothing):** Esta é uma técnica de regularização onde os rótulos de referência "rígidos" (vetores one-hot, com 1 para a classe correta e 0 para as outras) são suavizados para uma distribuição mais "suave" (por exemplo, 0.9 para a classe correta e 0.1 distribuído entre as outras).33  
  * O treinamento padrão com entropia cruzada incentiva o modelo a empurrar a probabilidade do token correto para 1 e de todos os outros para 0, levando à **superconfiança**.34 Um modelo superconfiante produz distribuições de probabilidade muito "pontudas", tornando-o frágil e altamente suscetível aos laços de repetição da decodificação baseada em maximização. A suavização de rótulos penaliza essa superconfiança, forçando o modelo a reservar uma pequena massa de probabilidade para outros tokens.33 Isso resulta em uma distribuição de saída mais "suave" e menos pontuda, o que torna o modelo mais robusto e dá aos decodificadores baseados em amostragem mais opções viáveis para escolher, reduzindo indireta mas eficazmente o risco de degeneração.  
* **Treinamento por Inverossimilhança (Unlikelihood Training):** Esta técnica poderosa modifica a função de perda para não apenas maximizar a verossimilhança dos tokens corretos, mas também para *minimizar* ativamente a verossimilhança de tokens indesejáveis. Por exemplo, pode-se treinar o modelo para diminuir a probabilidade de gerar tokens que formariam uma repetição.3

### **4.3 Abordagens Centradas em Dados**

Com base na constatação de que os dados de treinamento são uma fonte significativa de repetição, as seguintes técnicas abordam o problema em sua origem.

* **Dropout de Repetição (Repetition Dropout):** Durante o treinamento, este método identifica n-gramas repetitivos nos dados de entrada e, seletivamente, "desliga" (faz dropout) a atenção do modelo para eles. Isso ensina diretamente o modelo a depender menos de pistas repetitivas no contexto para fazer suas previsões.1  
* **DITTO (PseuDo-RepetITion PenalizaTiOn):** Este é um método de treinamento onde o modelo é ajustado (fine-tuned) em dados repetitivos gerados sinteticamente. Ao fazer isso, o modelo aprende a penalizar as probabilidades de repetições em nível de sentença, tornando-se mais resistente a cair nesses laços durante a inferência.4

## **Conclusão: Uma Síntese de Estratégias para Geração de Texto Robusta**

A degeneração de texto neural, manifestada por laços de repetição e saídas incoerentes, não é um problema com uma única causa ou solução. É uma propriedade emergente da complexa interação entre o objetivo de treinamento (MLE), o processo de treinamento (teacher forcing), os dados de treinamento (repetições inerentes), a arquitetura do modelo (feedback de atenção) e o algoritmo de inferência (estratégia de decodificação). A discrepância entre métricas de treinamento saudáveis e um desempenho de inferência pobre é a prova de que otimizar para a previsão do próximo token é fundamentalmente diferente de otimizar para a geração de sequências coerentes e de alta qualidade.

A solução mais robusta e eficaz, portanto, não é uma única técnica, mas uma abordagem holística que combina intervenções em tempo de treinamento com estratégias sofisticadas em tempo de inferência. Um sistema de geração de texto de nível de produção deve ser construído sobre uma base sólida: um modelo treinado com técnicas que promovem a robustez, como a suavização de rótulos e abordagens centradas em dados para mitigar a influência de repetições no corpus. Em seguida, na implantação, esse modelo robusto deve ser guiado por estratégias de decodificação bem ajustadas, como a Amostragem por Núcleo (Top-p) combinada com penalidades de repetição, para garantir uma saída diversificada e coerente.

A tabela a seguir fornece um mapa conciso das causas raiz do problema para as soluções primárias, servindo como um guia prático para diagnóstico e correção.

| Causa Raiz | Descrição | Solução(ões) Primária(s) em Tempo de Inferência | Solução(ões) Primária(s) em Tempo de Treinamento |
| :---- | :---- | :---- | :---- |
| **Decodificação Baseada em Maximização** | Escolhas determinísticas e míopes levam a laços de alta frequência. | Amostragem por Núcleo (Top-p) / Top-K, Escalonamento de Temperatura. | N/A (Este é um problema de inferência). |
| **Loop de Auto-Reforço** | Repetições no contexto aumentam a probabilidade de mais repetições. | Penalidade de Repetição, Bloqueio de N-gramas. | Treinamento por Inverossimilhança, DITTO. |
| **Viés de Exposição (Exposure Bias)** | O modelo não é treinado para lidar com seus próprios erros, levando à acumulação de erros. | (Indiretamente) Métodos de amostragem que exploram caminhos diversos. | Amostragem Programada (Scheduled Sampling). |
| **Superconfiança do Modelo** | Distribuições de probabilidade pontudas tornam o modelo frágil e propenso a laços. | Escalonamento de Temperatura ($T \> 1.0$). | Suavização de Rótulos (Label Smoothing). |
| **Dados de Treinamento Repetitivos** | O modelo aprende a imitar as repetições presentes no corpus de treinamento. | N/A (Não pode ser corrigido na inferência). | Limpeza de Dados, Dropout de Repetição. |

#### **Referências citadas**

1. Repetition In Repetition Out: Towards Understanding Neural Text Degeneration from the Data Perspective \- arXiv, acessado em outubro 18, 2025, [https://arxiv.org/html/2310.10226](https://arxiv.org/html/2310.10226)  
2. A Theoretical Analysis of the Repetition Problem in Text ... \- arXiv, acessado em outubro 18, 2025, [https://arxiv.org/pdf/2012.14660](https://arxiv.org/pdf/2012.14660)  
3. neural text degeneration with unlikelihood training \- arXiv, acessado em outubro 18, 2025, [https://arxiv.org/pdf/1908.04319](https://arxiv.org/pdf/1908.04319)  
4. Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation, acessado em outubro 18, 2025, [https://machinelearning.apple.com/research/analyzing-mitigating-repetitions](https://machinelearning.apple.com/research/analyzing-mitigating-repetitions)  
5. Learning to Break the Loop: Analyzing and Mitigating ... \- OpenReview, acessado em outubro 18, 2025, [https://openreview.net/pdf?id=sexfswCc7B](https://openreview.net/pdf?id=sexfswCc7B)  
6. What is Neural Text Degeneration? \- Dasha.AI, acessado em outubro 18, 2025, [https://dasha.ai/blog/neural-text-degeneration](https://dasha.ai/blog/neural-text-degeneration)  
7. QUANTIFYING EXPOSURE BIAS FOR NEURAL ... \- OpenReview, acessado em outubro 18, 2025, [https://openreview.net/pdf?id=rJg2fTNtwr](https://openreview.net/pdf?id=rJg2fTNtwr)  
8. THE CURIOUS CASE OF NEURAL TEXT DeGENERATION \- OpenReview, acessado em outubro 18, 2025, [https://openreview.net/pdf?id=rygGQyrFvH](https://openreview.net/pdf?id=rygGQyrFvH)  
9. The Curious Case of Neural Text Degeneration \- Hyunyoung2, acessado em outubro 18, 2025, [https://hyunyoung2.github.io/2020/06/04/The\_Curious\_Case\_of\_Neural\_Text\_Degeneration/](https://hyunyoung2.github.io/2020/06/04/The_Curious_Case_of_Neural_Text_Degeneration/)  
10. Understanding the Modern LLM — Part 5: Understanding Text Degeneration During Decoding and Methods to Combat Degeneration. | by Inkyu Kim | Medium, acessado em outubro 18, 2025, [https://medium.com/@ikim1994914/understanding-the-modern-llm-part-5-understanding-text-degeneration-during-decoding-and-methods-966a4d33e9c8](https://medium.com/@ikim1994914/understanding-the-modern-llm-part-5-understanding-text-degeneration-during-decoding-and-methods-966a4d33e9c8)  
11. Transformers From Scratch: Part 9 — Inference & Greedy Generation \- Medium, acessado em outubro 18, 2025, [https://medium.com/@kavierim/transformers-from-scratch-part-9-inference-greedy-generation-9436681abb2e](https://medium.com/@kavierim/transformers-from-scratch-part-9-inference-greedy-generation-9436681abb2e)  
12. General Understanding of Decoding Strategies Commonly Used in Text Generation, acessado em outubro 18, 2025, [https://blog.gopenai.com/general-understanding-of-decoding-strategies-commonly-used-in-text-generation-512128bacfeb](https://blog.gopenai.com/general-understanding-of-decoding-strategies-commonly-used-in-text-generation-512128bacfeb)  
13. Why Exposure Bias Matters: An Imitation Learning Perspective of Error Accumulation in Language Generation | Request PDF \- ResearchGate, acessado em outubro 18, 2025, [https://www.researchgate.net/publication/361063299\_Why\_Exposure\_Bias\_Matters\_An\_Imitation\_Learning\_Perspective\_of\_Error\_Accumulation\_in\_Language\_Generation](https://www.researchgate.net/publication/361063299_Why_Exposure_Bias_Matters_An_Imitation_Learning_Perspective_of_Error_Accumulation_in_Language_Generation)  
14. Generation strategies \- Hugging Face, acessado em outubro 18, 2025, [https://huggingface.co/docs/transformers/en/generation\_strategies](https://huggingface.co/docs/transformers/en/generation_strategies)  
15. Greedy Decoding \- Aussie AI, acessado em outubro 18, 2025, [https://www.aussieai.com/research/greedy-decoding](https://www.aussieai.com/research/greedy-decoding)  
16. Text generation strategies \- Hugging Face, acessado em outubro 18, 2025, [https://huggingface.co/docs/transformers/v4.27.0/generation\_strategies](https://huggingface.co/docs/transformers/v4.27.0/generation_strategies)  
17. Decoding Strategies for Transformers \- Scaler Topics, acessado em outubro 18, 2025, [https://www.scaler.com/topics/nlp/decoding-strategies-for-transformers/](https://www.scaler.com/topics/nlp/decoding-strategies-for-transformers/)  
18. \[D\] What happened to "creative" decoding strategy? : r/MachineLearning \- Reddit, acessado em outubro 18, 2025, [https://www.reddit.com/r/MachineLearning/comments/1e42das/d\_what\_happened\_to\_creative\_decoding\_strategy/](https://www.reddit.com/r/MachineLearning/comments/1e42das/d_what_happened_to_creative_decoding_strategy/)  
19. A Call for Clarity in Beam Search: How It Works and When It Stops \- arXiv, acessado em outubro 18, 2025, [https://arxiv.org/html/2204.05424v3](https://arxiv.org/html/2204.05424v3)  
20. What's the difference between teacher forcing and scheduled ..., acessado em outubro 18, 2025, [https://medium.com/@sharetonschool/whats-the-difference-between-teacher-forcing-and-scheduled-sampling-in-sequence-to-sequence-models-aa3e313fbf8a](https://medium.com/@sharetonschool/whats-the-difference-between-teacher-forcing-and-scheduled-sampling-in-sequence-to-sequence-models-aa3e313fbf8a)  
21. Generalization in Generation: A closer look at Exposure Bias \- ACL Anthology, acessado em outubro 18, 2025, [https://aclanthology.org/D19-5616.pdf](https://aclanthology.org/D19-5616.pdf)  
22. \[1910.11235\] Rethinking Exposure Bias In Language Modeling \- arXiv, acessado em outubro 18, 2025, [https://arxiv.org/abs/1910.11235](https://arxiv.org/abs/1910.11235)  
23. Improving Scheduled Sampling with Elastic Weight Consolidation for Neural Machine Translation \- ACL Anthology, acessado em outubro 18, 2025, [https://aclanthology.org/2022.findings-emnlp.536/](https://aclanthology.org/2022.findings-emnlp.536/)  
24. Dynamic Scheduled Sampling with Imitation Loss for Neural Text Generation \- arXiv, acessado em outubro 18, 2025, [https://arxiv.org/abs/2301.13753](https://arxiv.org/abs/2301.13753)  
25. Generation strategies \- Hugging Face, acessado em outubro 18, 2025, [https://huggingface.co/docs/transformers/generation\_strategies](https://huggingface.co/docs/transformers/generation_strategies)  
26. Foundation model parameters: decoding and stopping criteria \- IBM, acessado em outubro 18, 2025, [https://www.ibm.com/docs/en/watsonx/saas?topic=prompts-model-parameters-prompting](https://www.ibm.com/docs/en/watsonx/saas?topic=prompts-model-parameters-prompting)  
27. What is the paper that introduced the settings used in transformers? \- AI Stack Exchange, acessado em outubro 18, 2025, [https://ai.stackexchange.com/questions/47822/what-is-the-paper-that-introduced-the-settings-used-in-transformers](https://ai.stackexchange.com/questions/47822/what-is-the-paper-that-introduced-the-settings-used-in-transformers)  
28. How to generate text: using different decoding methods for language ..., acessado em outubro 18, 2025, [https://huggingface.co/blog/how-to-generate](https://huggingface.co/blog/how-to-generate)  
29. Worth reading: The Curious Case of Neural Text Degeneration. : r/MachineLearning \- Reddit, acessado em outubro 18, 2025, [https://www.reddit.com/r/MachineLearning/comments/bowknb/worth\_reading\_the\_curious\_case\_of\_neural\_text/](https://www.reddit.com/r/MachineLearning/comments/bowknb/worth_reading_the_curious_case_of_neural_text/)  
30. \[1906.07651\] Scheduled Sampling for Transformers \- arXiv, acessado em outubro 18, 2025, [https://arxiv.org/abs/1906.07651](https://arxiv.org/abs/1906.07651)  
31. Scheduled Sampling for Transformers \- ACL Anthology, acessado em outubro 18, 2025, [https://aclanthology.org/P19-2049/](https://aclanthology.org/P19-2049/)  
32. DYNAMIC SCHEDULED SAMPLING WITH IMITATION LOSS FOR NEURAL TEXT GENERATION \- OpenReview, acessado em outubro 18, 2025, [https://openreview.net/pdf?id=UmHG2bD7X3w](https://openreview.net/pdf?id=UmHG2bD7X3w)  
33. Label Smoothing for Enhanced Text Sentiment Classification \- arXiv, acessado em outubro 18, 2025, [https://arxiv.org/html/2312.06522v1](https://arxiv.org/html/2312.06522v1)  
34. From Label Smoothing to Label Relaxation \- ResearchGate, acessado em outubro 18, 2025, [https://www.researchgate.net/publication/363387611\_From\_Label\_Smoothing\_to\_Label\_Relaxation](https://www.researchgate.net/publication/363387611_From_Label_Smoothing_to_Label_Relaxation)  
35. Decoding the Transformer Model: Architecture, Loss Function, and ..., acessado em outubro 18, 2025, [https://medium.com/@praveenkumar2909/decoding-the-transformer-model-architecture-loss-function-and-inference-from-the-attention-is-717b98d183b3](https://medium.com/@praveenkumar2909/decoding-the-transformer-model-architecture-loss-function-and-inference-from-the-attention-is-717b98d183b3)  
36. Label Smoothing for Text Mining \- ACL Anthology, acessado em outubro 18, 2025, [https://aclanthology.org/2022.coling-1.193.pdf](https://aclanthology.org/2022.coling-1.193.pdf)  
37. Adaptive Label Smoothing with Self-Knowledge in Natural Language Generation \- ACL Anthology, acessado em outubro 18, 2025, [https://aclanthology.org/2022.emnlp-main.664.pdf](https://aclanthology.org/2022.emnlp-main.664.pdf)