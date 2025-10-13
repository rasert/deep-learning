# Tradutor Português-Inglês
A ideia é construir um Transformer simplificado que aprende a traduzir frases curtas de um idioma para outro.

## Por que este é o exercício ideal para Transformers?

* **Demonstra a Arquitetura Completa:** Um tradutor usa a arquitetura completa de "Encoder-Decoder" do Transformer.

    - O **Encoder** "lê" e "entende" a frase inteira no idioma de origem (ex: Português).

    - O **Decoder** usa esse entendimento para gerar, palavra por palavra, a frase traduzida no idioma de destino (ex: Inglês).

* **Destaca o Poder da Atenção:** Você verá na prática como o mecanismo de atenção permite que o decoder, ao gerar uma palavra em inglês, "preste atenção" às palavras mais relevantes na frase original em português.

* **É um Problema do Mundo Real:** É uma aplicação direta e muito poderosa da tecnologia.

## O Desafio e o Aprendizado:

Construir um Transformer envolve implementar seus componentes-chave como camadas customizadas no Keras. Nosso objetivo será criar:

1. **A Camada de Positional Encoding:** Como o Transformer não tem recorrência (não processa palavra por palavra em ordem), precisamos de uma forma de "informar" ao modelo a posição de cada palavra na frase.

2. **A Camada de Multi-Head Attention:** O coração do Transformer. É o mecanismo que calcula a relevância de cada palavra em relação a todas as outras.

3. **O Bloco Encoder:** Um conjunto que combina a camada de atenção com uma rede feed-forward.

4. **O Bloco Decoder:** Similar ao encoder, mas com um mecanismo de atenção extra para "olhar" para a saída do encoder.

## Dataset Sugerido:

Para manter o projeto gerenciável, usaremos um dataset de pares de frases curtas Português-Inglês, como o disponível em `manythings.org/anki/`. Ele contém milhares de frases como "Eu te amo." -> "I love you.", que são ideais para treinar um modelo didático.

É um projeto ambicioso, mas incrivelmente recompensador. No final, você não apenas terá usado um Transformer, mas terá construído um.