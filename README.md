# Cadeias de Markov aplicadas ao fluxo de pedidos de delivery

Trabalho prático da disciplina **SPOMVAL — Vetores, Geometria Analítica e Álgebra Linear**
(IFSP, Câmpus São Paulo, Bacharelado em Sistemas de Informação, 2026/2).

Professora: Josceli Maria Tenorio.

## Tema

Tema 1 do enunciado: previsão de comportamento de mercado com cadeias de Markov, aplicada ao
fluxo de um pedido em um aplicativo de delivery (carrinho, pagamento, preparo, trânsito,
entregue, cancelado).

## O que o programa faz

1. Lê uma matriz de transição quadrada de ordem *n* e um vetor de estado inicial (ou usa o
   exemplo de 6 estados do delivery).
2. Calcula a distribuição depois de *k* transições por multiplicação sucessiva de vetor por matriz.
3. Calcula a distribuição estacionária por dois métodos independentes: sistema linear associado
   ao autovalor 1, resolvido por eliminação de Gauss com pivoteamento parcial, e convergência
   iterativa de v = vP.
4. Gera o gráfico da evolução das probabilidades em PNG e imprime a interpretação do
   comportamento de longo prazo.

A multiplicação de vetores e matrizes usa NumPy, que é operação elementar. O que o trabalho
precisa demonstrar — a montagem de (Pᵀ − I)π = 0, o posto e a eliminação de Gauss com
pivotamento parcial — está escrito no próprio arquivo, sem `numpy.linalg.solve` e sem função
pronta de autovalores. O gráfico usa matplotlib.

## Requisitos

Python 3.8 ou superior, com NumPy e matplotlib:

```bash
pip install numpy matplotlib
```

## Execução

```bash
python3 markov.py            # menu interativo
python3 markov.py --exemplo  # caso do delivery (matriz 6x6 pronta)
python3 markov.py --testes   # dois casos de teste com verificação automática
```

Na opção 2 do menu, o programa pede a ordem da matriz, o nome dos estados, as linhas da matriz e
o vetor inicial, validando cada entrada antes de calcular. O gráfico é salvo em
`grafico_evolucao.png`.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `markov.py` | implementação completa, com funções separadas e comentadas |
| `relatorio-markov.pdf` | relatório do trabalho (teoria, aplicação, testes e análise) |
| `grafico_evolucao.png` | gráfico gerado pelo programa com matplotlib |

## Resultado principal

Com a matriz do modelo de delivery e todos os pedidos começando no carrinho, a distribuição
converge em 19 transições para **71,8626% entregues** e **28,1374% cancelados**. O mesmo valor é
obtido resolvendo o sistema linear da probabilidade de absorção, o que confirma o resultado.

## Testes

| Teste | Entrada | Saída esperada |
|---|---|---|
| 1 — previsão do tempo | P = [0,8 0,2; 0,3 0,7], v(0) = (1; 0) | estacionário (60,0000%; 40,0000%) |
| 2 — migração de marcas | P 3×3 com 90%/85%/90% de fidelidade, v(0) = (1; 0; 0) | estacionário (43,8596%; 31,5789%; 24,5614%) = (25/57; 18/57; 14/57) |

Os dois casos terminam com `Teste 1: OK` e `Teste 2: OK`.

## Grupo

- Leonardo Alves Guimarães — SP3199584
- Lincoln Arruda Carbonell Dominguez — SP319986X
- Gabriel Marques Ribeiro — SP3216365
