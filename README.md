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
4. Gera o gráfico da evolução das probabilidades em SVG e imprime a interpretação do
   comportamento de longo prazo.

Toda a álgebra linear é implementada com laços e listas: o programa **não usa NumPy** nem
qualquer biblioteca de álgebra linear, conforme a exigência do enunciado.

## Requisitos

Python 3.8 ou superior. Nenhuma dependência externa.

## Execução

```bash
python3 markov.py            # menu interativo
python3 markov.py --exemplo  # caso do delivery (matriz 6x6 pronta)
python3 markov.py --testes   # dois casos de teste com verificação automática
```

Na opção 2 do menu, o programa pede a ordem da matriz, o nome dos estados, as linhas da matriz e
o vetor inicial, validando cada entrada antes de calcular. O gráfico é salvo em
`grafico_evolucao.svg`.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `markov.py` | implementação completa, com funções separadas e comentadas |
| `relatorio-markov.pdf` | relatório do trabalho (teoria, aplicação, testes e análise) |
| `grafico_evolucao.svg` | gráfico gerado pelo programa (criado na execução) |
| `grafico_evolucao.png` | versão em imagem, usada no relatório |

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
