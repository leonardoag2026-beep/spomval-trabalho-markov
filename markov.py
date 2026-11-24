#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cadeias de Markov: previsão de comportamento de mercado.

Trabalho prático da disciplina SPOMVAL - Vetores, Geometria Analítica e
Álgebra Linear (IFSP, 2026/2).

A multiplicação de vetores e matrizes usa NumPy; o cálculo do vetor
estacionário (eliminação de Gauss com pivoteamento parcial) e o posto da
matriz são escritos no próprio programa, sem solver pronto. O gráfico usa
matplotlib.

Uso:
    python3 markov.py            # menu interativo
    python3 markov.py --exemplo  # caso do delivery, sem digitar nada
    python3 markov.py --testes   # dois casos de teste com saída esperada
"""

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")  # backend sem janela: o gráfico é salvo em arquivo
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Entrada de dados
# ---------------------------------------------------------------------------


def ler_inteiro(mensagem, minimo):
    """Lê um inteiro do teclado e repete a pergunta enquanto o valor for inválido."""
    while True:
        texto = input(mensagem).strip()
        try:
            valor = int(texto)
        except ValueError:
            print("  Valor inválido: digite um número inteiro.")
            continue
        if valor < minimo:
            print(f"  O valor precisa ser maior ou igual a {minimo}.")
            continue
        return valor


def ler_decimal(mensagem):
    """Lê um número real aceitando vírgula ou ponto como separador decimal."""
    while True:
        texto = input(mensagem).strip().replace(",", ".")
        try:
            return float(texto)
        except ValueError:
            print("  Valor inválido: digite um número, por exemplo 0.85.")


def ler_matriz(n):
    """Lê uma matriz quadrada n x n, linha por linha, com n valores por linha."""
    linhas = []
    for i in range(n):
        while True:
            texto = input(f"  Linha {i + 1} (ou 'u' para distribuição uniforme): ").strip()
            if texto.lower() == "u":
                linhas.append([1.0 / n] * n)
                break
            partes = texto.replace(",", ".").split()
            if len(partes) != n:
                print(f"  Informe exatamente {n} valores separados por espaço.")
                continue
            try:
                linhas.append([float(p) for p in partes])
            except ValueError:
                print("  Todos os valores precisam ser numéricos.")
                continue
            break
    return np.array(linhas, dtype=float)


def ler_vetor(n):
    """Lê o vetor de estado inicial (distribuição de probabilidade)."""
    while True:
        partes = input(f"  Vetor inicial com {n} valores: ").replace(",", ".").split()
        if len(partes) != n:
            print(f"  Informe exatamente {n} valores separados por espaço.")
            continue
        try:
            return np.array([float(p) for p in partes], dtype=float)
        except ValueError:
            print("  Todos os valores precisam ser numéricos.")


# ---------------------------------------------------------------------------
# 2. Validação das entradas
# ---------------------------------------------------------------------------


def validar_matriz(matriz):
    """Confere se a matriz é quadrada, não negativa e se cada linha soma 1.

    Uma matriz de transição estocástica precisa ter linhas que somam 1,
    porque cada linha representa todas as possibilidades de saída do estado.
    """
    if matriz.ndim != 2 or matriz.shape[0] != matriz.shape[1]:
        raise ValueError("A matriz de transição precisa ser quadrada.")
    if np.any(matriz < 0) or np.any(matriz > 1):
        raise ValueError("As probabilidades precisam estar entre 0 e 1.")
    somas = matriz.sum(axis=1)
    if not np.allclose(somas, 1.0, atol=1e-9):
        linha = int(np.argmax(np.abs(somas - 1.0))) + 1
        raise ValueError(f"A linha {linha} soma {somas[linha - 1]:.6f}; toda linha precisa somar 1.")


def validar_vetor(vetor, n):
    """Confere se o vetor inicial tem dimensão n, entradas não negativas e soma 1."""
    if vetor.shape != (n,):
        raise ValueError(f"O vetor inicial precisa ter {n} componentes.")
    if np.any(vetor < 0):
        raise ValueError("O vetor inicial não pode ter valores negativos.")
    if not np.isclose(vetor.sum(), 1.0, atol=1e-9):
        raise ValueError(f"O vetor inicial soma {vetor.sum():.6f}; precisa somar 1.")


# ---------------------------------------------------------------------------
# 3. Álgebra linear do cálculo estacionário
# ---------------------------------------------------------------------------


def multiplicar_vetor_matriz(vetor, matriz):
    """Calcula o produto de um vetor linha pela matriz: w = v · P (NumPy)."""
    return vetor @ matriz


def posto(matriz):
    """Calcula o posto da matriz pelo escalonamento (número de linhas independentes).

    Serve para saber quantas soluções o sistema (P^T - I)π = 0 admite:
    se o posto for n - 1, o vetor estacionário é único; se for menor, a cadeia
    tem mais de uma classe fechada e o limite depende do vetor inicial.
    """
    a = np.array(matriz, dtype=float, copy=True)
    n = a.shape[0]
    linhas_com_pivo = 0
    for coluna in range(n):
        # procura o maior valor absoluto da coluna a partir da linha atual (pivô)
        resto = np.abs(a[linhas_com_pivo:, coluna])
        if resto.size == 0 or resto.max() < 1e-12:
            # coluna praticamente nula: não há pivô, então o posto não cresce
            continue
        pivo = linhas_com_pivo + int(np.argmax(resto))
        if pivo != linhas_com_pivo:
            a[[linhas_com_pivo, pivo]] = a[[pivo, linhas_com_pivo]]
        for i in range(n):
            if i != linhas_com_pivo and abs(a[i, coluna]) > 0:
                fator = a[i, coluna] / a[linhas_com_pivo, coluna]
                a[i, coluna:] -= fator * a[linhas_com_pivo, coluna:]
        linhas_com_pivo += 1
    return linhas_com_pivo


def resolver_sistema(matriz, termos):
    """Resolve A x = b por eliminação de Gauss com pivoteamento parcial.

    Devolve None quando o sistema é singular (não há solução única).
    """
    n = matriz.shape[0]
    a = np.hstack([np.array(matriz, dtype=float), np.array(termos, dtype=float).reshape(n, 1)])
    for coluna in range(n):
        # pivoteamento parcial: sobe para a linha atual o maior pivô disponível
        pivo = coluna + int(np.argmax(np.abs(a[coluna:, coluna])))
        if abs(a[pivo, coluna]) < 1e-12:
            return None  # sistema singular: não existe solução única
        if pivo != coluna:
            a[[coluna, pivo]] = a[[pivo, coluna]]
        # eliminação: zera a coluna abaixo do pivô em todas as linhas seguintes
        for i in range(coluna + 1, n):
            fator = a[i, coluna] / a[coluna, coluna]
            a[i, coluna:] -= fator * a[coluna, coluna:]
    # substituição regressiva: da última incógnita para a primeira
    solucao = np.zeros(n)
    for i in range(n - 1, -1, -1):
        solucao[i] = (a[i, n] - a[i, i + 1:n] @ solucao[i + 1:n]) / a[i, i]
    return solucao


def matriz_do_sistema_estacionario(matriz):
    """Monta (P^T - I), a matriz do sistema π(P - I) = 0 escrito como Aπ = 0."""
    return matriz.T - np.eye(matriz.shape[0])


def distribuicao_estacionaria_por_sistema(matriz):
    """Acha o vetor estacionário resolvendo o sistema linear ligado ao autovalor 1.

    A condição π = πP pode ser escrita como π(P - I) = 0, ou seja, (P^T - I)π = 0.
    Esse sistema tem infinitas soluções (o vetor nulo é uma delas), então uma
    linha é substituída pela condição de normalização soma(π) = 1.
    Devolve None quando a cadeia tem mais de uma classe fechada (mais de um
    estado absorvente, por exemplo), pois aí o limite depende de v(0).
    """
    n = matriz.shape[0]
    sistema = matriz_do_sistema_estacionario(matriz)
    if n - posto(sistema) > 1:
        return None
    # a última equação do sistema homogêneo é trocada por soma(π) = 1
    sistema[-1] = np.ones(n)
    termos = np.zeros(n)
    termos[-1] = 1.0
    return resolver_sistema(sistema, termos)


def distribuicao_estacionaria_por_convergencia(vetor, matriz, tolerancia=1e-14, maximo=100000):
    """Aproxima o vetor estacionário repetindo v = vP até a variação ficar abaixo da tolerância.

    Devolve a distribuição final e o número de passos usados.
    """
    atual = np.array(vetor, dtype=float)
    for passo in range(1, maximo + 1):
        proximo = multiplicar_vetor_matriz(atual, matriz)
        variacao = float(np.max(np.abs(atual - proximo)))
        atual = proximo
        if variacao < tolerancia:
            return atual, passo
    raise ValueError("A cadeia não convergiu dentro do limite de passos.")


def evoluir(vetor, matriz, passos):
    """Devolve a história da distribuição: [v(0), v(1), ..., v(k)] como matriz (k+1) x n."""
    historia = np.zeros((passos + 1, vetor.size))
    historia[0] = vetor
    # cada linha é a distribuição da etapa anterior multiplicada por P
    for passo in range(1, passos + 1):
        historia[passo] = historia[passo - 1] @ matriz
    return historia


# ---------------------------------------------------------------------------
# 4. Saída de dados
# ---------------------------------------------------------------------------


def formatar_linha(nomes, valores):
    """Monta uma linha de tabela com nome do estado e probabilidade em porcentagem."""
    return "  ".join(f"{nome}={valor * 100:7.3f}%" for nome, valor in zip(nomes, valores))


def imprimir_distribuicao(titulo, nomes, valores):
    """Imprime uma distribuição com uma linha por estado."""
    print(f"\n{titulo}")
    for nome, valor in zip(nomes, valores):
        print(f"  {nome:<12} {valor * 100:9.4f}%")
    print(f"  {'soma':<12} {sum(valores) * 100:9.4f}%")


def escrever_grafico(historia, nomes, caminho, titulo):
    """Gera o gráfico da evolução com matplotlib e salva em PNG.

    As curvas usam tons de cinza e estilos de traço diferentes para continuarem
    distinguíveis quando o gráfico é impresso em preto e branco. A legenda fica
    fora da área do gráfico, para não cruzar as linhas.
    """
    estilos = [
        ("-", "#000000"),
        ("--", "#444444"),
        (":", "#777777"),
        ("-.", "#999999"),
        ((0, (1, 1)), "#bbbbbb"),
        ((0, (6, 2, 1, 2)), "#333333"),
    ]
    passos = np.arange(historia.shape[0])

    figura, eixo = plt.subplots(figsize=(9.0, 4.8), dpi=200, layout="constrained")
    for indice, nome in enumerate(nomes):
        estilo, cor = estilos[indice % len(estilos)]
        eixo.plot(passos, historia[:, indice] * 100, linestyle=estilo, color=cor,
                  linewidth=1.8, label=nome)
    eixo.set_xlabel("número de transições (k)", fontsize=10)
    eixo.set_ylabel("probabilidade (%)", fontsize=10)
    eixo.set_ylim(-2, 102)
    eixo.set_xlim(0, passos[-1])
    eixo.set_title(titulo, fontsize=11)
    eixo.grid(True, which="major", color="#dddddd", linewidth=0.6)
    eixo.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    eixo.spines["top"].set_visible(False)
    eixo.spines["right"].set_visible(False)
    figura.savefig(caminho, facecolor="white")
    plt.close(figura)
    return caminho


# ---------------------------------------------------------------------------
# 5. Casos prontos (usados no menu e nos testes)
# ---------------------------------------------------------------------------

NOMES_DELIVERY = ["Carrinho", "Pagamento", "Preparo", "Transito", "Entregue", "Cancelado"]

# Linhas = estado atual; colunas = próximo estado. Cada linha soma 1.
MATRIZ_DELIVERY = np.array([
    [0.05, 0.85, 0.00, 0.00, 0.00, 0.10],
    [0.00, 0.05, 0.85, 0.00, 0.00, 0.10],
    [0.00, 0.00, 0.05, 0.85, 0.05, 0.05],
    [0.00, 0.00, 0.00, 0.10, 0.85, 0.05],
    [0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
])
VETOR_DELIVERY = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def executar_caso(nomes, matriz, vetor, passos, titulo, caminho_png="grafico_evolucao.png"):
    """Roda o caso completo: valida, evolui, calcula o estacionário e gera o gráfico."""
    validar_matriz(matriz)
    validar_vetor(vetor, matriz.shape[0])
    historia = evoluir(vetor, matriz, passos)

    print(f"\n=== {titulo} ===")
    print(f"Ordem da matriz: {matriz.shape[0]}   Passos: {passos}")
    imprimir_distribuicao("Distribuição inicial v(0):", nomes, vetor)
    imprimir_distribuicao(f"Depois de {passos} transições:", nomes, historia[-1])

    por_sistema = distribuicao_estacionaria_por_sistema(matriz)
    por_convergencia, passos_ate_convergir = distribuicao_estacionaria_por_convergencia(vetor, matriz)
    if por_sistema is None:
        # Cadeia com dois ou mais estados absorventes: o sistema linear não tem
        # solução única, porque existem infinitos vetores fixos. O limite real
        # depende do vetor inicial, então só a convergência responde à pergunta.
        n_classes = matriz.shape[0] - posto(matriz_do_sistema_estacionario(matriz))
        print(
            f"\n  A matriz (P^T - I) tem {n_classes} vetores próprios associados ao autovalor 1:"
            "\n  a cadeia tem mais de uma classe fechada e o vetor limite depende de v(0)."
            "\n  Nesse caso o sistema linear sozinho não define o resultado; vale a convergência."
        )
    else:
        imprimir_distribuicao("Estacionário pelo sistema linear (autovalor 1):", nomes, por_sistema)
    imprimir_distribuicao(
        f"Estacionário por convergência ({passos_ate_convergir} passos):", nomes, por_convergencia
    )
    if por_sistema is not None:
        diferenca = float(np.max(np.abs(por_sistema - por_convergencia)))
        print(f"\n  Diferença máxima entre os dois métodos: {diferenca:.2e}")
    else:
        por_sistema = por_convergencia

    escrever_grafico(historia, nomes, caminho_png, titulo)
    print(f"  Gráfico salvo em: {caminho_png}")
    return historia, por_sistema


def caso_delivery():
    """Executa o exemplo do enunciado: fluxo de pedidos em um aplicativo de delivery."""
    return executar_caso(
        NOMES_DELIVERY, MATRIZ_DELIVERY, VETOR_DELIVERY, 20,
        "Fluxo de pedidos de delivery",
    )


def caso_interativo():
    """Lê do usuário a ordem, a matriz de transição, o vetor inicial e o número de passos."""
    print("\n--- Entrada de dados ---")
    n = ler_inteiro("Ordem da matriz de transição (n): ", 2)
    nomes = []
    for i in range(n):
        nomes.append(input(f"  Nome do estado {i + 1}: ").strip() or f"E{i + 1}")
    print("Digite cada linha da matriz com as probabilidades separadas por espaço.")
    matriz = ler_matriz(n)
    vetor = ler_vetor(n)
    passos = ler_inteiro("Número de transições a exibir no gráfico: ", 1)
    return executar_caso(nomes, matriz, vetor, passos, "Cadeia informada pelo usuário")


# ---------------------------------------------------------------------------
# 6. Testes automáticos
# ---------------------------------------------------------------------------


def teste_clima():
    """Caso 1: clima com dois estados; o estacionário exato é (0,6; 0,4).

    Sistema: π1 = 0,8π1 + 0,3π2 e π1 + π2 = 1  =>  0,2π1 = 0,3π2  =>  π1 = 1,5π2.
    Logo π2 = 1/2,5 = 0,4 e π1 = 0,6.
    """
    nomes = ["Sol", "Chuva"]
    matriz = np.array([[0.8, 0.2], [0.3, 0.7]])
    vetor = np.array([1.0, 0.0])
    _, estacionario = executar_caso(nomes, matriz, vetor, 10, "Teste 1 - previsão do tempo",
                                    "grafico_teste1.png")
    esperado = np.array([0.6, 0.4])
    erro = float(np.max(np.abs(estacionario - esperado)))
    print("\n  Saída esperada: Sol=60,000%  Chuva=40,000%")
    print(f"  Erro máximo: {erro:.2e}")
    assert erro < 1e-9, "Teste 1 falhou: estacionário diferente de (0,6; 0,4)."
    print("  Teste 1: OK")
    return erro


def teste_marcas():
    """Caso 2: migração de clientes entre três marcas, verificado por π = πP.

    O estacionário é calculado pelo sistema linear e depois conferido por
    substituição direta: se π é estacionário, então πP devolve o próprio π.
    """
    nomes = ["Marca A", "Marca B", "Marca C"]
    matriz = np.array([
        [0.90, 0.08, 0.02],
        [0.10, 0.85, 0.05],
        [0.05, 0.05, 0.90],
    ])
    vetor = np.array([1.0, 0.0, 0.0])
    _, estacionario = executar_caso(nomes, matriz, vetor, 30, "Teste 2 - migração de marcas",
                                    "grafico_teste2.png")
    conferencia = multiplicar_vetor_matriz(estacionario, matriz)
    residuo = float(np.max(np.abs(conferencia - estacionario)))
    print("\n  Conferência πP = π:")
    print("   ", formatar_linha(nomes, conferencia))
    print(f"  Resíduo máximo: {residuo:.2e}")
    assert residuo < 1e-12, "Teste 2 falhou: πP diferente de π."
    assert abs(estacionario.sum() - 1.0) < 1e-12, "Teste 2 falhou: soma diferente de 1."
    print("  Teste 2: OK")
    return residuo


def rodar_testes():
    """Executa os dois casos de teste exigidos no relatório."""
    print("Executando os testes do trabalho...")
    teste_clima()
    teste_marcas()
    print("\nTodos os testes passaram.")


# ---------------------------------------------------------------------------
# 7. Menu principal
# ---------------------------------------------------------------------------


def menu():
    """Mostra as opções e executa a escolha do usuário."""
    print("=" * 60)
    print(" Cadeias de Markov - Vetores, GA e Álgebra Linear (SPOMVAL)")
    print("=" * 60)
    print("1 - Exemplo do delivery (matriz 6x6 pronta)")
    print("2 - Informar uma matriz própria")
    print("3 - Rodar os testes automáticos")
    print("0 - Sair")
    escolha = input("Opção: ").strip()
    if escolha == "1":
        caso_delivery()
    elif escolha == "2":
        caso_interativo()
    elif escolha == "3":
        rodar_testes()
    elif escolha == "0":
        print("Fim.")
    else:
        print("Opção inválida.")
    return escolha


def main(argumentos):
    """Ponto de entrada: interpreta os argumentos de linha de comando."""
    if "--testes" in argumentos:
        rodar_testes()
    elif "--exemplo" in argumentos:
        caso_delivery()
    else:
        menu()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
