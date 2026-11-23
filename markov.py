#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cadeias de Markov: previsão de comportamento de mercado.

Trabalho prático da disciplina SPOMVAL - Vetores, Geometria Analítica e
Álgebra Linear (IFSP, 2026/2).

O programa não usa bibliotecas de álgebra linear: multiplicação de matrizes,
eliminação de Gauss e geração do gráfico são implementadas com funções básicas
da linguagem, como pede o enunciado.

Uso:
    python3 markov.py            # menu interativo
    python3 markov.py --exemplo  # caso do delivery, sem digitar nada
    python3 markov.py --testes   # dois casos de teste com saída esperada
"""

import sys

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
    matriz = []
    for i in range(n):
        while True:
            linha = input(f"  Linha {i + 1} (ou 'u' para distribuição uniforme): ").strip()
            if linha.lower() == "u":
                matriz.append([1.0 / n] * n)
                break
            partes = linha.replace(",", ".").split()
            if len(partes) != n:
                print(f"  Informe exatamente {n} valores separados por espaço.")
                continue
            try:
                valores = [float(p) for p in partes]
            except ValueError:
                print("  Todos os valores precisam ser numéricos.")
                continue
            matriz.append(valores)
            break
    return matriz


def ler_vetor(n):
    """Lê o vetor de estado inicial (distribuição de probabilidade)."""
    while True:
        partes = input(f"  Vetor inicial com {n} valores: ").replace(",", ".").split()
        if len(partes) != n:
            print(f"  Informe exatamente {n} valores separados por espaço.")
            continue
        try:
            return [float(p) for p in partes]
        except ValueError:
            print("  Todos os valores precisam ser numéricos.")


# ---------------------------------------------------------------------------
# 2. Validação das entradas
# ---------------------------------------------------------------------------


def validar_matriz(matriz):
    """Confere se a matriz é quadrada, não negativa e se cada linha soma 1.

    Uma matriz de transição estocástica precisa ter linhas que somam 1,
    porque cada linha representa todas as possibilidades de saída do estado.
    A tolerância 1e-9 evita erro de arredondamento de ponto flutuante.
    """
    linhas = len(matriz)
    if linhas == 0:
        raise ValueError("A matriz está vazia.")
    for i, linha in enumerate(matriz):
        if len(linha) != linhas:
            raise ValueError(f"A matriz não é quadrada: a linha {i + 1} tem {len(linha)} valores.")
        for valor in linha:
            if valor < 0 or valor > 1:
                raise ValueError(f"Probabilidade fora do intervalo [0, 1] na linha {i + 1}.")
        if abs(sum(linha) - 1.0) > 1e-9:
            raise ValueError(
                f"A linha {i + 1} soma {sum(linha):.6f}; toda linha precisa somar 1."
            )


def validar_vetor(vetor, n):
    """Confere se o vetor inicial tem dimensão n, entradas não negativas e soma 1."""
    if len(vetor) != n:
        raise ValueError(f"O vetor inicial precisa ter {n} componentes.")
    if any(valor < 0 for valor in vetor):
        raise ValueError("O vetor inicial não pode ter valores negativos.")
    if abs(sum(vetor) - 1.0) > 1e-9:
        raise ValueError(f"O vetor inicial soma {sum(vetor):.6f}; precisa somar 1.")


# ---------------------------------------------------------------------------
# 3. Álgebra linear implementada com funções básicas
# ---------------------------------------------------------------------------


def multiplicar_vetor_matriz(vetor, matriz):
    """Calcula o produto de um vetor linha pela matriz: w_j = soma_i (v_i * P_ij).

    É a operação que aplica uma etapa da cadeia de Markov à distribuição.
    """
    n = len(vetor)
    resultado = []
    for j in range(n):
        soma = 0.0
        for i in range(n):
            soma += vetor[i] * matriz[i][j]
        resultado.append(soma)
    return resultado


def multiplicar_matrizes(a, b):
    """Calcula o produto de duas matrizes quadradas (usado para obter P^k)."""
    n = len(a)
    produto = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            soma = 0.0
            for k in range(n):
                soma += a[i][k] * b[k][j]
            produto[i][j] = soma
    return produto


def potencia_matriz(matriz, expoente):
    """Eleva a matriz de transição ao expoente por quadrados sucessivos."""
    n = len(matriz)
    resultado = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    base = [linha[:] for linha in matriz]
    while expoente > 0:
        if expoente % 2 == 1:
            resultado = multiplicar_matrizes(resultado, base)
        base = multiplicar_matrizes(base, base)
        expoente //= 2
    return resultado


def resolver_sistema(matriz, termos):
    """Resolve A x = b por eliminação de Gauss com pivoteamento parcial.

    Retorna None quando o sistema é singular (não há solução única).
    """
    n = len(matriz)
    a = [linha[:] + [termos[i]] for i, linha in enumerate(matriz)]
    for coluna in range(n):
        pivo = max(range(coluna, n), key=lambda i: abs(a[i][coluna]))
        if abs(a[pivo][coluna]) < 1e-12:
            return None
        a[coluna], a[pivo] = a[pivo], a[coluna]
        for i in range(coluna + 1, n):
            fator = a[i][coluna] / a[coluna][coluna]
            for j in range(coluna, n + 1):
                a[i][j] -= fator * a[coluna][j]
    solucao = [0.0] * n
    for i in range(n - 1, -1, -1):
        soma = a[i][n] - sum(a[i][j] * solucao[j] for j in range(i + 1, n))
        solucao[i] = soma / a[i][i]
    return solucao


def posto(matriz):
    """Calcula o posto da matriz pelo escalonamento (número de linhas independentes).

    Serve para saber quantas soluções o sistema (P^T - I)π = 0 admite:
    se o posto for n - 1, o vetor estacionário é único; se for menor, a cadeia
    tem mais de uma classe fechada e o limite depende do vetor inicial.
    """
    n = len(matriz)
    a = [linha[:] for linha in matriz]
    linhas_com_pivo = 0
    for coluna in range(n):
        pivo = max(range(linhas_com_pivo, n), key=lambda i: abs(a[i][coluna]), default=None)
        if pivo is None or abs(a[pivo][coluna]) < 1e-12:
            continue
        a[linhas_com_pivo], a[pivo] = a[pivo], a[linhas_com_pivo]
        for i in range(n):
            if i != linhas_com_pivo and abs(a[i][coluna]) > 0:
                fator = a[i][coluna] / a[linhas_com_pivo][coluna]
                for j in range(coluna, n):
                    a[i][j] -= fator * a[linhas_com_pivo][j]
        linhas_com_pivo += 1
    return linhas_com_pivo


def matriz_do_sistema_estacionario(matriz):
    """Monta (P^T - I), a matriz do sistema π(P - I) = 0 escrito como Aπ = 0."""
    n = len(matriz)
    sistema = []
    for i in range(n):
        linha = []
        for j in range(n):
            linha.append(matriz[j][i] - (1.0 if i == j else 0.0))
        sistema.append(linha)
    return sistema


def distribuicao_estacionaria_por_sistema(matriz):
    """Acha o vetor estacionário resolvendo o sistema linear ligado ao autovalor 1.

    A condição π = πP pode ser escrita como π(P - I) = 0, ou seja, (P^T - I)π = 0.
    Esse sistema tem infinitas soluções (o vetor nulo é uma delas), então uma
    linha é substituída pela condição de normalização soma(π) = 1.
    Devolve None quando a cadeia tem mais de uma classe fechada (mais de um
    estado absorvente, por exemplo), pois aí o limite depende de v(0).
    """
    n = len(matriz)
    sistema = matriz_do_sistema_estacionario(matriz)
    if n - posto(sistema) > 1:
        return None
    sistema[-1] = [1.0] * n  # normalização: soma das componentes = 1
    termos = [0.0] * (n - 1) + [1.0]
    return resolver_sistema(sistema, termos)


def distribuicao_estacionaria_por_convergencia(vetor, matriz, tolerancia=1e-14, maximo=100000):
    """Aproxima o vetor estacionário repetindo v = vP até a variação ficar abaixo da tolerância.

    Devolve a distribuição final e o número de passos usados.
    """
    atual = vetor[:]
    for passo in range(1, maximo + 1):
        proximo = multiplicar_vetor_matriz(atual, matriz)
        variacao = max(abs(a - b) for a, b in zip(atual, proximo))
        atual = proximo
        if variacao < tolerancia:
            return atual, passo
    raise ValueError("A cadeia não convergiu dentro do limite de passos.")


def evoluir(vetor, matriz, passos):
    """Devolve a história da distribuição: [v(0), v(1), ..., v(k)]."""
    historia = [vetor[:]]
    atual = vetor[:]
    for _ in range(passos):
        atual = multiplicar_vetor_matriz(atual, matriz)
        historia.append(atual)
    return historia


# ---------------------------------------------------------------------------
# 4. Saída de dados
# ---------------------------------------------------------------------------


def formatar_linha(nomes, valores):
    """Monta uma linha de tabela com nome do estado e probabilidade em porcentagem."""
    partes = []
    for nome, valor in zip(nomes, valores):
        partes.append(f"{nome}={valor * 100:7.3f}%")
    return "  ".join(partes)


def imprimir_distribuicao(titulo, nomes, valores):
    """Imprime uma distribuição com uma linha por estado."""
    print(f"\n{titulo}")
    for nome, valor in zip(nomes, valores):
        print(f"  {nome:<12} {valor * 100:9.4f}%")
    print(f"  {'soma':<12} {sum(valores) * 100:9.4f}%")


def escrever_grafico_svg(historia, nomes, caminho, titulo):
    """Gera um gráfico de linhas em SVG puro, sem biblioteca de desenho.

    O eixo horizontal é o número de transições e o vertical, a probabilidade
    de cada estado. As linhas usam tons de cinza e estilos de traço diferentes
    para continuarem distinguíveis quando o gráfico é impresso em preto e branco.
    """
    largura, altura = 1060, 560
    esquerda, direita, topo, base = 90, 300, 60, 70
    area_largura = largura - esquerda - direita
    area_altura = altura - topo - base
    passos = len(historia) - 1

    def x(passo):
        """Converte o índice do passo em coordenada horizontal do SVG."""
        return esquerda + (passo / passos) * area_largura if passos else esquerda

    def y(valor):
        """Converte uma probabilidade em coordenada vertical (eixo invertido)."""
        return topo + area_altura - valor * area_altura

    estilos = [
        ("#000000", "6,0", 2.4),
        ("#444444", "5,3", 2.0),
        ("#777777", "2,3", 2.0),
        ("#999999", "8,2,2,2", 1.8),
        ("#bbbbbb", "1,3", 2.2),
        ("#333333", "10,3", 1.6),
    ]

    partes = []
    partes.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" '
        f'viewBox="0 0 {largura} {altura}" font-family="Helvetica, Arial, sans-serif">'
    )
    partes.append(f'<rect width="{largura}" height="{altura}" fill="#ffffff"/>')
    partes.append(
        f'<text x="{largura / 2:.0f}" y="32" font-size="19" text-anchor="middle">{titulo}</text>'
    )
    # grade horizontal e rótulos do eixo y
    for i in range(6):
        valor = i / 5
        yy = y(valor)
        partes.append(
            f'<line x1="{esquerda}" y1="{yy:.1f}" x2="{esquerda + area_largura}" y2="{yy:.1f}" '
            f'stroke="#dddddd" stroke-width="1"/>'
        )
        partes.append(
            f'<text x="{esquerda - 12}" y="{yy + 5:.1f}" font-size="13" text-anchor="end">'
            f"{valor * 100:.0f}%</text>"
        )
    # eixos
    partes.append(
        f'<line x1="{esquerda}" y1="{topo}" x2="{esquerda}" y2="{topo + area_altura}" stroke="#000" stroke-width="1.4"/>'
    )
    partes.append(
        f'<line x1="{esquerda}" y1="{topo + area_altura}" x2="{esquerda + area_largura}" '
        f'y2="{topo + area_altura}" stroke="#000" stroke-width="1.4"/>'
    )
    # marcas do eixo x
    marcas = sorted(set([0, passos // 4, passos // 2, (3 * passos) // 4, passos]))
    for passo in marcas:
        partes.append(
            f'<line x1="{x(passo):.1f}" y1="{topo + area_altura}" x2="{x(passo):.1f}" '
            f'y2="{topo + area_altura + 6}" stroke="#000" stroke-width="1.2"/>'
        )
        partes.append(
            f'<text x="{x(passo):.1f}" y="{topo + area_altura + 24}" font-size="13" '
            f'text-anchor="middle">{passo}</text>'
        )
    partes.append(
        f'<text x="{esquerda + area_largura / 2:.0f}" y="{altura - 22}" font-size="14" '
        f'text-anchor="middle">número de transições (k)</text>'
    )
    partes.append(
        f'<text x="26" y="{topo + area_altura / 2:.0f}" font-size="14" text-anchor="middle" '
        f'transform="rotate(-90 26 {topo + area_altura / 2:.0f})">probabilidade do estado</text>'
    )
    # uma polilinha por estado
    for indice, nome in enumerate(nomes):
        cor, tracejado, espessura = estilos[indice % len(estilos)]
        pontos = " ".join(
            f"{x(passo):.1f},{y(historia[passo][indice]):.1f}" for passo in range(passos + 1)
        )
        partes.append(
            f'<polyline points="{pontos}" fill="none" stroke="{cor}" stroke-width="{espessura}" '
            f'stroke-dasharray="{tracejado}" stroke-linejoin="round"/>'
        )
    # legenda fora da área do gráfico, para não cruzar as linhas
    base_legenda = esquerda + area_largura + 30
    for indice, nome in enumerate(nomes):
        cor, tracejado, espessura = estilos[indice % len(estilos)]
        yy = topo + 10 + indice * 24
        partes.append(
            f'<line x1="{base_legenda}" y1="{yy}" x2="{base_legenda + 42}" '
            f'y2="{yy}" stroke="{cor}" stroke-width="{espessura}" stroke-dasharray="{tracejado}"/>'
        )
        partes.append(
            f'<text x="{base_legenda + 52}" y="{yy + 5}" font-size="13">{nome}</text>'
        )
    partes.append("</svg>")

    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(partes))
    return caminho


# ---------------------------------------------------------------------------
# 5. Casos prontos (usados no menu e nos testes)
# ---------------------------------------------------------------------------

NOMES_DELIVERY = ["Carrinho", "Pagamento", "Preparo", "Transito", "Entregue", "Cancelado"]

# Linhas = estado atual; colunas = próximo estado. Cada linha soma 1.
MATRIZ_DELIVERY = [
    [0.05, 0.85, 0.00, 0.00, 0.00, 0.10],
    [0.00, 0.05, 0.85, 0.00, 0.00, 0.10],
    [0.00, 0.00, 0.05, 0.85, 0.05, 0.05],
    [0.00, 0.00, 0.00, 0.10, 0.85, 0.05],
    [0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
]
VETOR_DELIVERY = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def executar_caso(nomes, matriz, vetor, passos, titulo, caminho_svg="grafico_evolucao.svg"):
    """Roda o caso completo: valida, evolui, calcula o estacionário e gera o gráfico."""
    validar_matriz(matriz)
    validar_vetor(vetor, len(matriz))
    historia = evoluir(vetor, matriz, passos)

    print(f"\n=== {titulo} ===")
    print(f"Ordem da matriz: {len(matriz)}   Passos: {passos}")
    imprimir_distribuicao("Distribuição inicial v(0):", nomes, vetor)
    imprimir_distribuicao(f"Depois de {passos} transições:", nomes, historia[-1])

    por_sistema = distribuicao_estacionaria_por_sistema(matriz)
    por_convergencia, passos_ate_convergir = distribuicao_estacionaria_por_convergencia(vetor, matriz)
    if por_sistema is None:
        # Cadeia com dois ou mais estados absorventes: o sistema linear não tem
        # solução única, porque existem infinitos vetores fixos. O limite real
        # depende do vetor inicial, então só a convergência responde à pergunta.
        n_classes = len(matriz) - posto(matriz_do_sistema_estacionario(matriz))
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
        diferenca = max(abs(a - b) for a, b in zip(por_sistema, por_convergencia))
        print(f"\n  Diferença máxima entre os dois métodos: {diferenca:.2e}")
    else:
        por_sistema = por_convergencia

    escrever_grafico_svg(historia, nomes, caminho_svg, titulo)
    print(f"  Gráfico salvo em: {caminho_svg}")
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
    validar_matriz(matriz)
    validar_vetor(vetor, n)
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
    matriz = [[0.8, 0.2], [0.3, 0.7]]
    vetor = [1.0, 0.0]
    historia, estacionario = executar_caso(nomes, matriz, vetor, 10, "Teste 1 - previsão do tempo")
    esperado = [0.6, 0.4]
    erro = max(abs(a - b) for a, b in zip(estacionario, esperado))
    print(f"\n  Saída esperada: Sol=60,000%  Chuva=40,000%")
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
    matriz = [
        [0.90, 0.08, 0.02],
        [0.10, 0.85, 0.05],
        [0.05, 0.05, 0.90],
    ]
    vetor = [1.0, 0.0, 0.0]
    historia, estacionario = executar_caso(nomes, matriz, vetor, 30, "Teste 2 - migração de marcas")
    conferencia = multiplicar_vetor_matriz(estacionario, matriz)
    residuo = max(abs(a - b) for a, b in zip(conferencia, estacionario))
    print("\n  Conferência πP = π:")
    print("   ", formatar_linha(nomes, conferencia))
    print(f"  Resíduo máximo: {residuo:.2e}")
    assert residuo < 1e-12, "Teste 2 falhou: πP diferente de π."
    assert abs(sum(estacionario) - 1.0) < 1e-12, "Teste 2 falhou: soma diferente de 1."
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
