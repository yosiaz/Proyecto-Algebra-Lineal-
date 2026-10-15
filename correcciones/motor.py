"""
motor.py — El motor del Compresor Matricial MTXC
══════════════════════════════════════════════════════════════════════
Este archivo contiene TODA la matemática y la lógica del proyecto.
No tiene interfaz de usuario: solo recibe datos y devuelve datos.

Lo usan dos programas que lo importan:
    mtxc_consola.py   muestra resultados en la terminal
    mtxc_gui.py      muestra resultados en una ventana con botones

MAPA DE ESTE ARCHIVO:
    Sección 1 — Matrices      (la Álgebra Lineal del proyecto)
    Sección 2 — Bloques       (preparar datos para las matrices)
    Sección 3 — Entropía      (métrica matemática para decidir)
    Sección 4 — Huffman       (la compresión real en bytes)
    Sección 5 — Lector BMP    (para imágenes, sin librerías externas)
    Sección 6 — Compresor     (une todo lo anterior)
    Sección 7 — Formato .mtxc (cómo guardamos el resultado en disco)

REGLA CENTRAL: las matrices transforman los datos (no reducen tamaño).
Huffman comprime los datos transformados (sí reduce tamaño).
La entropía decide si la transformación vale la pena.
══════════════════════════════════════════════════════════════════════
"""

# struct: empaca y desempaca números en bytes con formato exacto.
#   Ejemplo: struct.pack("<I", 5000) → 4 bytes que representan el número 5000.
#   Lo usamos para escribir y leer el encabezado del archivo .mtxc.
import struct

# heapq: montículo de mínimos (min-heap).
#   Es la estructura de datos que necesita el algoritmo de Huffman
#   para siempre sacar los dos elementos de menor frecuencia primero.
import heapq

# numpy: álgebra lineal numérica.
#   Lo usamos para construir matrices (np.eye, np.tril, np.zeros),
#   multiplicarlas (D @ X) y manejar arrays de bytes eficientemente.
#   La multiplicación de matrices con dtype=uint8 hace mod 256 automático.
import numpy as np

# os y Path: para crear carpetas, construir rutas y obtener nombres de archivos
#   de forma que funcione igual en Windows, Mac y Linux.
import os
from pathlib import Path


# ==================================
# CONSTANTES
# ==================================

# Tamaño de cada bloque 1D (número de bytes por vector).
# Elegimos 8 porque es lo suficientemente grande para que la
# transformación tenga efecto, y lo suficientemente pequeño para
# que sea fácil de explicar a mano (una matriz 8×8).
N = 8

# Tamaño de cada bloque 2D para imágenes (B×B píxeles por bloque).
# Usamos 8×8 porque ese mismo tamaño usa JPEG internamente:
B = 8

# Firma del formato: los primeros 4 bytes de cualquier archivo .mtxc.
# Sirve para verificar que el archivo es nuestro antes de descomprimir.
FIRMA = b"MTXC"

# Códigos de modo que se guardan en el archivo .mtxc.
# Le dicen al descompresor qué tipo de datos tiene el archivo.
MODO_VACIO      = 0   # el archivo original estaba vacío
MODO_GENERICO   = 1   # texto, PDF, exe, cualquier cosa que no sea BMP 24-bit
MODO_IMAGEN     = 2   # imagen BMP 24-bit sin comprimir (transformación 2D)
MODO_PAQUETE    = 3   # varios archivos empacados en uno solo
MODO_ALMACENADO = 4   # archivo guardado tal cual (comprimir lo agrandaba)


# ================================
# SECCIÓN 1: MATRICES
# ================================

# ANILLO DE TRABAJO: Z/256Z
# Trabajamos con enteros de 0 a 255 (un byte).
# La aritmética es módulo 256: 255+1=0, 0-1=255.
# Esto es exactamente cómo funciona un byte en cualquier computadora.
# Con numpy dtype=uint8, esto sucede automáticamente en cada operación.
# CONDICIÓN DE INVERTIBILIDAD EN Z/256Z:
# Una matriz cuadrada A es invertible módulo 256 si y solo si
# su determinante es impar (no divisible entre 2).
# D1 tiene determinante 1 (impar) → siempre es invertible.


def construir_d1(n):
    """
    Construye D1: la matriz de diferencias de 1er orden, tamaño N×N.

    QUÉ HACE:
        Transforma un bloque de bytes x en un bloque y donde:
            y[0]   = x[0]              (primer valor: se copia tal cual)
            y[i]   = x[i] - x[i-1]    (resto: diferencia con el anterior)

    POR QUÉ:
        Si los bytes consecutivos son similares (como en imágenes),
        las diferencias son pequeñas y se repiten → Huffman los
        comprime mucho mejor que los valores originales.

    INVERTIBILIDAD:
        det(D1) = 1×1×...×1 = 1 -> impar -> D1-1 existe en Z/256Z.

    ESTRUCTURA (para N=4):
        [  1   0   0   0 ]
        [ -1   1   0   0 ]    -1 se escribe como 255 (= -1 mod 256)
        [  0  -1   1   0 ]
        [  0   0  -1   1 ]
    """
    D = np.eye(n, dtype=np.uint8)      # empieza como identidad
    for i in range(1, n):
        D[i, i - 1] = 255              # 255 = -1 mod 256, en la subdiagonal
    return D


def construir_d1_inversa(n):
    """
    Construye D1-1: la inversa de D1, tamaño N×N.

    QUÉ HACE:
        Deshace exactamente lo que hizo D1.
        Si y = D1·x, entonces x = D1-1·y.
        La operación es la "suma acumulada":
            x[i] = y[0] + y[1] + ... + y[i]

    ESTRUCTURA (para N=4):
        [ 1  0  0  0 ]
        [ 1  1  0  0 ]    triangular inferior de puros unos
        [ 1  1  1  0 ]
        [ 1  1  1  1 ]
    """
    return np.tril(np.ones((n, n), dtype=np.uint8))


def verificar_inversa(n):
    
    #Comprueba que D1 @ D1-1 == Identidad para un tamaño N dado.

    #Retorna True si la verificación es correcta (siempre debería serlo)

    D     = construir_d1(n)
    D_inv = construir_d1_inversa(n)
    I     = np.eye(n, dtype=np.uint8)
    return np.array_equal(D @ D_inv, I)


def aplicar_d1(X, D):
    """
    Aplica la transformación Y = D·X (mod 256) a TODOS los bloques a la vez.

    X es una matriz N×M donde cada columna es un bloque del archivo.
    Multiplicar D (N×N) por X (N×M) transforma los M bloques de una vez.
    numpy hace el módulo 256 automáticamente con dtype=uint8.
    """
    return D @ X


def desaplicar_d1(Y, D_inv):
    """
    Aplica la transformación inversa X = D-1 ·Y (mod 256).
    Reconstruye los bloques originales a partir de los transformados.
    """
    return D_inv @ Y


def aplicar_d1_2d(bloque, D):
    """
    Aplica la transformación 2D: Y = D·X·Dt (mod 256).

    Para imágenes usamos esta versión porque los píxeles tienen
    estructura en DOS dimensiones (filas y columnas), no en una sola.

    QUÉ HACE:
        D·X      aplica diferencias a las FILAS del bloque
        (D·X)·Dt  aplica diferencias a las COLUMNAS del resultado

    bloque es una matriz B×B de píxeles de un solo canal de color.
    """
    return D @ bloque @ D.T    # primero filas (D@bloque), luego columnas (@D.T)


def desaplicar_d1_2d(bloque, D_inv):
    """
    Inversa de aplicar_d1_2d: X = D-1·Y·(D-1)t (mod 256).
    Reconstruye el bloque original de píxeles.
    """
    return D_inv @ bloque @ D_inv.T


# =====================================================================
# SECCIÓN 2: BLOQUES
# =====================================================================
# Las matrices operan sobre vectores de tamaño N.
# Estas funciones convierten los bytes del archivo en vectores,
# y reconvierten los vectores a bytes al final.



def bytes_a_bloques(datos, n):
    """
    Convierte una secuencia de bytes en la matriz de bloques X.

    Agrupa los bytes de N en N. Cada grupo es un vector columna.
    El resultado X tiene N filas y M columnas, donde M = [len(datos)/N].

    RELLENO:
        Si len(datos) no es múltiplo de N, se agregan ceros al final.
        La longitud original se devuelve para eliminar ese relleno al descomprimir.

    Retorna: (X, longitud_original)
    """
    L = len(datos)
    if L == 0:
        return np.zeros((n, 0), dtype=np.uint8), 0

    M = -(-L // n)                                    # ceil(L / n)
    arreglo = np.zeros(M * n, dtype=np.uint8)
    arreglo[:L] = np.frombuffer(datos, dtype=np.uint8)
    return arreglo.reshape(M, n).T, L                 # .T  columnas = bloques


def bloques_a_bytes(X, L):
    """
    Operación inversa de bytes_a_bloques.

    Convierte la matriz X de vuelta a bytes y recorta el relleno
    usando la longitud original L.
    """
    return X.T.reshape(-1)[:L].astype(np.uint8).tobytes()


# ========================================================
# SECCIÓN 3: ENTROPÍA
# ========================================================
# La entropía de Shannon mide qué tan comprimible es una secuencia.
# La usamos para decidir si D1 vale la pena en cada archivo concreto.


def entropia(valores):
    """
    Calcula la entropía de Shannon en bits/byte.

    FÓRMULA:  H = -Σ p(v) · log₂(p(v))
    p(v) es la proporción de veces que aparece el valor v

    INTERPRETACIÓN:
        H bajo (cerca de 0) = distribución muy desigual = muy comprimible.
        H alto (cerca de 8) = distribución uniforme = poco comprimible.

    CÓMO SE USA:
        Se calcula H de los datos sin transformar y H de los datos con D1
        Si H_con_D1 < H_sin_D1  D1 ayuda se aplica
        Si no  se usan los datos originales directamente
    """
    if len(valores) == 0:
        return 0.0
    frec = np.bincount(valores.ravel().astype(np.uint8), minlength=256).astype(float)
    frec = frec[frec > 0]           # ignorar bytes que no aparecen
    p    = frec / frec.sum()        # convertir a probabilidades
    return float(-(p * np.log2(p)).sum())


# ========================================
# SECCIÓN 4: HUFFMAN
# ========================================
# Huffman asigna códigos binarios de longitud variable a cada byte:
#los bytes más frecuentes reciben los códigos más cortos
#los menos frecuentes reciben los más largos
# Esto REDUCE el tamaño real en bytes.

# IMPORTANTE: Huffman NO es Álgebra Lineal. Es teoría de la información.
# Las matrices preparan los datos (reorganizan su distribución).
# Huffman los empaca eficientemente aprovechando esa distribución.


def _construir_codigos(frecuencias):
    """
    [Interna] Construye el árbol de Huffman y devuelve los códigos.

    ALGORITMO:
        1. Crear una hoja por cada byte que aparece en los datos
        2. Repetir hasta quedar con un solo nodo:
            a. Sacar los dos nodos de MENOR frecuencia
            b. Unirlos en un nodo nuevo (frecuencia = suma de ambas)
        3. Recorrer el árbol resultante: izquierda="0", derecha="1"

    Retorna: dict {byte: string_de_bits}
        Ejemplo: {32: "00", 101: "010", 116: "11"}
        El byte más frecuente tiene el código más corto.
    """
    if not frecuencias:
        return {}
    if len(frecuencias) == 1:
        # caso especial: solo hay un tipo de byte → código "0"
        return {list(frecuencias.keys())[0]: "0"}

    # el heap guarda: (frecuencia, id_único, símbolo_o_None, hijo_izq, hijo_der)
    # el id_único rompe empates sin comparar estructuras complejas
    heap = []
    for idx, (simbolo, frec) in enumerate(frecuencias.items()):
        heapq.heappush(heap, (frec, idx, simbolo, None, None))

    siguiente_id = len(frecuencias)
    while len(heap) > 1:
        # sacar los dos de menor frecuencia
        f1, _, s1, iz1, de1 = heapq.heappop(heap)
        f2, _, s2, iz2, de2 = heapq.heappop(heap)
        # crear nodo interno que los une
        nodo_iz = (f1, 0, s1, iz1, de1)
        nodo_de = (f2, 0, s2, iz2, de2)
        heapq.heappush(heap, (f1 + f2, siguiente_id, None, nodo_iz, nodo_de))
        siguiente_id += 1

    # recorrer el árbol completo para asignar un código a cada hoja
    codigos = {}
    pila = [(heap[0], "")]
    while pila:
        nodo, prefijo = pila.pop()
        f, _, simbolo, izq, der = nodo
        if simbolo is not None:
            codigos[simbolo] = prefijo or "0"   # hoja: guardar su código
        else:
            pila.append((izq, prefijo + "0"))   # bajar izquierda: añadir "0"
            pila.append((der, prefijo + "1"))   # bajar derecha: añadir "1"
    return codigos


def huffman_comprimir(valores):
    """
    Comprime un array de bytes usando codificación de Huffman.

    PROCESO:
        1. Contar cuántas veces aparece cada byte (frecuencias)
        2. Construir el árbol y obtener los códigos
        3. Reemplazar cada byte por su código (string de "0" y "1")
        4. Empacar esos bits en bytes reales (grupos de 8)

    Retorna: (bytes_comprimidos, bits_de_relleno, frecuencias)
        bits_de_relleno: cuántos "0" se añadieron al final para
                         completar el último byte (se necesita al descomprimir)
        frecuencias:     se guarda en el archivo para poder reconstruir
                         el mismo árbol al descomprimir
    """
    if len(valores) == 0:
        return b"", 0, {}

    # contar frecuencias de cada byte
    frec_arr    = np.bincount(valores.ravel().astype(np.uint8), minlength=256)
    frecuencias = {int(v): int(c) for v, c in enumerate(frec_arr) if c > 0}

    # construir los códigos del árbol de Huffman
    codigos = _construir_codigos(frecuencias)

    # codificar: cada byte → su código de bits
    tabla  = [""] * 256
    for simbolo, codigo in codigos.items():
        tabla[simbolo] = codigo
    bits   = "".join(tabla[int(v)] for v in valores.ravel())

    # rellenar con ceros hasta el próximo múltiplo de 8
    relleno = (8 - len(bits) % 8) % 8
    bits   += "0" * relleno

    # empacar el string de bits en bytes reales
    empacado = int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""
    return empacado, relleno, frecuencias


def huffman_descomprimir(empacado, relleno, frecuencias, total):
    """
    Descomprime bytes comprimidos con huffman_comprimir().

    CÓMO FUNCIONA:
        Reconstruye el árbol de Huffman con las mismas frecuencias
        que se guardaron en el archivo (el árbol es siempre idéntico
        si las frecuencias son las mismas). Luego lee el flujo de bits
        y lo recorre bit a bit hasta encontrar cada símbolo.

    total: cuántos bytes hay que producir (se guardó en el archivo)
    """
    if total == 0:
        return np.array([], dtype=np.uint8)

    codigos = _construir_codigos(frecuencias)

    # construir árbol de nodos para decodificar bit a bit
    class _N:
        __slots__ = ("izq", "der", "val")
        def __init__(self): self.izq = self.der = self.val = None

    raiz = _N()
    for simbolo, codigo in codigos.items():
        n = raiz
        for bit in codigo:
            if bit == "0":
                if n.izq is None: n.izq = _N()
                n = n.izq
            else:
                if n.der is None: n.der = _N()
                n = n.der
        n.val = simbolo    # hoja: guardar el símbolo

    # convertir bytes comprimidos  string de bits
    nbits = len(empacado) * 8
    bits  = bin(int.from_bytes(empacado, "big"))[2:].zfill(nbits) if empacado else ""
    utiles = len(bits) - relleno    # ignorar los bits de relleno del final

    # decodificar leyendo bit a bit
    salida = np.empty(total, dtype=np.uint8)
    nodo = raiz
    idx  = 0
    for i in range(utiles):
        nodo = nodo.izq if bits[i] == "0" else nodo.der
        if nodo.val is not None:        # llegamos a una hoja
            salida[idx] = nodo.val
            idx  += 1
            nodo  = raiz               # volver a la raíz para el siguiente
            if idx == total:
                break
    return salida


# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 5: LECTOR/ESCRITOR BMP
# ══════════════════════════════════════════════════════════════════════
# Las imágenes BMP de 24 bits sin comprimir son matrices 2D de píxeles.
# Las leemos manualmente (sin Pillow ni librerías de imagen) para poder
# garantizar reconstrucción byte a byte exacta.
#
# ¿Por qué solo BMP y no PNG/JPG?
# PNG y JPG ya vienen comprimidos internamente. Modificar sus píxeles y
# recodificarlos nunca reproduciría los bytes exactos del original.
# Solo BMP sin comprimir garantiza eso, porque sus píxeles son datos crudos.
# ══════════════════════════════════════════════════════════════════════

def _leer_bmp(datos):
    """
    [Interna] Lee un BMP de 24 bits sin comprimir (BI_RGB).

    Retorna un dict con todos los campos del BMP, o None si el archivo
    no es un BMP de 24 bits compatible.

    El dict contiene:
        encabezado:   bytes originales del encabezado (se guardan verbatim)
        ancho, alto:  dimensiones en píxeles
        relleno_fila: bytes de padding por fila (BMP alinea a múltiplos de 4)
        pad_datos:    los bytes de relleno reales (para reconstruir exacto)
        canales:      lista [B, G, R] de matrices numpy uint8 (alto×ancho)
        extra:        bytes tras los píxeles (normalmente vacío)
    """
    if len(datos) < 54 or datos[:2] != b"BM":
        return None                                     # no es BMP

    offset = struct.unpack_from("<I", datos, 10)[0]    # dónde empiezan los píxeles
    dib    = struct.unpack_from("<I", datos, 14)[0]    # tamaño del DIB header
    if dib != 40:
        return None                                     # solo BITMAPINFOHEADER

    ancho      = struct.unpack_from("<i", datos, 18)[0]
    alto_raw   = struct.unpack_from("<i", datos, 22)[0]
    bpp        = struct.unpack_from("<H", datos, 28)[0]
    compresion = struct.unpack_from("<I", datos, 30)[0]

    if bpp != 24 or compresion != 0:
        return None                                     # solo 24-bit sin comprimir

    alto   = abs(alto_raw)
    stride = ((ancho * 3 + 3) // 4) * 4               # ancho en bytes, alineado a 4
    pad_f  = stride - ancho * 3                        # bytes de relleno por fila

    pixeles = datos[offset: offset + stride * alto]
    extra   = datos[offset + stride * alto:]

    # separar los 3 canales de color
    canales   = [np.zeros((alto, ancho), dtype=np.uint8) for _ in range(3)]
    pad_bytes = bytearray()
    for f in range(alto):
        fila = pixeles[f * stride: f * stride + stride]
        px   = np.frombuffer(fila[:ancho * 3], dtype=np.uint8).reshape(ancho, 3)
        canales[0][f] = px[:, 0]    # azul
        canales[1][f] = px[:, 1]    # verde
        canales[2][f] = px[:, 2]    # rojo
        pad_bytes.extend(fila[ancho * 3:])

    return {
        "encabezado":  datos[:offset],
        "ancho":       ancho,
        "alto":        alto,
        "relleno_fila": pad_f,
        "pad_datos":   bytes(pad_bytes),
        "canales":     canales,
        "extra":       extra,
    }


def _escribir_bmp(info):
    """
    [Interna] Reconstruye los bytes exactos del BMP original
    a partir de los canales de color (posiblemente transformados/restaurados).
    """
    ancho, alto = info["ancho"], info["alto"]
    pad_f       = info["relleno_fila"]
    pad_datos   = info["pad_datos"]
    canales     = info["canales"]

    filas = []
    for f in range(alto):
        px   = np.stack([canales[0][f], canales[1][f], canales[2][f]], axis=1)
        fila = px.astype(np.uint8).tobytes()
        if pad_f:
            fila += pad_datos[f * pad_f: (f + 1) * pad_f]
        filas.append(fila)

    return info["encabezado"] + b"".join(filas) + info["extra"]


# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 6: COMPRESOR CENTRAL
# ══════════════════════════════════════════════════════════════════════
# Estas funciones unen las secciones 1-5 para comprimir y descomprimir
# un blob de bytes. Son la "cocina" del programa.
# ══════════════════════════════════════════════════════════════════════

def _comprimir_1d(datos):
    """
    [Interna] Comprime bytes arbitrarios con D1 (1D) + Huffman.

    PROCESO:
        1. Convertir bytes → bloques (vectores de tamaño N)
        2. Medir entropía SIN transformar (H_sin) y CON D1 (H_con)
        3. Usar la representación de menor entropía
        4. Comprimir esa representación con Huffman

    Retorna: dict con todos los datos necesarios para descomprimir
    """
    X, L = bytes_a_bloques(datos, N)

    D     = construir_d1(N)
    D_inv = construir_d1_inversa(N)
    Y     = aplicar_d1(X, D)           # Y = D * X  (mod 256)

    h_sin = entropia(X)                 # entropía sin transformar
    h_con = entropia(Y)                 # entropía con D1
    usa   = h_con < h_sin              # True si D1 ayuda

    vals            = Y.T.ravel() if usa else X.T.ravel()
    comp, rell, frec = huffman_comprimir(vals)

    return {
        "usa_d1": usa, "n": N, "L": L,
        "total":  len(vals),
        "relleno": rell, "frecuencias": frec, "comprimido": comp,
        "h_sin": h_sin, "h_con": h_con,
    }


def _descomprimir_1d(r):
    """
    [Interna] Inversa de _comprimir_1d(). Recibe el dict del registro.
    """
    vals = huffman_descomprimir(r["comprimido"], r["relleno"], r["frecuencias"], r["total"])
    M    = r["total"] // r["n"]
    Y    = vals.reshape(M, r["n"]).T    # reconstruir matriz de bloques

    if r["usa_d1"]:
        X = desaplicar_d1(Y, construir_d1_inversa(r["n"]))
    else:
        X = Y                           # no se transformó, Y ya es X

    return bloques_a_bytes(X, r["L"])


def _comprimir_imagen(info_bmp):
    """
    [Interna] Comprime una imagen BMP con D1 2D (Y=D*X*Dt) + Huffman.

    PROCESO:
        1. Dividir cada canal de color en bloques de B×B píxeles
        2. Medir entropía global SIN y CON transformación 2D
        3. Aplicar (o no) D·bloque·Dᵀ según cuál dé menor entropía
        4. Aplanar todos los bloques transformados  Huffman

    LA CLAVE MATEMÁTICA:
        Los píxeles vecinos de una foto son similares entre sí
        tanto en dirección horizontal como vertical.
        D·X·Dt captura esa correlación en AMBAS dimensiones,
        logrando diferencias más pequeñas y más repetidas.
        Resultado: H baja dramáticamente (ej: 7.7 → 3.0 bits/byte).
    """
    D     = construir_d1(B)
    D_inv = construir_d1_inversa(B)
    alto  = info_bmp["alto"]
    ancho = info_bmp["ancho"]

    # dimensiones redondeadas al próximo múltiplo de B
    alto_r  = -(-alto  // B) * B
    ancho_r = -(-ancho // B) * B

    def canal_a_lista_bloques(canal):
        """Rellena el canal a (alto_r × ancho_r) y devuelve lista de bloques B×B."""
        lienzo = np.zeros((alto_r, ancho_r), dtype=np.uint8)
        lienzo[:alto, :ancho] = canal
        return [lienzo[i:i+B, j:j+B].copy()
                for i in range(0, alto_r, B)
                for j in range(0, ancho_r, B)]

    # recolectar todos los bloques de los 3 canales
    bloques = [canal_a_lista_bloques(c) for c in info_bmp["canales"]]

    # medir entropía sin y con transformación 2D
    vals_sin = np.concatenate([b.ravel() for canal in bloques for b in canal])
    vals_con = np.concatenate(
        [aplicar_d1_2d(b, D).ravel() for canal in bloques for b in canal]
    )
    h_sin = entropia(vals_sin)
    h_con = entropia(vals_con)
    usa   = h_con < h_sin

    vals_finales    = vals_con if usa else vals_sin
    comp, rell, frec = huffman_comprimir(vals_finales)

    return {
        "usa_2d": usa,
        "alto": alto, "ancho": ancho,
        "alto_r": alto_r, "ancho_r": ancho_r,
        "num_bloques": len(bloques[0]),     # bloques por canal
        "total":   len(vals_finales),
        "relleno": rell, "frecuencias": frec, "comprimido": comp,
        "h_sin": h_sin, "h_con": h_con,
    }


def _descomprimir_imagen(r, info_bmp_meta):
    """
    [Interna] Inversa de _comprimir_imagen().

    info_bmp_meta: dict con encabezado, ancho, alto, relleno, etc.
                   (se leyó del archivo .mtxc, no del BMP original)
    """
    D_inv = construir_d1_inversa(B)
    vals  = huffman_descomprimir(r["comprimido"], r["relleno"], r["frecuencias"], r["total"])

    alto_r  = r["alto_r"]
    ancho_r = r["ancho_r"]
    alto    = r["alto"]
    ancho   = r["ancho"]
    nb      = r["num_bloques"]          # bloques por canal

    # desempacar: vals está organizado como [canal0_bloque0, canal0_bloque1, ..., canal2_bloque_M]
    canales = []
    for c in range(3):
        segmento = vals[c * nb * B * B: (c + 1) * nb * B * B]
        bloques  = segmento.reshape(nb, B, B)

        # reconstruir el lienzo del canal
        lienzo   = np.zeros((alto_r, ancho_r), dtype=np.uint8)
        idx_blq  = 0
        for i in range(0, alto_r, B):
            for j in range(0, ancho_r, B):
                blk = bloques[idx_blq]
                if r["usa_2d"]:
                    blk = desaplicar_d1_2d(blk, D_inv)
                lienzo[i:i+B, j:j+B] = blk
                idx_blq += 1

        canales.append(lienzo[:alto, :ancho])

    return _escribir_bmp({**info_bmp_meta, "canales": canales})


"""
SECCIÓN 7: FORMATO .mtxc
Definición del formato binario del archivo de salida.

ESTRUCTURA GENERAL:
[4 B] FIRMA    b"MTXC"
[1 B] VERSION  1
[1 B] MODO     0=vacío  1=genérico  2=imagen  3=paquete
[...] contenido según el modo

REGISTRO GENÉRICO (usado en modo 1 y dentro de modo 3):
[1 B] usa_d1     0 o 1
[1 B] N          tamaño de bloque
[4 B] L          longitud original
[4 B] total      total de valores comprimidos
[1 B] relleno    bits de relleno Huffman
[1 B] K-1        número de símbolos en la tabla - 1
[5×K B] tabla    {símbolo(1B) + frecuencia(4B)} × K
[resto] flujo    bits Huffman empacados

MODO 2 (imagen):
[4 B] len_enc    longitud del encabezado BMP
[?]   enc        encabezado BMP original
[4 B] len_pad    longitud del bloque de padding de filas
[?]   pad        padding de filas del BMP
[4 B] len_ext    longitud de datos extra
[?]   ext        datos extra del BMP
[2 B] ancho      en píxeles
[2 B] alto       en píxeles
[1 B] relleno_f  bytes de relleno por fila
[1 B] usa_2d     0 o 1
[2 B] alto_r     alto redondeado a múltiplo de B
[2 B] ancho_r    ancho redondeado a múltiplo de B
[4 B] num_bloq   bloques por canal
[4 B] total      total de valores comprimidos
[1 B] relleno    bits de relleno Huffman
[1 B] K-1
[5×K B] tabla
[resto] flujo
"""

def _serializar_registro(r):
    """[Interna] Convierte el dict de _comprimir_1d() a bytes."""
    K     = len(r["frecuencias"])
    tabla = b"".join(struct.pack("<BI", s, f) for s, f in r["frecuencias"].items())
    return (
        struct.pack("<BB II BB",
            int(r["usa_d1"]), r["n"],
            r["L"], r["total"],
            r["relleno"], K - 1)
        + tabla + r["comprimido"]
    )


def _deserializar_registro(datos, pos=0):
    """[Interna] Lee un registro genérico desde `datos` a partir de `pos`."""
    usa_d1, n, L, total, relleno, K_menos_1 = struct.unpack_from("<BB II BB", datos, pos)
    pos += 12
    K    = K_menos_1 + 1
    frec = {}
    for _ in range(K):
        s, f  = struct.unpack_from("<BI", datos, pos)
        frec[s] = f
        pos    += 5
    comprimido = datos[pos:]
    return {"usa_d1": bool(usa_d1), "n": n, "L": L, "total": total,
            "relleno": relleno, "frecuencias": frec, "comprimido": comprimido}


#lo que el usuario verá 

def comprimir(rutas_entrada, ruta_salida):
    """
    Comprime uno o varios archivos a un único archivo .mtxc.

    MODO AUTOMÁTICO:
        • 1 archivo BMP 24-bit sin comprimir → modo IMAGEN (D1 2D + Huffman)
        • 1 archivo de cualquier otro tipo   → modo GENÉRICO (D1 1D + Huffman)
        • 2 o más archivos                   → modo PAQUETE (genérico por archivo)

    Retorna: dict con estadísticas para mostrar en la interfaz
    """
# normalizar: si se pasa una sola ruta como string, convertir a lista
    if isinstance(rutas_entrada, (str, Path)):
        rutas_entrada = [rutas_entrada]
    rutas_entrada = [Path(r) for r in rutas_entrada]

    # leer todos los archivos
    contenidos = [r.read_bytes() for r in rutas_entrada]
    tam_original = sum(len(c) for c in contenidos)

# determinar el modo
    if len(contenidos) == 0 or all(len(c) == 0 for c in contenidos):
        modo = MODO_VACIO

    elif len(contenidos) == 1:
        info_bmp = _leer_bmp(contenidos[0])
        if info_bmp is not None:
            modo = MODO_IMAGEN
        else:
            modo = MODO_GENERICO

    else:
        modo = MODO_PAQUETE

# construir el cuerpo del archivo según el modo
    encabezado = FIRMA + struct.pack("<BB", modo)
    stats      = {"modo": modo, "tam_original": tam_original}

    if modo == MODO_VACIO:
        cuerpo = b""

    elif modo == MODO_GENERICO:
        r     = _comprimir_1d(contenidos[0])
        cuerpo = _serializar_registro(r)
        stats.update({"usa_d1": r["usa_d1"], "h_sin": r["h_sin"], "h_con": r["h_con"]})

    elif modo == MODO_IMAGEN:
        r_img = _comprimir_imagen(info_bmp)
# encabezado BMP, padding y extra se guardan verbatim
        enc    = info_bmp["encabezado"]
        pad    = info_bmp["pad_datos"]
        ext    = info_bmp["extra"]
        tabla  = b"".join(struct.pack("<BI", s, f)
                           for s, f in r_img["frecuencias"].items())
        K      = len(r_img["frecuencias"])
        cuerpo = (
            struct.pack("<I", len(enc)) + enc +
            struct.pack("<I", len(pad)) + pad +
            struct.pack("<I", len(ext)) + ext +
            struct.pack("<HH B B HH I I BB",
                r_img["ancho"], r_img["alto"],
                info_bmp["relleno_fila"],
                int(r_img["usa_2d"]),
                r_img["alto_r"], r_img["ancho_r"],
                r_img["num_bloques"], r_img["total"],
                r_img["relleno"], K - 1)
            + tabla + r_img["comprimido"]
        )
        stats.update({"usa_2d": r_img["usa_2d"],
                      "h_sin": r_img["h_sin"], "h_con": r_img["h_con"]})

    else:  # MODO_PAQUETE
        partes = [struct.pack("<H", len(contenidos))]
        for ruta, datos in zip(rutas_entrada, contenidos):
            nombre   = ruta.name.encode("utf-8")
            r        = _comprimir_1d(datos)
            registro = _serializar_registro(r)
            partes.append(
                struct.pack("<B", len(nombre)) + nombre +
                struct.pack("<I", len(registro)) + registro
            )
        cuerpo = b"".join(partes)

    # verificar si conviene comprimir o guardar tal cual.
    # Si el resultado comprimido es igual o más grande que el original + 6 bytes
    # de encabezado mínimo, no tiene sentido comprimir: guardar el archivo tal cual
    # con el modo ALMACENADO (solo se añaden 6 bytes de encabezado).
    contenido_final = encabezado + cuerpo
    if modo not in (MODO_VACIO, MODO_PAQUETE) and len(contenido_final) >= tam_original + 6:
        contenido_final = (FIRMA
                           + struct.pack("<BB", MODO_ALMACENADO)
                           + contenidos[0])          # bytes originales, sin tocar
        stats["almacenado"] = True
    else:
        stats["almacenado"] = False

    Path(ruta_salida).write_bytes(contenido_final)
    stats["tam_comprimido"] = len(contenido_final)
    return stats


def descomprimir(ruta_mtxc, carpeta_salida=None):
    """
    Descomprime un archivo .mtxc y escribe los archivos originales.

    Si carpeta_salida es None, los archivos se escriben en el mismo
    directorio que el .mtxc, con el sufijo "_reconstruido".

    Retorna: lista de rutas de los archivos escritos
    """
    datos  = Path(ruta_mtxc).read_bytes()

    if datos[:4] != FIRMA:
        raise ValueError("El archivo no tiene la firma MTXC válida.")

    version = datos[4]
    modo    = datos[5]
    cuerpo  = datos[6:]

    if carpeta_salida is None:
        carpeta_salida = Path(ruta_mtxc).parent

    os.makedirs(carpeta_salida, exist_ok=True)
    archivos_escritos = []

    # modo almacenado (sin compresión)
    # El archivo se guardó tal cual porque comprimirlo lo agrandaba.
    # Solo hace falta escribir los bytes de vuelta.
    if modo == MODO_ALMACENADO:
        stem = Path(ruta_mtxc).stem
        ruta = Path(carpeta_salida) / (stem + "_reconstruido.bin")
        ruta.write_bytes(cuerpo)
        return [str(ruta)]

    #modo vacío 
    if modo == MODO_VACIO:
        ruta = Path(carpeta_salida) / (Path(ruta_mtxc).stem + "_reconstruido.bin")
        ruta.write_bytes(b"")
        archivos_escritos.append(str(ruta))

#modo genérico 
    elif modo == MODO_GENERICO:
        r         = _deserializar_registro(cuerpo)
        recuperado = _descomprimir_1d(r)
        stem      = Path(ruta_mtxc).stem
        ruta      = Path(carpeta_salida) / (stem + "_reconstruido.bin")
        ruta.write_bytes(recuperado)
        archivos_escritos.append(str(ruta))

#modo imagen 
    elif modo == MODO_IMAGEN:
        pos = 0

        len_enc = struct.unpack_from("<I", cuerpo, pos)[0]; pos += 4
        enc     = cuerpo[pos: pos + len_enc];               pos += len_enc
        len_pad = struct.unpack_from("<I", cuerpo, pos)[0]; pos += 4
        pad     = cuerpo[pos: pos + len_pad];               pos += len_pad
        len_ext = struct.unpack_from("<I", cuerpo, pos)[0]; pos += 4
        ext     = cuerpo[pos: pos + len_ext];               pos += len_ext

        ancho, alto, rf, usa_2d, alto_r, ancho_r, num_b, total, rell, K_m1 = \
            struct.unpack_from("<HH B B HH I I BB", cuerpo, pos); pos += 20

        K    = K_m1 + 1
        frec = {}
        for _ in range(K):
            s, f  = struct.unpack_from("<BI", cuerpo, pos)
            frec[s] = f; pos += 5

        comprimido = cuerpo[pos:]
        r_img      = {"usa_2d": bool(usa_2d), "alto": alto, "ancho": ancho,
                      "alto_r": alto_r, "ancho_r": ancho_r, "num_bloques": num_b,
                      "total": total, "relleno": rell,
                      "frecuencias": frec, "comprimido": comprimido}
        info_meta  = {"encabezado": enc, "relleno_fila": rf,
                      "pad_datos": pad, "extra": ext,
                      "alto": alto, "ancho": ancho, "canales": None}
        recuperado = _descomprimir_imagen(r_img, info_meta)
        stem       = Path(ruta_mtxc).stem
        ruta       = Path(carpeta_salida) / (stem + "_reconstruido.bmp")
        ruta.write_bytes(recuperado)
        archivos_escritos.append(str(ruta))

#modo paquete 
    elif modo == MODO_PAQUETE:
        pos       = 0
        num       = struct.unpack_from("<H", cuerpo, pos)[0]; pos += 2
        for _ in range(num):
            ln      = cuerpo[pos];                        pos += 1
            nombre  = cuerpo[pos: pos + ln].decode();    pos += ln
            lr      = struct.unpack_from("<I", cuerpo, pos)[0]; pos += 4
            reg_raw = cuerpo[pos: pos + lr];             pos += lr

            r          = _deserializar_registro(reg_raw)
            recuperado = _descomprimir_1d(r)
            stem       = Path(nombre).stem
            sufijo     = Path(nombre).suffix or ".bin"
            ruta       = Path(carpeta_salida) / (stem + "_reconstruido" + sufijo)
            ruta.write_bytes(recuperado)
            archivos_escritos.append(str(ruta))

    return archivos_escritos
