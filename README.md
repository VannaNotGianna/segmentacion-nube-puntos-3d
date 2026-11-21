# Segmentación de una nube de puntos 3D

**Curso:** Machine Learning  
**Docente:** Jan Hurtado - UTEC  
**Actividad:** Actividad 3 - Clustering

## 1. Descripción

El proyecto tiene como objetivo segmentar los objetos presentes en una nube de puntos 3D correspondiente a una fábrica, utilizando algoritmos de aprendizaje no supervisado.

La nube contiene información espacial (XYZ), color (RGB) y normales. Se busca identificar agrupaciones que permitan distinguir estructuras como pisos, techos, columnas, tuberías y otros objetos.

## 2. Objetivos

- Analizar y preparar la nube de puntos original en formato PLY.
- Aplicar algoritmos de clustering estudiados en clase.
- Comparar y evaluar los resultados de segmentación.
- Exportar una nube de puntos PLY coloreada según los segmentos identificados.

## 3. Metodología

La segmentación combina K-Means y DBSCAN en varias etapas, usando en cada una el atributo más informativo:

| Etapa | Algoritmo | Atributos | Resultado |
|---|---|---|---|
| 0. Preprocesamiento | Submuestreo por vóxel (0.08 m) + filtro de outliers | XYZ | ~455 mil puntos |
| 1a. Orientación | K-Means (k=3) | \|normal\| | Horizontal / vertical X / vertical Y |
| 1b-c. Superficies | DBSCAN (eps=0.15 m) + reglas de altura y tamaño | XYZ de puntos horizontales | Piso, techo, plataformas |
| 2a. Columnas | Cobertura vertical en planta + DBSCAN 2D | XY | Columnas |
| 2b. Materiales | K-Means (k=5) | Color (tono, saturación, brillo) | Claro, verde, óxido, oscuro |
| 2c-d. Objetos | DBSCAN (eps=0.2 m) por material + reglas | XYZ, material, altura | Tubería, viga/estructura, maquinaria |
| 2e. Suavizado | Voto mayoritario de 10 vecinos | XYZ | Etiquetas limpias |
| 3. Transferencia | Vecino más cercano | XYZ | Nube original (1.65 M puntos) etiquetada |

La elección de parámetros (k, eps, tamaño de vóxel) y su efecto se documentan en el notebook y en `results/metricas/`.

### Clases y colores

| Clase | Color |
|---|---|
| Piso | azul |
| Techo | celeste |
| Plataforma | amarillo |
| Columna | verde |
| Tubería | rojo |
| Viga / estructura metálica | morado |
| Maquinaria | naranja |
| Otros | gris |

## 4. Estructura del proyecto

```
data/raw/                 nube original (no incluida en el repositorio, ver 5.2)
notebooks/                exploración, experimentos y capturas para el reporte
src/
  preprocessing.py        carga, submuestreo y filtro de outliers
  clustering.py           etapas 1 y 2, transferencia de etiquetas
  visualization.py        colores por clase, exportación PLY, figuras y vistas 3D
  pipeline.py             ejecución completa del pipeline
  evaluation.py           métricas de K-Means y estabilidad
tests/                    pruebas unitarias y escena sintética
results/
  point_clouds/           nubes PLY segmentadas (generadas)
  figures/                figuras y capturas (generadas)
  labels/                 etiquetas por punto en .npy (generadas)
  metricas/               tablas CSV de evaluación
reports/                  reporte final de la actividad
```

## 5. Instalación

### 5.1. Entorno

Requiere Python 3.11. Desde la raíz del proyecto:

```bash
python -m venv .venv
```

Activar el entorno:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (Git Bash)
source .venv/Scripts/activate
# Linux / macOS
source .venv/bin/activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

### 5.2. Datos

La nube de puntos original no se incluye en el repositorio por su tamaño (44 MB). Copiar el archivo a:

```
data/raw/FARO_Scan_Full_Sub1_down3_with_normals.ply
```

## 6. Ejecución

Todos los comandos se ejecutan desde la raíz del proyecto, con el entorno activado.

### 6.1. Segmentación completa

```bash
python -m src.pipeline
```

Tarda alrededor de 30 segundos y genera:

| Archivo | Contenido |
|---|---|
| `results/point_clouds/segmentacion_final_completa.ply` | **Entregable:** nube original completa coloreada por clase |
| `results/point_clouds/segmentos_individuales.ply` | Nube original completa con un color por segmento (cluster); gris = ruido de DBSCAN |
| `results/point_clouds/segmentacion_final.ply` | Nube submuestreada coloreada por clase |
| `results/point_clouds/etapa1_estructura.ply` | Resultado de la etapa 1 (piso, techo, plataformas) |
| `results/labels/clases_final_completa.npy` | Clase de cada punto de la nube original |
| `results/labels/segmentos_completa.npy` | Segmento de cada punto de la nube original (-1 = ruido) |
| `results/metricas/conteo_clases_nube_completa.csv` | Número y porcentaje de puntos por clase |
| `results/metricas/segmentos.csv` | Origen, clase, tamaño y extensión de cada segmento |
| `results/figures/00_*.png` a `05_*.png`, `07_segmentos.png` | Proyecciones de cada etapa y de los segmentos |

Opciones: `--voxel` (tamaño de vóxel en metros, por defecto 0.08), `--surface-eps` (eps de DBSCAN para superficies, por defecto 0.15) e `--input` (ruta a otra nube PLY con normales).

### 6.2. Pruebas

```bash
python -m pytest
```

Ejecuta 33 pruebas en unos segundos: cada etapa por separado y el pipeline completo sobre una escena sintética con clases conocidas (piso, techo, columnas, tubería, viga y máquina).

### 6.3. Evaluación

```bash
python -m src.evaluation
```

Tarda alrededor de 2 minutos y genera:

- `results/metricas/kmeans_orientacion.csv` y `kmeans_color.csv`: inercia, silhouette y Davies-Bouldin para distintos k.
- `results/metricas/estabilidad.csv`: concordancia de las clases al cambiar la semilla y los parámetros principales.
- `results/figures/06_seleccion_k.png`: gráficos para la elección de k.

### 6.4. Notebook

```bash
jupyter notebook notebooks/segmentacion_actividad3.ipynb
```

Recorre cada etapa con sus experimentos y genera las capturas 3D para el reporte (`results/figures/10_*` a `23_*`, incluidas vistas por segmento y acercamientos). Se guarda con sus salidas, por lo que puede revisarse sin ejecutarlo. Lee las tablas de `results/metricas/` incluidas en el repositorio; si se cambian parámetros, conviene regenerarlas antes con `python -m src.evaluation`.

### 6.5. Visualización

Las nubes PLY generadas pueden abrirse en [CloudCompare](https://www.cloudcompare.org/) o [MeshLab](https://www.meshlab.net/).

## 7. Integrantes

- Elizabeth Huamán Santillán
- Marialejandra Bautista Gastelo
- Joaquín Arriaga Cosio

**Grupo:** 9

## 8. Entregables

- Reporte PDF con procedimiento y capturas.
- Código fuente del proyecto.
- Nube de puntos segmentada en formato PLY.
