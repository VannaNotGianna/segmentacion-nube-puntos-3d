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

Se evaluarán algoritmos de clustering presentados en la Lecture 10 del curso, principalmente K-Means y DBSCAN.

La combinación final de métodos y los parámetros se determinarán mediante experimentos sobre los datos.

## 4. Estructura del proyecto

- `data/`: datos originales y procesados.
- `notebooks/`: exploración y experimentos.
- `src/`: funciones reutilizables.
- `results/`: etiquetas, nubes segmentadas y figuras.
- `reports/`: reporte final de la actividad.

## 5. Instalación

Crear y activar un entorno virtual e instalar las dependencias:

```bash
pip install -r requirements.txt
```

## 6. Integrantes

- Integrante 1: Elizabeth Huamán
- Integrante 2: [Nombre completo]
- Integrante 3: [Nombre completo]

**Grupo:** [Identificador]

## 7. Entregables

- Reporte PDF con procedimiento y capturas.
- Código fuente del proyecto.
- Nube de puntos segmentada en formato PLY.