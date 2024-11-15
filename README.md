# BI_Project

Este repositorio contiene dos proyectos relacionados con el análisis y la implementación de soluciones de inteligencia de negocios (BI) en el ámbito de la salud neonatal en Colombia. Cada proyecto aborda diferentes aspectos del procesamiento y análisis de datos, utilizando técnicas de ETL, modelado dimensional y visualización en herramientas de BI.

## Estructura del Repositorio

- **Project 1**: Enfocado en la preparación de datos, creación de entornos virtuales, y ejecución de scripts para análisis y procesamiento de datos.
- **Project 2**: Desarrollo de un modelo dimensional, ETL automatizado y visualización de datos a través de tableros en Power BI conectados a Google BigQuery.

---

## Proyecto 1

### Descripción

Este proyecto está orientado a la configuración de un entorno de trabajo para el análisis de datos mediante Python. La parte 2 del proyecto se enfoca en la creación de un entorno virtual y la instalación de las dependencias necesarias para el procesamiento de datos en un entorno controlado.

### Pasos para ejecutar la parte 2 de Proyecto 1

1. **Crear el entorno virtual**: 
   ```bash
   python -m venv ./venv
   ```
2. **Instalar las librerías necesarias desde `requirements.txt`**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ubicarse en la carpeta `StageTwo`**:
   ```bash
   cd BI_Project-/Project One/StageTwo/
   ```
4. **Ejecutar la aplicación**:
   ```bash
   py ./app.py
   ```

---

## Proyecto 2: Análisis de Salud Neonatal en Colombia

### Descripción

El segundo proyecto tiene como objetivo construir una solución de inteligencia de negocios que permita a los expertos en salud neonatal en Colombia monitorear y analizar indicadores clave. Usando datos de nacimientos, condiciones ambientales y datos socioeconómicos, se ha diseñado una arquitectura de BI que incluye un modelo dimensional en BigQuery, alimentado por un flujo de ETL en DataPrep y visualizado en tableros de Power BI. Esta solución busca proporcionar insights para mejorar las decisiones de salud pública.

### Estructura del Proyecto 2

- **Google Cloud Platform (GCP)**: Almacenamiento de datos en Cloud Storage, con integración en BigQuery para análisis.
- **DataPrep**: Procesamiento y transformación de datos (ETL) para limpieza y preparación antes de ser cargados en BigQuery.
- **BigQuery**: Base de datos en la que se implementa el modelo dimensional para el análisis.
- **Power BI**: Visualización de los datos a través de tableros de control interactivos, permitiendo el monitoreo y análisis en tiempo real.

### Principales Funcionalidades

- **Análisis Descriptivo**: Tableros que muestran condiciones demográficas y ambientales de los nacimientos en cada municipio.
- **Análisis Clínico**: Visualización de métricas clave como estado civil, edad y consultas prenatales.
- **Análisis Inferencial**: Modelos estadísticos que examinan asociaciones entre variables, tales como el tipo de parto y puntajes APGAR.

### Justificación

Este proyecto es relevante para los actores de salud pública, pues permite comprender factores socioeconómicos, ambientales y clínicos que influyen en la salud neonatal en Colombia, orientando la toma de decisiones y políticas en el sector.

---

## Requerimientos

Cada proyecto utiliza librerías específicas detalladas en `requirements.txt`. Se recomienda ejecutar el código en un entorno virtual para asegurar la compatibilidad de dependencias.

## Estructura de Archivos

- **Project One**: Carpeta que contiene los archivos de configuración y scripts de procesamiento para el Proyecto 1.
- **Project Two**: Carpeta que contiene la documentación y archivos de configuración de GCP y DataPrep para el Proyecto 2.

---

Este repositorio ofrece una visión completa de cómo los datos pueden ser usados para mejorar la salud pública a través de análisis descriptivos y estadísticos, utilizando herramientas de procesamiento y visualización en un entorno de BI.

---
