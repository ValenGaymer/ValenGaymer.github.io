# Ejercicio Notas de Clase (1.0 puntos)

Considere los ejemplos vistos en clase, en los que se analizó la implementación de los modelos: **k-nn**, **regresión lineal**, **regresión logística**. Realice hiperparametrización y validación cruzada usando:

1. **GridSearchCV y Pipeline**
2. **Manual**: Sin usar GridSearchCV y Pipeline (utilice ciclos FOR, WHILE, etc.)

Verifique que los scores obtenidos en los ítems (1) y (2) son los mismos.

## Ejercicios

### Breast Cancer: (KNN, LogisticRegression)

Decida cuál es la métrica de mayor importancia en la aplicación de detección de cáncer (métrica de negocio). Utilice esta métrica para la evaluación y selección del modelo y justifique su respuesta. Los resultados deben ser presentados usando el Cuadro 1. El estudiante que obtenga el mejor score con esta métrica, será premiado con una décima para el corte. Los datos deben ser cargados mediante el siguiente par de líneas:

- **Datos**:
  ```python
  from sklearn.datasets import load_breast_cancer
  cancer = load_breast_cancer()
    ```

### Boston Housing: (LinearRegression, KNN)
Utilice la métrica R2 y RMSE durante la evaluación y selección del modelo. Los resultados deben ser presentados usando el Cuadro 2. Realice un gráfico en el que muestre en los conjuntos de entrenamiento y test, el precio original y el predicho. El estudiante que obtenga el mejor score con esta métrica, será premiado con una décima para el corte. Los datos deben ser cargados mediante el siguiente par de líneas:

- **Datos**:
  ```python
  import mglearn
  X, y = mglearn.datasets.load_extended_boston()

# EDA + ETL (2.0 puntos)

Dado los siguientes conjuntos de datos: **NeurIPS 2024 - Predict New Medicines with BELKA** y **Open Problems – Single-Cell Perturbations**, se requiere realizar un análisis exploratorio de datos que incluya lo siguiente:

## Análisis del Problema de Aplicación

- **Resumen del Problema**: Describa la importancia del problema, en qué consiste y la fuente de los datos.
- **Descripción de Variables**: Calcule el número de observaciones, media, desviación estándar, mínimo, máximo, cuartiles.
- **Conteo de Datos Faltantes**: Realice un conteo de los datos faltantes y su porcentaje.
- **Visualizaciones**:
  - **Histograma o Diagrama de Barras**: Para la variable respuesta e independientes según corresponda.
  - **Boxplot**: Realice un análisis de simetría, datos atípicos y dispersión.

## Análisis Bivariado

- **Scatterplot**: Trazado de scatterplot() para analizar la relación entre pares de variables.
- **Regplot**: Trazado de regplot() para visualizar la relación lineal entre variables.

En cada figura, agregue un análisis y descripción. Complete el EDA con visualizaciones que muestren patrones importantes presentes en los datos.

## Imputación de Datos Faltantes

- Realice imputación de datos faltantes usando imputación múltiple iterativa (ver IterativeImputer()).

## Reducción de Dimensionalidad

- **Eliminación de Columnas Altamente Correlacionadas**: Use Variance Inflation Factor (VIF) para esto. Se recomienda usar la librería `variance_inflation_factor()`.
  - Un VIF ≥ 10 indica alta multicolinealidad entre una variable independiente y más de dos variables explicativas.
  - **Recomendación**: Elimine una columna a la vez, comenzando por aquella con el máximo VIF ≥ 10. Luego, calcule nuevamente el VIF para el nuevo DataFrame y repita el proceso hasta obtener solo valores de VIF < 10.

- **Codificación de Variables Categóricas**: Las variables categóricas deben codificarse previamente usando técnicas como OneHotEncoder(). Mantenga las variables categóricas antes de la codificación previa al entrenamiento del modelo y reduzca la multicolinealidad usando la prueba chi2_contingency(). Si el número de variables explicativas es pequeño y cada una es de gran importancia para las predicciones, considere mantenerlas todas. Tome la decisión más adecuada y justifíquela.

# Modelos de Clasificación (1.0 puntos)
Considere el conjunto de datos NeurIPS 2024 - Predict New Medicines with BELKA. Implemente la versión de clasificación para cada uno de los modelos estudiados en clase, es decir, Regresión Logística y KNN. 

- **Tareas**:
  1. **Construir una tabla de error** que contenga las métricas usuales de clasificación: 
     - Precisión
     - Recall
     - F1-score
     - AUC
  2. **Agregar matrices de confusión** (ver `confusion_matrix`) y **curvas ROC** (ver `plot_roc`).
  3. Utilizar **GridSearchCV** y **Pipeline** para evaluar cada modelo.
  4. Verificar que la validación cruzada seleccionada es la adecuada y justificarlo.
  5. Utilizar la métrica **AUC** para seleccionar el mejor modelo de clasificación (maximizar AUC).

- **Resultados**:
  - Los resultados deben estar registrados en una tabla de error (ver **Tabla 1**) que resuma cada score obtenido por el modelo implementado.

# Modelos de Regresión (1.0 puntos)
Considere el conjunto de datos Open Problems – Single-Cell Perturbations. Implemente la versión de regresión de cada uno de los modelos estudiados en clase, es decir, KNN y Regresión Lineal en el conjunto de datos suministrado.

- **Tareas**:
  1. **Construir una tabla de error** con las métricas usuales de regresión:
     - MAPE
     - MAE
     - RMSE
     - MSE
     - R²
  2. Utilizar la métrica **Mean Rowwise Root Mean Squared Error (MRRMSE)** en la evaluación y validación, para seleccionar el mejor modelo de regresión.

- **Resultados**:
  - Los resultados deben estar registrados en una tabla de error (ver **Table 2**) que resuma cada score obtenido por el modelo implementado.





