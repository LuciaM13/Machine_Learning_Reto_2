# Machine_Learning_Reto_2

Este proyecto aplica varios modelos de machine learning para identificar si una transacción es fraudulenta o no. El objetivo es ayudar a instituciones a identificar comportamientos sospechosos de forma automática y prevenir pérdidas económicas en caso de fraudes.

## Contenido del repositorio:

requirements: Contiene los requerimientos y librerias necesarias.
00_Data: contiene los datasets utilizados. Raw y Cleaned con los datos originales y procesados.
03_Models: contiene los modelos entrenados en formato .pkl.
README.md: este archivo.

## Dataset
El conjunto de datos utilizado contiene transacciones financieras anónimas con características como:

Amount: Precio de la transacción.
Time: Tiempo desde la primera transacción registrada.
V1 a V28: variables transformadas mediante PCA para proteger la privacidad.
Class: Variable objetivo (0 = legítima, 1 = fraude).

