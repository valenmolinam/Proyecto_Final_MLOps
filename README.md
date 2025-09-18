# Proyecto_Final_MLOps
Objetivo del proyecto: El proyecto se centra en el análisis y detección de transacciones fraudulentas en sistemas de dinero móvil. Usaremos el dataset PaySim1, que contiene una simulación de millones de transacciones financieras realizadas en una aplicación de pagos digitales. El objetivo principal es identificar patrones de fraude mediante el uso de técnicas de análisis de datos y machine learning.

Variables del dataset:
step: Unidad de tiempo en el modelo de simulación (cada paso equivale a 1 hora).
type: Tipo de transacción (ej: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER).
amount: Monto de la transacción.
nameOrig: Identificador del cliente que inicia la transacción.
oldbalanceOrg: Balance de la cuenta de origen antes de la transacción.
newbalanceOrig: Balance de la cuenta de origen después de la transacción.
nameDest: Identificador del cliente que recibe la transacción.
oldbalanceDest: Balance de la cuenta destino antes de la transacción.
newbalanceDest: Balance de la cuenta destino después de la transacción.
isFraud: Variable objetivo → 1 si la transacción es fraudulenta, 0 en caso contrario.
isFlaggedFraud: Señal automática del sistema → 1 si el sistema marcó la transacción como sospechosa (ejemplo: transferencias superiores a 200,000).


