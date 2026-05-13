# ============================================================
#  🦸 ACADEMIA S.H.I.E.L.D. DE MACHINE LEARNING 🦹
#  Clasificando el bien y el mal con Ciencia de Datos
# ============================================================
#
#  Misión: Los superhéroes necesitan un sistema automático para
#  identificar si alguien es héroe o villano usando sus atributos.
#  ¿Podrá el ML salvar el mundo? ¡Descúbrelo!
#
#  Algoritmos cubiertos:
#    1. Regresión Logística
#    2. Árbol de Decisión
#    3. Random Forest
#    4. SVM (Support Vector Machine)
#    5. K-NN (K-Nearest Neighbors)
#    6. Matriz de Confusión (evaluación final)
# ============================================================

# ── Instalación (descomenta si necesitas instalar) ──────────
# pip install scikit-learn pandas numpy matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ============================================================
#  DATASET: La Base de Datos Secreta de la Liga de Héroes
# ============================================================
# Características de cada individuo:
#   fuerza         (0-100): ¿Cuánto puede levantar?
#   inteligencia   (0-100): ¿Sabe hacer los impuestos?
#   carisma        (0-100): ¿Le siguen en redes?
#   nivel_ego      (0-100): ¿Habla de sí mismo en tercera persona?
#   mascotas_malas (0/1):   ¿Tiene tiburones con rayos láser?
#   capa           (0/1):   ¿Usa capa? (importante indicador)
#   etiqueta:  1 = Héroe   |   0 = Villano

np.random.seed(42)
n = 200

heroes = {
    "nombre": [
        "Spider-Man","Wonder Woman","Iron Man","Black Panther","Capitana Marvel",
        "Thor","Hulk","Black Widow","Hawkeye","Ant-Man","Doctor Strange",
        "Scarlet Witch","Visión","Falcon","War Machine","Shazam","Aquaman",
        "Flash","Green Lantern","Cyborg","Supergirl","Batgirl","Zatanna",
        "Ms. Marvel","Moon Knight","She-Hulk","America Chavez","Ironheart",
        "Kate Bishop","Yelena Belova"
    ] + [f"Héroe_Anónimo_{i}" for i in range(n//2 - 30)],
    "fuerza":       np.concatenate([
                        [75,90,70,85,95,98,100,60,55,50,65,85,80,65,75,88,92,70,68,78,94,62,60,70,72,88,85,80,58,62],
                        np.random.randint(55, 100, n//2 - 30)
                    ]),
    "inteligencia": np.concatenate([
                        [95,85,100,92,80,72,45,98,85,88,99,90,95,82,85,78,75,88,85,90,88,97,92,83,80,90,87,96,88,92],
                        np.random.randint(60, 100, n//2 - 30)
                    ]),
    "carisma":      np.concatenate([
                        [88,95,85,90,87,90,60,80,75,82,78,85,72,80,70,92,88,90,85,80,88,85,90,90,75,88,85,80,82,80],
                        np.random.randint(60, 100, n//2 - 30)
                    ]),
    "nivel_ego":    np.concatenate([
                        [40,30,75,25,35,60,20,15,10,20,45,30,10,15,20,35,40,30,25,20,25,15,20,20,30,30,20,30,15,15],
                        np.random.randint(5, 50, n//2 - 30)
                    ]),
    "mascotas_malas": np.concatenate([
                        [0]*30,
                        np.random.choice([0,1], n//2 - 30, p=[0.9, 0.1])
                    ]),
    "capa":         np.concatenate([
                        [0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,1,1,0,1,0,0,0,0,0],
                        np.random.choice([0,1], n//2 - 30, p=[0.5, 0.5])
                    ]),
    "etiqueta":     [1] * (n//2)
}

villanos = {
    "nombre": [
        "Thanos","Loki","Ultron","Red Skull","Magneto",
        "Joker","Lex Luthor","Brainiac","Reverse Flash","Sinestro",
        "Doctora Doom","Hela","Dormammu","Kang","M.O.D.O.K.",
        "Zemo","Yellowjacket","Justin Hammer","Whiplash","Abominación",
        "Venom (malo)","Mysterio","Vulture","Electro","Rhino",
        "Scorpion","Sandman","Lizard","Shocker","Hydro-Man"
    ] + [f"Villano_Anónimo_{i}" for i in range(n//2 - 30)],
    "fuerza":       np.concatenate([
                        [100,70,95,75,88,50,65,80,85,72,85,95,100,82,60,65,70,40,75,95,88,75,70,80,85,78,80,75,65,72],
                        np.random.randint(40, 95, n//2 - 30)
                    ]),
    "inteligencia": np.concatenate([
                        [90,95,98,85,95,88,100,100,92,85,99,80,85,98,72,88,80,75,70,45,60,90,78,65,50,60,55,80,65,55],
                        np.random.randint(40, 95, n//2 - 30)
                    ]),
    "carisma":      np.concatenate([
                        [70,90,60,50,75,95,85,55,65,60,70,80,50,75,40,80,55,70,60,40,55,75,70,60,50,55,50,55,50,45],
                        np.random.randint(30, 80, n//2 - 30)
                    ]),
    "nivel_ego":    np.concatenate([
                        [95,85,90,100,88,80,95,85,90,80,95,92,100,88,75,82,85,90,80,70,65,80,78,75,72,70,68,72,70,65],
                        np.random.randint(60, 100, n//2 - 30)
                    ]),
    "mascotas_malas": np.concatenate([
                        [1,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0],
                        np.random.choice([0,1], n//2 - 30, p=[0.4, 0.6])
                    ]),
    "capa":         np.concatenate([
                        [1,1,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        np.random.choice([0,1], n//2 - 30, p=[0.6, 0.4])
                    ]),
    "etiqueta":     [0] * (n//2)
}

df_heroes   = pd.DataFrame(heroes)
df_villanos = pd.DataFrame(villanos)
df = pd.concat([df_heroes, df_villanos], ignore_index=True).sample(frac=1, random_state=42)

print("=" * 60)
print("🦸  ACADEMIA S.H.I.E.L.D. DE MACHINE LEARNING  🦹")
print("=" * 60)
print(f"\n📋 Dataset: {len(df)} individuos registrados")
print(f"   Héroes:   {df['etiqueta'].sum()}")
print(f"   Villanos: {(df['etiqueta'] == 0).sum()}")
print("\n🔍 Muestra del dataset:")
print(df[["nombre","fuerza","inteligencia","carisma","nivel_ego","mascotas_malas","capa","etiqueta"]].head(8).to_string(index=False))

# ── Preparar datos ───────────────────────────────────────────
features = ["fuerza", "inteligencia", "carisma", "nivel_ego", "mascotas_malas", "capa"]
X = df[features].values
y = df["etiqueta"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

resultados = {}  # Guardamos resultados para la comparativa final


# ============================================================
# ░░░░░░░░░  EJERCICIO 1: REGRESIÓN LOGÍSTICA  ░░░░░░░░░░░░░
# ============================================================
#
#  🧪 MISIÓN: Loki está llenando solicitudes para unirse a los
#  Vengadores con diferentes identidades. S.H.I.E.L.D. necesita
#  un clasificador rápido que dé una PROBABILIDAD de si el
#  solicitante es héroe o un disfrazado de azul y verde.
#
#  ¿Cómo funciona?
#  La Regresión Logística aplica una función sigmoide a una
#  combinación lineal de las características para producir
#  una probabilidad entre 0 y 1.
#  P(héroe) = 1 / (1 + e^(-z))   donde z = w·x + b
# ============================================================

print("\n" + "=" * 60)
print("⚡ EJERCICIO 1: REGRESIÓN LOGÍSTICA")
print("   'El detector de Lokis infiltrados'")
print("=" * 60)

from sklearn.linear_model import LogisticRegression

# 👇 AQUÍ EMPIEZA TU CÓDIGO
modelo_lr = LogisticRegression(max_iter=1000, random_state=42)
modelo_lr.fit(X_train_sc, y_train)
y_pred_lr = modelo_lr.predict(X_test_sc)
proba_lr  = modelo_lr.predict_proba(X_test_sc)

acc_lr = accuracy_score(y_test, y_pred_lr)
resultados["Regresión Logística"] = acc_lr

print(f"\n✅ Precisión: {acc_lr:.2%}")
print("\n📊 Importancia de cada característica (coeficientes):")
for feat, coef in sorted(zip(features, modelo_lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
    barra = "█" * int(abs(coef) * 10)
    signo = "🟢" if coef > 0 else "🔴"
    print(f"   {signo} {feat:<20} {coef:+.3f}  {barra}")

# Predecir a Loki recién llegado
loki_disfrazado = np.array([[85, 95, 90, 85, 0, 1]])   # carisma alto, ego altísimo
loki_scaled = scaler.transform(loki_disfrazado)
prob_heroe = modelo_lr.predict_proba(loki_scaled)[0][1]
veredicto  = "HÉROE ✅" if prob_heroe > 0.5 else "VILLANO ⚠️  ¡Alerta Loki!"
print(f"\n🕵️  Nuevo solicitante sospechoso...")
print(f"   Probabilidad de ser héroe: {prob_heroe:.1%}")
print(f"   Veredicto: {veredicto}")

print("\n📌 PARA REFLEXIONAR:")
print("   - Un coeficiente positivo → esa característica empuja hacia 'héroe'")
print("   - Un coeficiente negativo → empuja hacia 'villano'")
print("   - ¿Qué característica es la más reveladora?")


# ============================================================
# ░░░░░░░░░░  EJERCICIO 2: ÁRBOL DE DECISIÓN  ░░░░░░░░░░░░░░
# ============================================================
#
#  🌳 MISIÓN: El Detective Batman necesita un sistema de reglas
#  EXPLÍCITO que cualquier comisario Gordon pueda seguir sin
#  un ordenador. "Si el ego es > 80 Y tiene mascota mala →
#  VILLANO". Árbol de Decisión al rescate.
#
#  ¿Cómo funciona?
#  Divide el espacio de características usando preguntas
#  binarias (nodos), maximizando la pureza de cada hoja
#  (Gini o Entropía). El resultado es un árbol de reglas
#  legibles por humanos.
# ============================================================

print("\n" + "=" * 60)
print("🌳 EJERCICIO 2: ÁRBOL DE DECISIÓN")
print("   'El manual de reglas del Detective Batman'")
print("=" * 60)

from sklearn.tree import DecisionTreeClassifier, export_text

# 👇 TU TURNO: prueba con distintos valores de max_depth (1-10)
modelo_dt = DecisionTreeClassifier(max_depth=4, random_state=42, criterion="gini")
modelo_dt.fit(X_train, y_train)   # sin escalar — los árboles no lo necesitan
y_pred_dt = modelo_dt.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
resultados["Árbol de Decisión"] = acc_dt

print(f"\n✅ Precisión: {acc_dt:.2%}")
print(f"   Profundidad real del árbol: {modelo_dt.get_depth()}")
print(f"   Número de hojas: {modelo_dt.get_n_leaves()}")

print("\n🌲 Reglas del árbol (manual del Detective Batman):")
reglas = export_text(modelo_dt, feature_names=features)
# Mostrar solo las primeras líneas para no saturar la terminal
for i, linea in enumerate(reglas.split("\n")[:25]):
    print("  ", linea)
if reglas.count("\n") > 25:
    print("   ... (árbol completo disponible en 'reglas' variable)")

print("\n🏆 Importancia de características:")
for feat, imp in sorted(zip(features, modelo_dt.feature_importances_), key=lambda x: x[1], reverse=True):
    barra = "█" * int(imp * 40)
    print(f"   {feat:<20} {imp:.3f}  {barra}")

print("\n📌 PARA REFLEXIONAR:")
print("   - ¿Qué pasa si aumentas max_depth=10? (sobreajuste)")
print("   - ¿Y con max_depth=1? (subajuste)")
print("   - ¿Puedes leer el árbol y entender la lógica?")


# ============================================================
# ░░░░░░░░░░░░  EJERCICIO 3: RANDOM FOREST  ░░░░░░░░░░░░░░░░
# ============================================================
#
#  🌲🌲🌲 MISIÓN: Un solo árbol puede ser engañado por Mystique
#  (la mutante que cambia de forma). Necesitamos un COMITÉ de
#  árboles que vote en conjunto. ¡El poder de la democracia
#  mutante!
#
#  ¿Cómo funciona?
#  Entrena N árboles de decisión, cada uno con una muestra
#  aleatoria del dataset (bagging) y características aleatorias.
#  La predicción final es la votación mayoritaria de todos.
#  → Menos varianza, más robusto que un solo árbol.
# ============================================================

print("\n" + "=" * 60)
print("🌲🌲 EJERCICIO 3: RANDOM FOREST")
print("   'El comité anti-Mystique'")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier

# 👇 TU TURNO: experimenta con n_estimators (10, 50, 100, 500)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)
y_pred_rf = modelo_rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
resultados["Random Forest"] = acc_rf

print(f"\n✅ Precisión: {acc_rf:.2%}")
print(f"   Árboles en el bosque: {modelo_rf.n_estimators}")

print("\n🔮 Importancia de características (consenso del bosque):")
importancias = sorted(zip(features, modelo_rf.feature_importances_), key=lambda x: x[1], reverse=True)
for feat, imp in importancias:
    barra = "█" * int(imp * 50)
    print(f"   {feat:<20} {imp:.3f}  {barra}")

# Simulación de votación del bosque para un caso concreto
caso_dudoso = np.array([[72, 80, 75, 72, 1, 1]])  # ¿Magneto rehabilitado?
votos = np.array([tree.predict(caso_dudoso)[0] for tree in modelo_rf.estimators_])
votos_heroe   = votos.sum()
votos_villano = len(votos) - votos_heroe
print(f"\n⚖️  Caso 'Magneto pide asilo heroico':")
print(f"   Árboles que votan HÉROE:   {votos_heroe}")
print(f"   Árboles que votan VILLANO: {votos_villano}")
print(f"   Veredicto del bosque: {'HÉROE ✅' if votos_heroe > votos_villano else 'VILLANO ⚠️'}")

print("\n📌 PARA REFLEXIONAR:")
print("   - ¿Por qué Random Forest suele superar a un solo árbol?")
print("   - ¿Qué ventaja tiene la votación sobre una sola predicción?")


# ============================================================
# ░░░░░░░░░░░░░░░░  EJERCICIO 4: SVM  ░░░░░░░░░░░░░░░░░░░░░
# ============================================================
#
#  🔮 MISIÓN: Professor X necesita un campo de fuerza mental
#  que separe PERFECTAMENTE a mutantes buenos y malos en el
#  espacio multidimensional. La SVM encontrará el hiperplano
#  que maximiza la distancia entre clases. ¡Como la Cámara
#  de Privacidad, pero matemática!
#
#  ¿Cómo funciona?
#  Encuentra el hiperplano de mayor margen entre clases.
#  Con kernel RBF puede separar datos no lineales proyectándolos
#  a dimensiones superiores (¡dimensiones cuánticas!).
#  Parámetros clave: C (tolerancia errores) y gamma (influencia
#  de cada punto).
# ============================================================

print("\n" + "=" * 60)
print("🔮 EJERCICIO 4: SVM (Support Vector Machine)")
print("   'El campo de fuerza mental del Profesor X'")
print("=" * 60)

from sklearn.svm import SVC

# 👇 TU TURNO: prueba kernel='linear', 'poly', 'rbf', 'sigmoid'
modelo_svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
modelo_svm.fit(X_train_sc, y_train)
y_pred_svm = modelo_svm.predict(X_test_sc)

acc_svm = accuracy_score(y_test, y_pred_svm)
resultados["SVM"] = acc_svm

print(f"\n✅ Precisión: {acc_svm:.2%}")
print(f"   Kernel usado: {modelo_svm.kernel}")
print(f"   Vectores de soporte (los casos más difíciles de clasificar): {modelo_svm.n_support_}")
print(f"     → Héroes en el límite:   {modelo_svm.n_support_[1]}")
print(f"     → Villanos en el límite: {modelo_svm.n_support_[0]}")

# Caso extremo: Deadpool (¿héroe? ¿villano? ¿mercenario carismático?)
deadpool = np.array([[80, 78, 99, 88, 0, 0]])  # ego alto pero sin mascota mala
deadpool_sc = scaler.transform(deadpool)
prob_dp = modelo_svm.predict_proba(deadpool_sc)[0]
print(f"\n🎭 ¿Dónde queda Deadpool en el hiperplano?")
print(f"   Probabilidad héroe:   {prob_dp[1]:.1%}")
print(f"   Probabilidad villano: {prob_dp[0]:.1%}")
print(f"   La SVM dice: {'HÉROE ✅' if prob_dp[1] > 0.5 else 'VILLANO ⚠️'}")
print(f"   (Deadpool ha tuiteado que no está de acuerdo)")

print("\n📌 PARA REFLEXIONAR:")
print("   - kernel='linear' → frontera recta (simple)")
print("   - kernel='rbf'    → frontera curva (más flexible)")
print("   - Aumentar C → más ajustado al train (¿sobreajuste?)")
print("   - ¿Qué pasa con C=0.01 vs C=100?")


# ============================================================
# ░░░░░░░░░░░░░░░  EJERCICIO 5: K-NN  ░░░░░░░░░░░░░░░░░░░░░
# ============================================================
#
#  🗺️ MISIÓN: Groot solo dice "I am Groot", pero sus VECINOS
#  más cercanos (Rocket, Gamora, Star-Lord) pueden explicar
#  si alguien nuevo es Guardian de la Galaxia o agente de Thanos.
#  K-NN usa la sabiduría de los vecinos más similares.
#
#  ¿Cómo funciona?
#  Para clasificar a un nuevo individuo, busca los K más
#  similares (distancia euclidiana u otra) y hace una votación.
#  No hay entrenamiento real → es un clasificador "perezoso".
#  ⚠️ Necesita escalado: si fuerza va de 0-100 y capa de 0-1,
#     la distancia estaría sesgada.
# ============================================================

print("\n" + "=" * 60)
print("🗺️  EJERCICIO 5: K-NN (K-Nearest Neighbors)")
print("   'Dime quiénes son tus vecinos y te diré si eres Guardián'")
print("=" * 60)

from sklearn.neighbors import KNeighborsClassifier

# 👇 TU TURNO: experimenta con k = 1, 3, 5, 11, 21, 51
k = 5
modelo_knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
modelo_knn.fit(X_train_sc, y_train)
y_pred_knn = modelo_knn.predict(X_test_sc)

acc_knn = accuracy_score(y_test, y_pred_knn)
resultados["K-NN"] = acc_knn

print(f"\n✅ Precisión con K={k}: {acc_knn:.2%}")

# Buscar el K óptimo (mini experimento)
print("\n🔬 Buscando el K óptimo (experimento rápido):")
ks = [1, 3, 5, 7, 9, 11, 15, 21]
accs_k = []
for ki in ks:
    m = KNeighborsClassifier(n_neighbors=ki)
    m.fit(X_train_sc, y_train)
    accs_k.append(accuracy_score(y_test, m.predict(X_test_sc)))

for ki, ac in zip(ks, accs_k):
    barra = "█" * int(ac * 40)
    marca = " ← óptimo" if ac == max(accs_k) else ""
    print(f"   K={ki:>2}  acc={ac:.2%}  {barra}{marca}")

# Groot presenta a un nuevo ser
nuevo_ser = np.array([[68, 82, 78, 55, 0, 0]])
nuevo_sc  = scaler.transform(nuevo_ser)
vecinos_dist, vecinos_idx = modelo_knn.kneighbors(nuevo_sc)
etiquetas_vecinos = y_train[vecinos_idx[0]]

print(f"\n🌿 Groot trae a un desconocido para evaluación...")
print(f"   Sus {k} vecinos más cercanos votan:")
for i, (dist, etq) in enumerate(zip(vecinos_dist[0], etiquetas_vecinos)):
    rol = "Héroe ✅" if etq == 1 else "Villano ⚠️"
    print(f"     Vecino {i+1}: {rol}  (distancia: {dist:.2f})")
pred_nuevo = modelo_knn.predict(nuevo_sc)[0]
print(f"   → Veredicto final: {'HÉROE ✅' if pred_nuevo == 1 else 'VILLANO ⚠️'}")
print(f"   Groot: 'I am Groot.' (traducción: correcto)")

print("\n📌 PARA REFLEXIONAR:")
print("   - K=1 → memoriza exactamente el train (sobreajuste)")
print("   - K muy grande → ignora patrones locales (subajuste)")
print("   - ¿Por qué el escalado es OBLIGATORIO aquí?")


# ============================================================
# ░░░░░░░░░░░░  EJERCICIO 6: MATRIZ DE CONFUSIÓN  ░░░░░░░░░
# ============================================================
#
#  🎯 MISIÓN FINAL: Nick Fury necesita un informe completo.
#  No solo "¿cuántos acertamos?", sino:
#  - ¿A cuántos villanos dejamos pasar haciéndose pasar por héroes?
#    (Falsos Positivos → ¡ERROR CRÍTICO!)
#  - ¿A cuántos héroes acusamos injustamente de villanos?
#    (Falsos Negativos → también problemático)
#
#  La Matriz de Confusión muestra los 4 escenarios posibles.
# ============================================================

print("\n" + "=" * 60)
print("🎯 EJERCICIO 6: MATRIZ DE CONFUSIÓN")
print("   'El informe final de Nick Fury'")
print("=" * 60)

# Usamos Random Forest para el análisis (el mejor modelo)
cm = confusion_matrix(y_test, y_pred_rf)
tn, fp, fn, tp = cm.ravel()

print(f"""
  ┌─────────────────────────────────────────────────────┐
  │           MATRIZ DE CONFUSIÓN  (Random Forest)      │
  │                                                     │
  │              Predicción Héroe  Predicción Villano   │
  │  Real Héroe      TP={tp:<4}  ✅         FN={fn:<4}  ❌     │
  │  Real Villano    FP={fp:<4}  ⚠️          TN={tn:<4}  ✅     │
  └─────────────────────────────────────────────────────┘

  ✅ Verdaderos Positivos (TP={tp}): Héroes correctamente identificados
  ✅ Verdaderos Negativos (TN={tn}): Villanos correctamente detectados
  ❌ Falsos Positivos    (FP={fp}): Villanos que pasaron como héroes ← PELIGROSO
  ❌ Falsos Negativos    (FN={fn}): Héroes acusados injustamente ← INJUSTO
""")

# Métricas derivadas
precision  = tp / (tp + fp) if (tp + fp) > 0 else 0
recall     = tp / (tp + fn) if (tp + fn) > 0 else 0
f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
especific  = tn / (tn + fp) if (tn + fp) > 0 else 0

print("📊 Métricas clave:")
print(f"   Accuracy    = (TP+TN)/Total = {(tp+tn)/(tp+tn+fp+fn):.2%}  ← % global de aciertos")
print(f"   Precision   = TP/(TP+FP)   = {precision:.2%}  ← de los que llamamos héroe, ¿cuántos lo son?")
print(f"   Recall      = TP/(TP+FN)   = {recall:.2%}  ← de todos los héroes, ¿cuántos encontramos?")
print(f"   F1-Score    = 2·P·R/(P+R)  = {f1:.2%}  ← balance precision-recall")
print(f"   Especific.  = TN/(TN+FP)   = {especific:.2%}  ← de todos los villanos, ¿cuántos detectamos?")

print("\n⚠️  DILEMA DE NICK FURY:")
print("   Si priorizas RECALL  → no te pierdes ningún héroe (pero acusas a inocentes)")
print("   Si priorizas PRECISION → no acusas inocentes (pero dejas escapar villanos)")
print("   F1-Score busca el equilibrio. ¿Cuál priorizarías tú?")

# Visualización
print("\n🎨 Generando visualización de la matriz de confusión...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d0d1a")

# Gráfico 1: Matriz de confusión
ax = axes[0]
ax.set_facecolor("#0d0d1a")
labels = np.array([[f"Héroes\ncorrectos\n✅ {tp}", f"Héroes\nperdidos\n❌ {fn}"],
                   [f"Villanos\ninfiltr.\n⚠️ {fp}",  f"Villanos\ndetect.\n✅ {tn}"]])
colores = np.array([[tp, -fn], [-fp, tn]])  # positivo=bueno, negativo=malo
im = ax.imshow([[tp, fn], [fp, tn]], cmap="RdYlGn", vmin=0, vmax=max(tp, tn)+5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, labels[i, j], ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred: HÉROE", "Pred: VILLANO"], color="white", fontsize=10)
ax.set_yticklabels(["Real: HÉROE", "Real: VILLANO"], color="white", fontsize=10)
ax.set_title("🎯 Matriz de Confusión\n(Random Forest)", color="white", fontsize=12, pad=10)
ax.tick_params(colors="white")
for spine in ax.spines.values(): spine.set_edgecolor("#333")

# Gráfico 2: Comparativa de todos los modelos
ax2 = axes[1]
ax2.set_facecolor("#0d0d1a")
modelos  = list(resultados.keys())
accs_all = list(resultados.values())
colores_bar = ["#00d4ff", "#7fff00", "#ff6b35", "#e040fb", "#ffd700"]
bars = ax2.barh(modelos, accs_all, color=colores_bar, edgecolor="white", linewidth=0.5)
ax2.set_xlim(0.5, 1.0)
ax2.set_xlabel("Accuracy", color="white", fontsize=10)
ax2.set_title("🏆 Comparativa de Modelos\n¿Quién salva mejor el mundo?", color="white", fontsize=12, pad=10)
ax2.tick_params(colors="white")
for spine in ax2.spines.values(): spine.set_edgecolor("#333")
ax2.set_facecolor("#0d0d1a")
for bar, acc in zip(bars, accs_all):
    ax2.text(acc + 0.005, bar.get_y() + bar.get_height()/2,
             f"{acc:.1%}", va="center", color="white", fontsize=10, fontweight="bold")

plt.suptitle("🦸 ACADEMIA S.H.I.E.L.D. DE MACHINE LEARNING 🦹",
             color="white", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("shield_ml_report.png", dpi=150, bbox_inches="tight",
            facecolor="#0d0d1a", edgecolor="none")
print("   Guardado como 'shield_ml_report.png'")
plt.show()


# ============================================================
#  🏆  INFORME FINAL DE NICK FURY
# ============================================================
print("\n" + "=" * 60)
print("🏆 INFORME FINAL DE NICK FURY")
print("=" * 60)
print("\nResultados de la operación de clasificación:\n")
ganador = max(resultados, key=resultados.get)
for modelo, acc in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
    medalla = "🥇" if modelo == ganador else "  "
    barra = "█" * int(acc * 30)
    print(f"  {medalla} {modelo:<22} {acc:.2%}  {barra}")

print(f"""
  🎖️  Ganador: {ganador} ({resultados[ganador]:.2%})

  Conclusión de Nick Fury:
  "Con estos modelos hemos protegido el mundo de infiltrados.
   Pero recuerden: ningún algoritmo es perfecto.
   Siempre habrá un Loki que se nos cuele..."

  ─────────────────────────────────────────────
  📚 RETOS ADICIONALES PARA PADAWANS DEL ML:
  ─────────────────────────────────────────────
  1. Añade más características: ¿tiene base secreta? ¿traicionó aliados?
  2. Prueba GridSearchCV para encontrar los mejores hiperparámetros
  3. Implementa validación cruzada (cross_val_score) en todos los modelos
  4. ¿Qué pasa si el dataset está desbalanceado (90% héroes)?
  5. Añade un héroe real tuyo y clasifícalo con todos los modelos

  ¡La misión continúa, Agente! 🦸‍♀️
""")