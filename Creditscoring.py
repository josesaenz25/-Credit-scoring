# -*- coding: utf-8 -*-
"""
Pipeline de Credit Scoring con exportación de métricas a JSON
"""

import pandas as pd
import numpy as np
import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---------------------------------------------------
# 1. Generar dataset sintético con variables numéricas y categóricas
# ---------------------------------------------------
X, y = make_classification(
    n_samples=200,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    n_classes=2,
    weights=[0.7, 0.3],
    random_state=42
)

data = pd.DataFrame(X, columns=[
    "age", "income", "loan_amount", "loan_term",
    "credit_history_length", "past_due"
])
data["default"] = y

# Variables categóricas simuladas
np.random.seed(42)
data["gender"] = np.random.choice(["Male", "Female"], size=len(data))
data["employment_status"] = np.random.choice(["Employed", "Unemployed", "Self-employed"], size=len(data))
data["marital_status"] = np.random.choice(["Single", "Married", "Divorced"], size=len(data))

print("✅ Dataset sintético generado con variables categóricas")

# ---------------------------------------------------
# 2. Preparación de datos
# ---------------------------------------------------
X = data.drop("default", axis=1)
y = data["default"]

cat_cols = ["gender", "employment_status", "marital_status"]
num_cols = [c for c in X.columns if c not in cat_cols]

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Preprocesamiento
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# ---------------------------------------------------
# 3. Modelos
# ---------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": Pipeline([
        ("preprocess", preprocess),
        ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

# ---------------------------------------------------
# 4. Entrenamiento y evaluación
# ---------------------------------------------------
results = {}
cv_results = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métricas
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_proba)
    }

    # Validación cruzada
    scores = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=cv)
    cv_results[name] = (scores.mean(), scores.std())

    # Mostrar resultados en consola
    print(f"\n==== {name} ====")
    for metric, value in results[name].items():
        print(f"{metric}: {value:.4f}")
    print(f"CV ROC-AUC: {cv_results[name][0]:.4f} ± {cv_results[name][1]:.4f}")

# ---------------------------------------------------
# 5. Exportar métricas a JSON
# ---------------------------------------------------
with open("metrics.json", "w") as f:
    json.dump({"results": results, "cv_results": cv_results}, f, indent=4)

print("\n✅ Métricas guardadas en metrics.json")







# ---------------------------------------------------
# Medidor de Puntaje Crediticio con colores de igual tamaño angular
# ---------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# Entrenar modelo Random Forest
rf_model = models["Random Forest"]
rf_model.fit(X_train, y_train)

# Escala de puntaje
min_score, max_score = 300, 850
credit_scores = rf_model.predict_proba(X_test)[:, 1]
scaled_scores = min_score + (max_score - min_score) * credit_scores
avg_score = np.mean(scaled_scores)

# Crear gráfico semicircular
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': 'polar'})
ax.set_theta_offset(0)              # orientación hacia arriba
ax.set_theta_direction(1)           # sentido antihorario
ax.set_facecolor('none')            # fondo transparente
ax.grid(False)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_ylim(0, 12)

# Definir zonas con colores iguales en tamaño angular
equal_angles = np.linspace(0, np.pi, 5)  # 4 zonas = 5 divisiones
zones = [
    (equal_angles[0], equal_angles[1], '#B71C1C', 'MALO', 300, 629),
    (equal_angles[1], equal_angles[2], '#E65100', 'REGULAR', 630, 689),
    (equal_angles[2], equal_angles[3], '#AED581', 'BUENO', 690, 719),
    (equal_angles[3], equal_angles[4], '#1B5E20', 'EXCELENTE', 720, 850)
]

# Dibujar cada zona como una barra continua
for theta1, theta2, color, label, start, end in zones:
    ax.bar(
        x=np.linspace(theta1, theta2, 100),
        height=12,
        width=(theta2 - theta1) / 100,
        bottom=0,
        color=color,
        edgecolor='none',
        linewidth=0,
        alpha=1.0
    )

    # Etiqueta interna en forma semicircular
    angle_label = (theta1 + theta2) / 2
    ax.text(
        angle_label, 8.5, f'{label}\n{start}–{end}',
        ha='center', va='center', fontsize=10, color='white', weight='bold',
        rotation=np.degrees(angle_label - np.pi/2),
        rotation_mode='anchor'
    )

# Aguja del puntaje promedio
needle_angle = (avg_score - min_score) / (max_score - min_score) * np.pi
ax.plot([needle_angle, needle_angle], [0, 12], color='black', linewidth=4)
ax.plot(needle_angle, 12, 'o', color='black', markersize=9)

# Etiqueta del valor promedio
ax.text(
    0, -2, f'Puntaje promedio: {avg_score:.0f}',
    ha='center', fontsize=14, weight='bold',
    bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.35', alpha=0.95)
)

# Título
plt.title("Medidor de Puntaje Crediticio (Random Forest)", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig("medidor_puntaje_crediticio.png", dpi=200)
plt.close()

print("✅ Medidor de puntaje crediticio generado correctamente como medidor_puntaje_crediticio.png")
