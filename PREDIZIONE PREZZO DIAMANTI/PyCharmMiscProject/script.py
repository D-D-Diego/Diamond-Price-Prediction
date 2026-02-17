import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor # <--- Aggiunto TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Caricamento dei dati
try:
    df = pd.read_csv('diamond.csv')
    print("Dataset caricato con successo!")
except FileNotFoundError:
    print("Errore: Assicurati che 'diamond.csv' sia nella cartella.")
    exit()

#Mappatura
cut_order = ['Fair', 'Good', 'Very Good', 'Ideal', 'Signature-Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL']

#Definizione delle Colonne
ordinal_features = ['Cut', 'Color', 'Clarity']
nominal_features = ['Polish', 'Symmetry', 'Report']
numerical_features = ['Carat Weight']

#Preprocessing e trasformazione delle colonne
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(categories=[cut_order, color_order, clarity_order],
                               handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features),
        ('nom', OneHotEncoder(handle_unknown='ignore'), nominal_features)
    ],
    remainder='passthrough'
)

#Pipeline con Trasformazione Logaritmica del Target Value
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definiamo il modello che avvolge il Random Forest con il Logaritmo
model_log = TransformedTargetRegressor(
    regressor=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    func=np.log1p,          # Applica log(1+x) in addestramento
    inverse_func=np.expm1   # Riporta a dollari (exp(x)-1) in predizione
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model_log)
])

print("Addestramento con Log-Transformation in corso...")
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

#Calcolo delle metriche di valutazione
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*40)
print(f"   REPORT PRESTAZIONI (LOG-TRANSFORM)")
print("="*40)
print(f"R^2 Score (Precisione): {r2:.4f}")
print(f"MAE (Errore medio):    ${mae:.2f}")
print(f"RMSE (Penalità outlier): ${rmse:.2f}")
print("="*40)

#Generazione dei grafici
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.3)

sns.histplot(df['Price'], kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('1. Distribuzione dei Prezzi (Target)')

sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=axes[0, 1], color='teal')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
axes[0, 1].set_title('2. Prezzi Reali vs Predetti (Con Log-Transform)')

residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, alpha=0.4, ax=axes[1, 0], color='coral')
axes[1, 0].axhline(0, color='black', linestyle='--')
axes[1, 0].set_title('3. Analisi dei Residui')


importances = model_pipeline.named_steps['regressor'].regressor_.feature_importances_
ohe_names = model_pipeline.named_steps['preprocessor'].named_transformers_['nom'].get_feature_names_out(nominal_features)
all_feat_names = ordinal_features + list(ohe_names) + numerical_features
feat_imp_series = pd.Series(importances, index=all_feat_names)

feat_imp_series.nlargest(10).sort_values().plot(kind='barh', ax=axes[1, 1], color='darkgreen')
axes[1, 1].set_title('4. Top 10 Fattori Influenti')

plt.show()