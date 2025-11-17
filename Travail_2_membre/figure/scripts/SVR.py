
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv("cleaned_building_data.csv")
X = df.drop(['Heating_Load', 'Cooling_Load'], axis=1)
y_heating = df['Heating_Load']


X_train, X_test, y_train, y_test = train_test_split(
    X, y_heating, test_size=0.2, random_state=42
)

print("Données prêtes :")
print("Taille train :", X_train.shape)
print("Taille test  :", X_test.shape)
print("-" * 50)

print(" Régression Linéaire en cours...")

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Régression Linéaire terminée !")
print(f"MSE : {mse_lr:.4f}")
print(f"MAE : {mae_lr:.4f}")
print(f"R²  : {r2_lr:.4f}")
print("-" * 50)


plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_lr, alpha=0.7, color='teal')
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Régression Linéaire – Heating Load")
plt.grid(True)
plt.show()


print("SVR (RBF Kernel) en cours...")

svr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=50, epsilon=0.1))
])

svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)

# Évaluation
mse_svr = mean_squared_error(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print("SVR terminé !")
print(f"MSE : {mse_svr:.4f}")
print(f"MAE : {mae_svr:.4f}")
print(f"R²  : {r2_svr:.4f}")
print("-" * 50)


plt.figure(figsize=(6,6))
plt.plot(y_test.values[:50], label='Réel', marker='o')
plt.plot(y_pred_svr[:50], label='SVR', marker='x')
plt.legend()
plt.title("Comparaison : SVR vs Réel (Heating Load)")
plt.show()


results = pd.DataFrame({
    'Modèle': ['Régression Linéaire', 'SVR (RBF)'],
    'MSE': [mse_lr, mse_svr],
    'MAE': [mae_lr, mae_svr],
    'R²': [r2_lr, r2_svr]
})

print("\n Résumé des performances :")
print(results)

sns.barplot(data=results, x='Modèle', y='R²', hue='Modèle', legend=False, palette='viridis')

plt.title("Comparaison des performances (R²)")
plt.show()


best_model = results.loc[results['R²'].idxmax(), 'Modèle']
print(f"\nLe meilleur modèle pour la prédiction du Heating Load est : {best_model}")
