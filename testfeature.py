import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split

# Exemple : remplacer par ton vrai dataset
# X = pd.read_csv("features.csv")
# y = pd.read_csv("target.csv").values.ravel()

# Pour l'exemple minimal :
import numpy as np
np.random.seed(42)
X = pd.DataFrame(np.random.rand(37000, 18), columns=[f'feat_{i}' for i in range(18)])
y = np.random.choice(np.arange(0, 160, 5), size=37000)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraînement d'un modèle XGBoost pour régression
model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)

# Explications SHAP (TreeExplainer optimisé pour les arbres)
explainer = shap.TreeExplainer(model)
# Pour accélérer : échantillonner si dataset trop grand
X_sample = X_test.sample(5000, random_state=42)
shap_values = explainer(X_sample)

# Importance globale
shap.summary_plot(shap_values, X_sample)

