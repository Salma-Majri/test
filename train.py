from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print("--- Début du test d'IA ---")

# 1. Charger un petit dataset simple (les fleurs d'Iris)
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# 2. Entraîner un modèle simple (Régression Logistique)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 3. Tester et afficher le résultat
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print(f"Modèle entraîné avec succès ! Précision : {acc * 100:.2f}%")
print("--- Fin du test ---")