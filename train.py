import tensorflow as tf
from tensorflow.keras import layers, models

print("--- Début de l'entraînement Keras (Fashion-MNIST) ---")

# 1. Charger un dataset Keras intégré (images de vêtements 28x28)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normaliser les pixels entre 0 et 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# 2. Construire un modèle simple (Réseau de neurones Dense)
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Entraîner le modèle (1 seule époque 'epochs=1' juste pour tester rapidement le flux)
print("Entraînement en cours sur le dataset...")
model.fit(X_train, y_train, epochs=1, batch_size=64)

# 4. Évaluer sur le jeu de test
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Modèle entraîné avec succès ! Précision sur le test : {test_acc * 100:.2f}%")
print("--- Fin du test Keras ---")