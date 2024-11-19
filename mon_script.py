import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

# Définition du modèle simple (par exemple un MLP)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 64)  # Entrée avec 10 features
        self.fc2 = nn.Linear(64, 1)   # Sortie binaire
        self.sigmoid = nn.Sigmoid()   # Activation sigmoid pour la classification binaire

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

# Fonction d'entraînement
def train(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Remise à zéro des gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Rétropropagation
            optimizer.step()  # Mise à jour des poids

            running_loss += loss.item()

        print(f"Époque {epoch}, Perte entraînement : {running_loss / len(train_loader):.4f}")

# Fonction d'évaluation
def evaluation(model, test_loader):
    model.eval()  # Passage en mode évaluation
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()  # Classification binaire
            y_true.extend(labels.numpy())
            y_pred.extend(predictions.numpy())

    # Calcul des métriques
    accuracy = (np.array(y_pred) == np.array(y_true)).mean()
    print(f"Accuracy sur les données de test: {accuracy * 100:.2f}%")

    # Rapport de classification
    print("Rapport de classification :")
    print(classification_report(y_true, y_pred))

# Exemple d'utilisation
if __name__ == "__main__":
    # Génération de données fictives pour l'exemple
    num_samples = 1000
    X_train = np.random.randn(num_samples, 10)  # 1000 échantillons, 10 features
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(float)  # Cible binaire
    X_test = np.random.randn(200, 10)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(float)

    # Conversion en tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Création des DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialisation du modèle, du critère et de l'optimiseur
    model = SimpleNN()
    criterion = nn.BCELoss()  # Pour une classification binaire
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entraînement
    train(model, train_loader, criterion, optimizer, epochs=100)

    # Évaluation sur les données de test
    evaluation(model, test_loader)

    # Sauvegarder le modèle entraîné
    torch.save(model.state_dict(), 'mon_modele.pth')
    print("Modèle sauvegardé sous 'mon_modele.pth'")

    # --- Partie ajoutée pour charger le modèle et faire des prédictions ---
    
    # Recréer la même architecture du modèle
    model = SimpleNN()

    # Charger les poids du modèle sauvegardé
    model.load_state_dict(torch.load('mon_modele.pth', weights_only=True))


    # Passer le modèle en mode évaluation
    model.eval()

    # Exemple de nouvelles données (10 échantillons, 10 features)
    X_new = np.random.randn(10, 10)  # 10 nouveaux échantillons avec 10 features

    # Convertir les nouvelles données en Tensor
    X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

    # Faire des prédictions
    with torch.no_grad():  # Pas besoin de calculer les gradients pour les prédictions
        predictions = model(X_new_tensor)
        predicted_classes = (predictions > 0.5).float()  # Classe 1 si > 0.5, sinon classe 0

    # Afficher les résultats
    print("Prédictions sur les nouvelles données :", predicted_classes.numpy())
