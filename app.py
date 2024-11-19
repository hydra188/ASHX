from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import torch
import jwt
from fastapi.security import OAuth2PasswordBearer
from mon_script import SimpleNN  # Importer la classe SimpleNN de mon_script.py

# Création de l'application FastAPI
app = FastAPI()

# Définir un modèle Pydantic pour la demande d'entrée avec validation
class InputData(BaseModel):
    feature1: float = Field(..., gt=0, lt=1)
    feature2: float = Field(..., gt=0, lt=1)
    feature3: float = Field(..., gt=0, lt=1)
    feature4: float = Field(..., gt=0, lt=1)
    feature5: float = Field(..., gt=0, lt=1)
    feature6: float = Field(..., gt=0, lt=1)
    feature7: float = Field(..., gt=0, lt=1)
    feature8: float = Field(..., gt=0, lt=1)
    feature9: float = Field(..., gt=0, lt=1)
    feature10: float = Field(..., gt=0, lt=1)

# Authentification avec OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Fonction pour récupérer l'utilisateur actuel à partir du jeton JWT
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, "secret-key", algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Token invalid")

# Charger ton modèle préalablement sauvegardé
model = SimpleNN()  # Créer une instance du modèle
model.load_state_dict(torch.load('mon_modele.pth', weights_only=True))  # Charger les poids
model.eval()  # Passer le modèle en mode évaluation

# Définir une route pour prédire avec le modèle
@app.post("/predict/")
async def predict(data: InputData, current_user: str = Depends(get_current_user)):
    # Transformer les données d'entrée en un format compatible avec ton modèle
    input_data = torch.tensor([[data.feature1, data.feature2, data.feature3, data.feature4, 
                                data.feature5, data.feature6, data.feature7, data.feature8,
                                data.feature9, data.feature10]], dtype=torch.float32)
    
    # Faire une prédiction avec le modèle
    with torch.no_grad():  # Pas besoin de calculer les gradients pour les prédictions
        output = model(input_data)
    
    # Traiter la sortie (classification binaire)
    prediction = (output > 0.5).float().item()  # Classe 1 si > 0.5, sinon classe 0
    
    # Retourner la prédiction et la probabilité associée
    return {
        "prediction": prediction,
        "probability": float(output.sigmoid().item())  # Probabilité associée à la classe 1
    }

# Pour tester une erreur (si besoin)
@app.get("/")
def read_root():
    return {"message": "API fonctionne correctement!"}
