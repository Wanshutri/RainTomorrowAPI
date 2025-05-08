from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from fastapi.middleware.cors import CORSMiddleware

# Cargar modelos y scalers previamente entrenados
best_logreg_tropical = pickle.load(open("models/best_logreg_tropical.pkl", "rb"))
best_logreg_subtropical = pickle.load(open("models/best_logreg_subtropical.pkl", "rb"))
best_logreg_desierto = pickle.load(open("models/best_logreg_desierto.pkl", "rb"))
best_logreg_llano = pickle.load(open("models/best_logreg_llano.pkl", "rb"))
best_logreg_templado = pickle.load(open("models/best_logreg_templado.pkl", "rb"))

tropical_scaler = pickle.load(open("scalers/tropical_scaler.pkl", "rb"))
subtropical_scaler = pickle.load(open("scalers/subtropical_scaler.pkl", "rb"))
desierto_scaler = pickle.load(open("scalers/desierto_scaler.pkl", "rb"))
llano_scaler = pickle.load(open("scalers/llano_scaler.pkl", "rb"))
templado_scaler = pickle.load(open("scalers/templado_scaler.pkl", "rb"))

# Diccionarios
location_to_region = {
    'Albury': 'Llano', 'BadgerysCreek': 'Subtropical', 'Cobar': 'Desierto', 'CoffsHarbour': 'Subtropical',
    'Moree': 'Llano', 'Newcastle': 'Subtropical', 'NorahHead': 'Subtropical', 'NorfolkIsland': 'Subtropical',
    'Penrith': 'Subtropical', 'Richmond': 'Subtropical', 'Sydney': 'Subtropical', 'SydneyAirport': 'Subtropical',
    'WaggaWagga': 'Llano', 'Williamtown': 'Subtropical', 'Wollongong': 'Subtropical', 'Canberra': 'Llano',
    'Tuggeranong': 'Llano', 'MountGinini': 'Llano','Ballarat': 'Llano','Bendigo': 'Llano','Sale': 'Templado',
    'MelbourneAirport': 'Templado','Melbourne': 'Templado','Mildura': 'Llano','Nhil': 'Llano','Portland': 'Templado',
    'Watsonia': 'Templado','Dartmoor': 'Templado','Brisbane': 'Subtropical','Cairns': 'Tropical','GoldCoast': 'Subtropical',
    'Townsville': 'Tropical','Adelaide': 'Templado','MountGambier': 'Templado','Nuriootpa': 'Templado','Woomera': 'Desierto',
    'Albany': 'Templado','Witchcliffe': 'Templado','PearceRAAF': 'Llano','PerthAirport': 'Llano','Perth': 'Llano',
    'SalmonGums': 'Llano','Walpole': 'Templado','Hobart': 'Templado','Launceston': 'Templado','AliceSprings': 'Desierto',
    'Darwin': 'Tropical','Katherine': 'Tropical','Uluru': 'Desierto'
}

region_to_model = {
    'Tropical': best_logreg_tropical,
    'Subtropical': best_logreg_subtropical,
    'Desierto': best_logreg_desierto,
    'Llano': best_logreg_llano,
    'Templado': best_logreg_templado
}

region_to_scaler = {
    'Tropical': tropical_scaler,
    'Subtropical': subtropical_scaler,
    'Desierto': desierto_scaler,
    'Llano': llano_scaler,
    'Templado': templado_scaler
}

# Pydantic model para entrada
class InputData(BaseModel):
    Location: str
    Humidity3pm: float
    RISK_MM: float
    RainToday: int
    Cloud3pm: float
    Sunshine: float
    Pressure3pm: float
    Temp3pm: float

app = FastAPI()

# Orígenes permitidos
origins = [
    "http://localhost:3000",   # Para desarrollo local con Next.js
    "https://rain-tomorrow-front.vercel.app",     # Cambia esto por tu dominio real
]

# Agregar middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # o usa ["*"] para permitir todos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(data: InputData):
    try:
        location = data.Location
        region = location_to_region.get(location)

        if not region:
            return {"error": f"Ubicación '{location}' no encontrada en el diccionario de regiones."}

        model = region_to_model.get(region)
        scaler = region_to_scaler.get(region)

        if not model or not scaler:
            return {"error": f"No hay modelo/escalador asociado a la región '{region}'."}

        # Crear DataFrame con los datos
        features_df = pd.DataFrame([{
            'Humidity3pm': data.Humidity3pm,
            'RISK_MM': data.RISK_MM,
            'RainToday': data.RainToday,
            'Cloud3pm': data.Cloud3pm,
            'Sunshine': data.Sunshine,
            'Pressure3pm': data.Pressure3pm,
            'Temp3pm': data.Temp3pm
        }])

        # Escalar y predecir
        scaled = scaler.transform(features_df)
        prediction = model.predict(scaled)[0]

        return {"prediction": int(prediction), "region": region}
    
    except Exception as e:
        return {"error": str(e)}
