from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sqlalchemy.orm import Session
from . import database, model as db_model

# Загрузка модели
import os
from pathlib import Path

# Определяем путь к текущей папке (app/)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "insurance_model.joblib"

# Загрузка модели
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Файл модели не найден по пути: {MODEL_PATH}")

app = FastAPI(title="Medical Insurance Prediction")

# Схема входных данных
class InsuranceInput(BaseModel):
    age: int
    sex: str      # "male", "female"
    bmi: float
    children: int
    smoker: str   # "yes", "no"
    region: str   # "southwest", "southeast", "northwest", "northeast"

# Инициализация БД
database.Base.metadata.create_all(bind=database.engine)

@app.post("/predict")
def predict(data: InsuranceInput, db: Session = Depends(database.get_db)):
    try:
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])
        
        # 1. Сначала превращаем "yes"/"no" в 1/0
        mapping = {
            "sex": {"female": 0, "male": 1},
            "smoker": {"no": 0, "yes": 1},
            "region": {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
        }
        for col, m in mapping.items():
            if col in df.columns:
                df[col] = df[col].map(m)

        # 2. Генерация признаков (ВАЖЕН ПОРЯДОК!)
        # Сначала создаем is_obese
        df['is_obese'] = (df['bmi'] >= 30).astype(int)
        # Потом создаем age_smoker_interaction
        df['age_smoker_interaction'] = df['age'] * df['smoker']

        # 3. Указываем ПРАВИЛЬНЫЙ порядок колонок для модели
        expected_columns = [
            "age", 
            "sex", 
            "bmi", 
            "children", 
            "smoker", 
            "region", 
            "is_obese",               # Шел 7-м в prepare_dataset.py
            "age_smoker_interaction"   # Шел 8-м в prepare_dataset.py
        ]
        df = df[expected_columns]

        # 4. Предсказание
        prediction = model.predict(df)[0]

        # 5. Сохранение в БД
        db_prediction = db_model.PredictionRecord(
            age=data.age, 
            bmi=data.bmi, 
            smoker=data.smoker, 
            prediction=float(prediction)
        )
        db.add(db_prediction)
        db.commit()

        return {"prediction": round(float(prediction), 2)}
    
    except Exception as e:
        print(f"\n[DEBUG ERROR]: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/health")
def health():
    return {"status": "ok"}