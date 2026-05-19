from sqlalchemy import Column, Integer, Float, String
from .database import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    bmi = Column(Float)
    smoker = Column(String)
    prediction = Column(Float)