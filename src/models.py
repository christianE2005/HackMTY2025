# src/models.py
from sqlalchemy import Column, Integer, String, Date, Numeric
from .database import Base


class ConsumptionPrediction(Base):
    __tablename__ = 'consumption_prediction'

    id = Column(Integer, primary_key=True)
    flight_id = Column(String(20))
    origin = Column(String(10))
    date = Column(Date)
    flight_type = Column(String(30))
    service_type = Column(String(30))
    passenger_count = Column(Integer)
    product_id = Column(String(20))
    product_name = Column(String(100))
    standard_specification_qty = Column(Integer)
    quantity_returned = Column(Integer)
    quantity_consumed = Column(Integer)
    unit_cost = Column(Numeric(10, 2))
    crew_feedback = Column(Numeric(3, 2))
