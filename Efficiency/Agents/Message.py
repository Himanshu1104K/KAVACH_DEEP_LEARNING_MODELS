from typing import List
from uagents import Model
class Message(Model):
    Timestamp: List[str]  # Use string for timestamps
    Temperature: List[float]
    Moisture: List[float]
    Water_Content: List[float]
    SpO2: List[float]
    Fatigue: List[float]
    Drowsiness: List[float]
    Stress: List[float]
    Heart_Rate: List[float]
    Respiration_Rate: List[float]
    Systolic_BP: List[int]
    Diastolic_BP: List[int]