from uagents import Agent, Context
import numpy as np
import pandas as pd
from Message import Message

agent = Agent(
    name="Data_Generator",
    seed="seed",
    endpoint=["http://127.0.0.1:8002"],
)
num_samples = 10


@agent.on_interval(period=2.0)
async def generate_data(ctx: Context):
    Timestamp = []
    Temperature = []
    Moisture = []
    Water_Content = []
    SpO2 = []
    Fatigue = []
    Drowsiness = []
    Stress = []
    Heart_Rate = []
    Respiration_Rate = []
    Systolic_BP = []
    Diastolic_BP = []
    for x in range(num_samples):
        timestamp = str(pd.Timestamp.now())
        category = np.random.choice(
            ["low", "medium", "high"], p=[0.3, 0.4, 0.3]
        )  # Define probabilities

        if category == "low":
            temp = np.random.uniform(38, 40)  # High fever
            moisture = np.random.uniform(10, 30)  # Very low hydration
            water_content = np.random.uniform(20, 40)  # Below normal range
            spO2 = np.random.uniform(80, 90)  # Critical oxygen levels
            fatigue = np.random.uniform(80, 100)  # High fatigue
            drowsiness = np.random.uniform(70, 100)  # High drowsiness
            stress = np.random.uniform(70, 100)  # High stress
            heart_rate = np.random.uniform(100, 130)  # Elevated heart rate
            respiration_rate = np.random.uniform(25, 35)  # High respiration rate
            systolic = np.random.randint(130, 140)
            diastolic = np.random.randint(85, 90)

        elif category == "medium":
            temp = np.random.uniform(36, 38)  # Slightly elevated temp
            moisture = np.random.uniform(30, 50)  # Moderate hydration
            water_content = np.random.uniform(40, 60)  # Acceptable levels
            spO2 = np.random.uniform(90, 95)  # Acceptable oxygen
            fatigue = np.random.uniform(40, 70)  # Moderate fatigue
            drowsiness = np.random.uniform(30, 60)  # Medium drowsiness
            stress = np.random.uniform(30, 60)  # Medium stress
            heart_rate = np.random.uniform(80, 100)  # Slightly high heart rate
            respiration_rate = np.random.uniform(18, 25)  # Normal respiration rate
            systolic = np.random.randint(115, 130)
            diastolic = np.random.randint(75, 85)

        else:  # High efficiency
            temp = np.random.uniform(35, 36.5)  # Normal body temperature
            moisture = np.random.uniform(50, 70)  # Good hydration
            water_content = np.random.uniform(60, 80)  # Excellent water retention
            spO2 = np.random.uniform(95, 100)  # Optimal oxygen levels
            fatigue = np.random.uniform(10, 40)  # Low fatigue
            drowsiness = np.random.uniform(10, 30)  # Low drowsiness
            stress = np.random.uniform(10, 30)  # Low stress
            heart_rate = np.random.uniform(60, 80)  # Normal heart rate
            respiration_rate = np.random.uniform(12, 18)  # Normal respiration rate
            systolic = np.random.randint(110, 120)
            diastolic = np.random.randint(70, 80)

        # Appending Data to the Lists
        Timestamp.append(timestamp)
        Temperature.append(temp)
        Moisture.append(moisture)
        Water_Content.append(water_content)
        SpO2.append(spO2)
        Fatigue.append(fatigue)
        Drowsiness.append(drowsiness)
        Stress.append(stress)
        Heart_Rate.append(heart_rate)
        Respiration_Rate.append(respiration_rate)
        Systolic_BP.append(systolic)
        Diastolic_BP.append(diastolic)

    from Communication import agent

    Address = agent.address
    message = Message(
        Timestamp=Timestamp,
        Temperature=Temperature,
        Moisture=Moisture,
        Water_Content=Water_Content,
        SpO2=SpO2,
        Fatigue=Fatigue,
        Drowsiness=Drowsiness,
        Stress=Stress,
        Heart_Rate=Heart_Rate,
        Respiration_Rate=Respiration_Rate,
        Systolic_BP=Systolic_BP,
        Diastolic_BP=Diastolic_BP,
    )
    try:
        ctx.logger.info(message)
        await ctx.send(
            Address,
            message,
        )
        ctx.logger.info(f"Data Send to the Prediction Agent at Address : {Address}")
    except Exception as E:
        ctx.logger.error(f"Failed to send data to PredictionAgent: {E}")
