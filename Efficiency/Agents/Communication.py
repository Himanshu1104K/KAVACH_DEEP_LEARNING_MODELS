import joblib
import numpy as np
from uagents import Agent, Context, Bureau
from Message import Message
import tensorflow as tf
import os
from TestingDataGeneratorAgent import agent

# Load the trained model
MODEL_FILE = "../TrainedModel/SHMS_Efficiency_Model.keras"
SCALER_FILE = "../TrainedModel/scaler.pkl"  # Path to the saved scaler

PredictionAgent = Agent(
    name="PredictionAgent",
    seed="seed",
    endpoint=["http://127.0.0.1:8001"],
)

# Initialize model and scaler
model = None
scaler = None


@PredictionAgent.on_event("startup")
async def model_load(ctx: Context):
    global model, scaler
    try:
        ctx.logger.info(f"Attempting to load model from {MODEL_FILE}")
        if not os.path.exists(MODEL_FILE):
            ctx.logger.error("Model file not found.")
            return
        model = tf.keras.models.load_model(MODEL_FILE)
        ctx.logger.info("Model loaded successfully.")

        # Load the saved scaler
        if not os.path.exists(SCALER_FILE):
            ctx.logger.error("Scaler file not found!")
            return
        scaler = joblib.load(SCALER_FILE)
        ctx.logger.info("Scaler loaded successfully!")

    except Exception as e:
        ctx.logger.error(f"Error during loading: {e}")


@PredictionAgent.on_message(model=Message)
async def perform_prediction(ctx: Context, sender: str, msg: Message):
    global model, scaler

    if model is None or scaler is None:
        ctx.logger.error("Model or Scaler not loaded! Cannot proceed with prediction.")
        return

    Results = []
    Temperature = msg.Temperature
    Moisture = msg.Moisture
    Water_Content = msg.Water_Content
    SpO2 = msg.SpO2
    Fatigue = msg.Fatigue
    Drowsiness = msg.Drowsiness
    Stress = msg.Stress
    Heart_Rate = msg.Heart_Rate
    Respiration_Rate = msg.Respiration_Rate
    Systolic_BP = msg.Systolic_BP
    Diastolic_BP = msg.Diastolic_BP
    if model:
        for x in range(0, len(Temperature)):
            input_data = np.array(
                [
                    Temperature[x],
                    Moisture[x],
                    Water_Content[x],
                    SpO2[x],
                    Fatigue[x],
                    Drowsiness[x],
                    Stress[x],
                    Heart_Rate[x],
                    Respiration_Rate[x],
                    Systolic_BP[x],
                    Diastolic_BP[x],
                ]
            ).reshape(1, -1)
            try:
                # Scale input data using pre-fitted scaler
                input_scaled = scaler.transform(input_data)

                # Perform prediction
                efficiency_score = model.predict(input_scaled).flatten()[0]
                Results.append(efficiency_score)

            except Exception as e:
                ctx.logger.error(f"Error during prediction: {e}")

    ctx.logger.info(f"Predictions is : {Results}")


# Bureau Setup
bureau = Bureau()
bureau.add(PredictionAgent)
bureau.add(agent)

if __name__ == "__main__":
    bureau.run()
