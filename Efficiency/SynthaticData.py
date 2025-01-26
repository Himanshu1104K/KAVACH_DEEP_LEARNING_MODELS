import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples in each efficiency category
num_samples_per_group = 1700  # Total ~5100 records

# **1. Generate Low Efficiency Data**
low_temperature = np.random.uniform(36, 38, num_samples_per_group)
low_moisture = np.random.uniform(30, 50, num_samples_per_group)
low_water_content = np.random.uniform(40, 60, num_samples_per_group)
low_spo2 = np.random.uniform(85, 90, num_samples_per_group)
low_fatigue = np.random.uniform(0.7, 1, num_samples_per_group)
low_drowsiness = np.random.uniform(0.6, 1, num_samples_per_group)
low_stress = np.random.uniform(0.7, 1, num_samples_per_group)
low_heart_rate = np.random.uniform(110, 140, num_samples_per_group)
low_respiration_rate = np.random.uniform(20, 30, num_samples_per_group)
low_systolic = np.random.randint(130, 140, num_samples_per_group)
low_diastolic = np.random.randint(85, 90, num_samples_per_group)

low_efficiency = (
    (0.3 * low_spo2 / 100 + 0.2 * low_water_content / 100) -
    (0.3 * low_fatigue + 0.2 * low_drowsiness + 0.25 * low_stress) -
    (0.15 * (low_heart_rate - 80) / 60) - 
    (0.1 * (low_systolic - 120) / 30)
)

# **2. Generate Medium Efficiency Data**
med_temperature = np.random.uniform(36, 39, num_samples_per_group)
med_moisture = np.random.uniform(40, 60, num_samples_per_group)
med_water_content = np.random.uniform(50, 80, num_samples_per_group)
med_spo2 = np.random.uniform(90, 96, num_samples_per_group)
med_fatigue = np.random.uniform(0.3, 0.7, num_samples_per_group)
med_drowsiness = np.random.uniform(0.3, 0.7, num_samples_per_group)
med_stress = np.random.uniform(0.3, 0.7, num_samples_per_group)
med_heart_rate = np.random.uniform(80, 110, num_samples_per_group)
med_respiration_rate = np.random.uniform(16, 25, num_samples_per_group)
med_systolic = np.random.randint(115, 130, num_samples_per_group)
med_diastolic = np.random.randint(75, 85, num_samples_per_group)

med_efficiency = (
    (0.3 * med_spo2 / 100 + 0.2 * med_water_content / 100) -
    (0.3 * med_fatigue + 0.2 * med_drowsiness + 0.25 * med_stress) -
    (0.15 * (med_heart_rate - 80) / 60) -
    (0.1 * (med_systolic - 120) / 30)
)

# **3. Generate High Efficiency Data**
high_temperature = np.random.uniform(35, 37, num_samples_per_group)
high_moisture = np.random.uniform(50, 70, num_samples_per_group)
high_water_content = np.random.uniform(70, 90, num_samples_per_group)
high_spo2 = np.random.uniform(96, 100, num_samples_per_group)
high_fatigue = np.random.uniform(0, 0.3, num_samples_per_group)
high_drowsiness = np.random.uniform(0, 0.3, num_samples_per_group)
high_stress = np.random.uniform(0, 0.3, num_samples_per_group)
high_heart_rate = np.random.uniform(60, 90, num_samples_per_group)
high_respiration_rate = np.random.uniform(12, 20, num_samples_per_group)
high_systolic = np.random.randint(110, 120, num_samples_per_group)
high_diastolic = np.random.randint(70, 80, num_samples_per_group)

high_efficiency = (
    (0.3 * high_spo2 / 100 + 0.2 * high_water_content / 100) -
    (0.3 * high_fatigue + 0.2 * high_drowsiness + 0.25 * high_stress) -
    (0.15 * (high_heart_rate - 80) / 60) -
    (0.1 * (high_systolic - 120) / 30)
)

# Normalize to range [0.1, 0.9]
low_efficiency = np.interp(low_efficiency, (low_efficiency.min(), low_efficiency.max()), (0.1, 0.3))
med_efficiency = np.interp(med_efficiency, (med_efficiency.min(), med_efficiency.max()), (0.3, 0.6))
high_efficiency = np.interp(high_efficiency, (high_efficiency.min(), high_efficiency.max()), (0.6, 0.9))

# **Combine Data**
df_low = pd.DataFrame({
    "Temperature": low_temperature, "Moisture": low_moisture, "Water_Content": low_water_content,
    "SpO2": low_spo2, "Fatigue": low_fatigue, "Drowsiness": low_drowsiness, "Stress": low_stress,
    "Heart_Rate": low_heart_rate, "Respiration_Rate": low_respiration_rate, 
    "Systolic_BP": low_systolic, "Diastolic_BP": low_diastolic, "Efficiency": low_efficiency
})

df_med = pd.DataFrame({
    "Temperature": med_temperature, "Moisture": med_moisture, "Water_Content": med_water_content,
    "SpO2": med_spo2, "Fatigue": med_fatigue, "Drowsiness": med_drowsiness, "Stress": med_stress,
    "Heart_Rate": med_heart_rate, "Respiration_Rate": med_respiration_rate, 
    "Systolic_BP": med_systolic, "Diastolic_BP": med_diastolic, "Efficiency": med_efficiency
})

df_high = pd.DataFrame({
    "Temperature": high_temperature, "Moisture": high_moisture, "Water_Content": high_water_content,
    "SpO2": high_spo2, "Fatigue": high_fatigue, "Drowsiness": high_drowsiness, "Stress": high_stress,
    "Heart_Rate": high_heart_rate, "Respiration_Rate": high_respiration_rate, 
    "Systolic_BP": high_systolic, "Diastolic_BP": high_diastolic, "Efficiency": high_efficiency
})

# Merge All Data
df_final = pd.concat([df_low, df_med, df_high], ignore_index=True)

# Shuffle the data
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Save as CSV
df_final.to_csv("balanced_soldier_data.csv", index=False)

# Display Summary
print(df_final["Efficiency"].describe())

# Display first few rows
df_final.head()
