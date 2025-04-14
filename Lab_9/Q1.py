import numpy as np
import skfuzzy as fuzz
import random
import time

temperature_range = np.arange(0, 51, 1)

cold_mf = fuzz.trimf(temperature_range, [0, 0, 20])
warm_mf = fuzz.trimf(temperature_range, [15, 25, 35])
hot_mf = fuzz.trimf(temperature_range, [30, 50, 50])

def classify_temperature(temp):
    cold = fuzz.interp_membership(temperature_range, cold_mf, temp)
    warm = fuzz.interp_membership(temperature_range, warm_mf, temp)
    hot = fuzz.interp_membership(temperature_range, hot_mf, temp)
    return {
        'cold': round(float(cold), 2),
        'warm': round(float(warm), 2),
        'hot': round(float(hot), 2)
    }

for _ in range(5):
    temp = round(random.uniform(0, 50), 2)
    result = classify_temperature(temp)
    print(f"Temp: {temp}°C | Classification: {result}")
    time.sleep(1)