import numpy as np
import skfuzzy as fuzz
import random

temp_r = np.arange(0, 51, 1)

cold_mf = fuzz.trimf(temp_r, [0, 0, 20])
warm_mf = fuzz.trimf(temp_r, [15, 25, 35])
hot_mf = fuzz.trimf(temp_r, [30, 50, 50])


def predict(temp):
    return {
        label: round(float(fuzz.interp_membership(temp_r, mf, temp)), 2)
        for label, mf in zip(["cold", "warm", "hot"], [cold_mf, warm_mf, hot_mf])
    }


for _ in range(5):
    temp = round(random.uniform(0, 50), 2)
    result = predict(temp)
    print(f"Temp: {temp}°C -> {result}")
