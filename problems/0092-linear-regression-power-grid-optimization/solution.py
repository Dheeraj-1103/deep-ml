import math
import numpy as np

def power_grid_forecast(consumption_data):
    # 1. Generate day indices (days 1, 2, 3, ...)
    days = np.arange(1, len(consumption_data) + 1)

    # 2. Remove daily fluctuation
    detrended = []

    for i, consumption in zip(days, consumption_data):
        fluctuation = 10 * math.sin(2 * math.pi * i / 10)
        detrended.append(consumption - fluctuation)

    # 3. Perform linear regression
    slope, intercept = np.polyfit(days, detrended, 1)

    # 4. Predict base consumption for day 15
    base_consumption = slope * 15 + intercept

    # 5. Add day 15 fluctuation
    fluctuation_15 = 10 * math.sin(2 * math.pi * 15 / 10)

    prediction = base_consumption + fluctuation_15

    # 6. Round prediction
    prediction = round(prediction)

    # 7. Add 5% safety margin and round up
    final_prediction = math.ceil(prediction * 1.05)

    return final_prediction