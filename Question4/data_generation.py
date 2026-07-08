import numpy as np
import pandas as pd


def create_roll_numbers(count):
    depts = ["ch", "ce", "ma", "ai", "cs", "me", "bt", "ee"]
    batches = ["22", "23", "24", "25"]

    ids = []
    for d in depts:
        for b in batches:
            for r in range(1, 51):
                roll = "110" + str(r).zfill(2)
                ids.append(f"{d}{b}btech{roll}")

    np.random.shuffle(ids)
    return ids[:count]


def simulate_mess_data(size=500, seed=1):
    np.random.seed(seed)

    id_pool = create_roll_numbers(600)
    rows = []

    for _ in range(size):

        sid = np.random.choice(id_pool)

        weekday = np.random.randint(0, 7)
        meal_slot = np.random.randint(0, 3)
        veg_flag = np.random.randint(0, 2)

        weekend = int(weekday >= 5)
        no_class = int(weekday >= 5)

        deadline = np.random.choice([0, 1], p=[0.8, 0.2])

        temp = np.random.normal(30, 4)
        hum = np.random.uniform(40, 90)
        wind = np.random.uniform(0, 18)
        rain_val = np.random.choice([0, np.random.uniform(1, 20)], p=[0.7, 0.3])
        aqi = np.random.uniform(60, 180)

        wake_time = np.random.randint(300, 600)
        sleep_time = np.random.randint(1320, 1500)

        if meal_slot == 0:
            duration = np.random.randint(15, 28)
        elif meal_slot == 1:
            duration = np.random.randint(28, 45)
        else:
            duration = np.random.randint(35, 55)

        if weekend:
            duration += np.random.randint(5, 12)

        if deadline == 1:
            duration -= np.random.randint(4, 8)

        if rain_val > 8:
            duration += 4

        if aqi > 140:
            duration -= 2

        duration = max(10, duration)

        rows.append([
            sid, weekday, meal_slot, veg_flag,
            weekend, no_class, deadline,
            temp, hum, wind, rain_val, aqi,
            wake_time, sleep_time, duration
        ])

    cols = [
        "student_id", "day_of_week", "meal_time", "meal_type",
        "is_weekend", "class_day", "assignment_deadline",
        "temperature", "humidity", "wind_speed", "rain",
        "air_quality", "rising_time", "sleeping_time",
        "mess_duration"
    ]

    return pd.DataFrame(rows, columns=cols)


def expand_with_noise(df, target_size=5000):

    repeat_factor = target_size // len(df)
    expanded = df.loc[df.index.repeat(repeat_factor)].reset_index(drop=True)

    numeric_noise_cols = [
        "temperature",
        "humidity",
        "wind_speed",
        "air_quality",
        "rising_time",
        "sleeping_time"
    ]

    for col in numeric_noise_cols:
        noise = np.random.normal(0, 2, len(expanded))
        expanded[col] = expanded[col] + noise

    rain_noise = np.random.normal(0, 1, len(expanded))
    expanded["rain"] = np.clip(expanded["rain"] + rain_noise, 0, None)

    expanded["wind_speed"] = np.clip(expanded["wind_speed"], 0, None)

    return expanded


base_dataset = simulate_mess_data(500)

large_dataset = expand_with_noise(base_dataset, 5000)

large_dataset.to_csv("iit_h_mess_dataset_5000.csv", index=False)

print("Initial dataset:", base_dataset.shape)
print("Expanded dataset:", large_dataset.shape)