import os
import sys
import pandas as pd
from src.utils import load_object
from dataclasses import dataclass
from src.exception import custom_exception


class prediction_pipeline:
    def __init__(self):
        pass

    def prediction(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessing_path = os.path.join('artifacts', 'data_preprocessing.pkl')

            model = load_object(file_path=model_path)
            preprocessing = load_object(file_path=preprocessing_path)

            scaled_data = preprocessing.transform(features)
            preds = model.predict(scaled_data)

            return preds

        except Exception as e:
            raise custom_exception(e, sys)


@dataclass
class CustomData:
    train_id: int
    region: str
    season: str
    train_type: str
    train_age_years: float
    average_speed_kmph: float
    distance_travelled_km: float
    ambient_temperature_c: float
    humidity_percent: float
    rainfall_mm: float
    wheel_wear_percent: float
    track_vibration_level: float
    rail_wear_mm: float
    bearing_temperature_c: float
    axle_temperature_c: float
    brake_pad_wear_percent: float
    brake_pressure_psi: float
    battery_voltage: float
    last_maintenance_days: int
    sensor_health_index: float
    inspection_score: float
    delay_minutes: float
    failure_type: str
    maintenance_required: int
    failure_severity: str
    risk_score: float

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "train_id": [self.train_id],
                "region": [self.region],
                "season": [self.season],
                "train_type": [self.train_type],
                "train_age_years": [self.train_age_years],
                "average_speed_kmph": [self.average_speed_kmph],
                "distance_travelled_km": [self.distance_travelled_km],
                "ambient_temperature_c": [self.ambient_temperature_c],
                "humidity_percent": [self.humidity_percent],
                "rainfall_mm": [self.rainfall_mm],
                "wheel_wear_percent": [self.wheel_wear_percent],
                "track_vibration_level": [self.track_vibration_level],
                "rail_wear_mm": [self.rail_wear_mm],
                "bearing_temperature_c": [self.bearing_temperature_c],
                "axle_temperature_c": [self.axle_temperature_c],
                "brake_pad_wear_percent": [self.brake_pad_wear_percent],
                "brake_pressure_psi": [self.brake_pressure_psi],
                "battery_voltage": [self.battery_voltage],
                "last_maintenance_days": [self.last_maintenance_days],
                "sensor_health_index": [self.sensor_health_index],
                "inspection_score": [self.inspection_score],
                "delay_minutes": [self.delay_minutes],
                "failure_type": [self.failure_type],
                "maintenance_required": [self.maintenance_required],
                "failure_severity": [self.failure_severity],
                "risk_score": [self.risk_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise custom_exception(e, sys)