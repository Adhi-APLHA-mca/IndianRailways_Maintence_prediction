from fastapi import FastAPI,Request,Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.pipeline.prediction import prediction_pipeline,CustomData

application = FastAPI()
templates = Jinja2Templates(directory="src/template")
application.mount("/static", StaticFiles(directory="src"), name="static")

@application.get("/")
def base_prediction(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )

@application.post('/predict')
def predict(
    request: Request,
    train_id: int = Form(...),
    region: str = Form(...),
    season: str = Form(...),
    train_type: str = Form(...),
    train_age_years: float = Form(...),
    average_speed_kmph: float = Form(...),
    distance_travelled_km: float = Form(...),
    ambient_temperature_c: float = Form(...),
    humidity_percent: float = Form(...),
    rainfall_mm: float = Form(...),
    wheel_wear_percent: float = Form(...),
    track_vibration_level: float = Form(...),
    rail_wear_mm: float = Form(...),
    bearing_temperature_c: float = Form(...),
    axle_temperature_c: float = Form(...),
    brake_pad_wear_percent: float = Form(...),
    brake_pressure_psi: float = Form(...),
    battery_voltage: float = Form(...),
    last_maintenance_days: int = Form(...),
    sensor_health_index: float = Form(...),
    inspection_score: float = Form(...),
    delay_minutes: float = Form(...)
):
    data = CustomData(
        train_id=train_id,
        region=region,
        season=season,
        train_type=train_type,
        train_age_years=train_age_years,
        average_speed_kmph=average_speed_kmph,
        distance_travelled_km=distance_travelled_km,
        ambient_temperature_c=ambient_temperature_c,
        humidity_percent=humidity_percent,
        rainfall_mm=rainfall_mm,
        wheel_wear_percent=wheel_wear_percent,
        track_vibration_level=track_vibration_level,
        rail_wear_mm=rail_wear_mm,
        bearing_temperature_c=bearing_temperature_c,
        axle_temperature_c=axle_temperature_c,
        brake_pad_wear_percent=brake_pad_wear_percent,
        brake_pressure_psi=brake_pressure_psi,
        battery_voltage=battery_voltage,
        last_maintenance_days=last_maintenance_days,
        sensor_health_index=sensor_health_index,
        inspection_score=inspection_score,
        delay_minutes=delay_minutes 
    )

    pipeline = prediction_pipeline()
    result = pipeline.prediction(data.get_data_as_dataframe())
    prediction_value = int(result[0])
    prediction_label = "Yes" if prediction_value == 1 else "No"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": f"Maintenance Required: {prediction_label}"
        }
    )