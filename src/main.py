from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

ingestion = DataIngestion()

train_path, test_path = ingestion.DataIngestion_pipeline()

transformation = DataTransformation()

X_train, X_test, y_train, y_test, preprocessor = (
    transformation.DataTransformation_pipeline(
        train_path,
        test_path
    )
)

trainer = ModelTrainer()

trainer.ModelTrainer_Pipeline(
    X_test,
    X_train,
    y_test,
    y_train
)