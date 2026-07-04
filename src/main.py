from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

ingestion = DataIngestion()

train_path, test_path = ingestion.DataIngestion_pipeline()

transformation = DataTransformation()

X_train, X_test, y_train, y_test, preprocessor = (
    transformation.DataTransformation_pipeline(
        train_path,
        test_path
    )
)