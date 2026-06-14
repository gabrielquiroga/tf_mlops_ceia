from airflow.decorators import dag, task
from datetime import datetime, timedelta

default_args = {
    "owner": "gabi",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "dagrun_timeout": timedelta(minutes=25)
}

@dag(
    dag_id="train_stellar_classifiers",
    description="Orquestador de entrenamiento de modelos y promoción.",
    default_args=default_args,
    catchup=False,
    tags=["train", "random_forest", "xgboost", "promotion"],
    start_date=datetime(2025, 1, 1),
    schedule="@weekly", # "0 0 * * 1", # Todos los Lunes a las 00:00 hs
)
def train_stellar_classifiers():
    
    @task(
        task_id="train_random_forest",
    )
    def train_random_forest() -> dict:
        """
        Entrena un modelo Random Forest, lo registra y devuelve un diccionario con:
        - run_id (mlflow)
        - model_type
        - model_version
        - metrics
        """
        from ml.train import train
        from airflow.sdk import Variable

        dataset_source = Variable.get("dataset_source")

        print("Entrenando modelo Random Forest con datos de ", dataset_source)
        return train(
            model_type="random_forest",
            source=dataset_source
        )

    @task(
        task_id="train_xgboost",
    )
    def train_xgboost() -> dict:
        """
        Entrena un modelo XGBoost, lo registra y devuelve un diccionario con:
        - run_id (mlflow)
        - model_type
        - model_version
        - metrics
        """
        from ml.train import train
        from airflow.sdk import Variable

        dataset_source = Variable.get("dataset_source")

        print("Entrenando modelo Random Forest con datos de ", dataset_source)
        return train(
            model_type="xgboost",
            source=dataset_source
        )

    @task(
        task_id="select_best_challenger"
    )
    def select_best_challenger(contestants: list[dict]) -> dict:
        """
        Recibe una lista de diccionarios con los resultados de los modelos entrenados.
        Retorna el diccionario del modelo con mejor f1 macro.
        """
        best_model = max(contestants, key=lambda x: x["metrics"]["f1_macro"])
        print("El mejor modelo para challenger es: ", best_model["model_type"])
        return best_model

    @task(
        task_id="retrieve_champion_info"
    )
    def retrieve_champion_info():
        from ml.predict import get_champion_metric
        """Recupera la información disponible del champion de MLflow."""
        return get_champion_metric("f1_macro")

    @task(
        task_id="compare_and_promote"
    )
    def compare_and_promote(
        champion: dict,
        challenger: dict,
        metric: str
    ) -> None:
        """
        Compara la métrica elegida entre el champion y el challenger.
        Si el challenger es mejor, es promovido a champion.
        Devuelve el dict del champion.
        """
        from ml.predict import promote_to_champion
        print("Comparando champion contra el challenger")
        print("Champion: ", champion)
        print("Challenger: ", challenger)
        # El champion inicial puede ser un dict vacío
        if not champion:
            print("No hay champion actual, se promueve al challenger")
            promote_to_champion(challenger["model_version"])
        else:
            if champion[metric] < challenger.get("metrics")[metric]:
                promote_to_champion(challenger["model_version"])
                print("Se ha promovido al challenger a champion")
            else:
                print("El champion sigue invicto")
        return None

    # Workflow
    model1 = train_random_forest()
    model2 = train_xgboost()
    best_challenger = select_best_challenger([model1, model2])
    champion_info = retrieve_champion_info()

    compare_and_promote(
        champion=champion_info,
        challenger=best_challenger,
        metric="f1_macro"
    )

dag = train_stellar_classifiers()
