from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Класс глобальных настроек приложения.

    Attributes:
        model (object | None):
            Загруженная ML-модель для предсказания одобрения ипотеки.
    """

    model: object = None


settings = Settings()
