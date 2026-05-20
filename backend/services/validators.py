from fastapi import HTTPException


def validate_binary_file(content_type: str) -> None:
    """
    Проверяет, что загруженный файл является бинарным.

    Args:
        content_type (str):
            MIME-тип загружаемого файла.

    Raises:
        HTTPException:
            Возникает, если файл имеет неверный формат.
    """

    if content_type != "application/octet-stream":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only binary files are allowed.",
        )


def validate_csv_file(content_type: str) -> None:
    """
    Проверяет, что загруженный файл является CSV.

    Args:
        content_type (str):
            MIME-тип загружаемого файла.

    Raises:
        HTTPException:
            Возникает, если файл имеет неверный формат.
    """

    if content_type != "text/csv":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only CSV files are allowed.",
        )