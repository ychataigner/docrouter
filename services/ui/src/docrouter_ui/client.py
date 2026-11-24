"""Client HTTP de l'API DocRouter."""

import httpx
from docrouter_core import ClassifyResponse, HealthResponse


class ApiError(Exception):
    """Erreur lisible par l'utilisateur (API injoignable, fichier refusé…)."""


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 60.0, transport=None) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def health(self) -> HealthResponse:
        response = self._request("GET", "/health", timeout=5.0)
        return HealthResponse.model_validate(response.json())

    def classify(self, filename: str, data: bytes, content_type: str) -> ClassifyResponse:
        response = self._request(
            "POST", "/v1/classify", files={"file": (filename, data, content_type)}
        )
        return ClassifyResponse.model_validate(response.json())

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiError("L'API n'a pas répondu à temps.") from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"API injoignable ({self._client.base_url}).") from exc

        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"Erreur {response.status_code} : {detail}")
        return response
