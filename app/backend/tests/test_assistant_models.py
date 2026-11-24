import pytest


def catalog(client):
    response = client.get("/api/assistant/models")
    assert response.status_code == 200, response.text
    return response.json()


def test_download_completion_enables_selection_without_switching_automatically(client, fake_services):
    fake_services.assistant_models.models.clear()
    response = client.post("/api/assistant/models/download", json={"model": "qwen3.5:9b"})
    assert response.status_code == 200, response.text
    result = catalog(client)
    assert result["download"]["status"] == "complete"
    assert result["models"][0]["installed"]
    assert client.get("/api/settings").json()["assistantProvider"] == "gemini"


def test_running_download_can_be_reattached_and_cannot_be_started_twice(client, fake_services):
    models = fake_services.assistant_models
    models.models.clear()

    def during_pull():
        result = catalog(client)
        assert result["download"]["model"] == "qwen3.5:9b"
        assert result["download"]["status"] == "downloading"
        assert result["download"]["progress"] == 50
        assert not any(model["installed"] for model in result["models"])
        response = client.post("/api/assistant/models/download", json={"model": "qwen3.5:9b"})
        assert response.status_code == 409
        assert response.json()["code"] == "DOWNLOAD_ALREADY_RUNNING"

    models.during_pull = during_pull
    assert client.post("/api/assistant/models/download", json={"model": "qwen3.5:9b"}).status_code == 200
    result = catalog(client)
    assert result["download"]["status"] == "complete"
    assert result["models"][0]["installed"]


def test_failed_download_stays_uninstalled_and_can_retry(client, fake_services):
    models = fake_services.assistant_models
    models.models.clear()
    models.pull_error = "Download interrupted"
    assert client.post("/api/assistant/models/download", json={"model": "qwen3.5:9b"}).status_code == 200
    result = catalog(client)
    assert result["download"]["status"] == "error"
    assert result["download"]["error"] == "Download interrupted"
    assert not any(model["installed"] for model in result["models"])
    models.pull_error = None
    assert client.post("/api/assistant/models/download", json={"model": "qwen3.5:9b"}).status_code == 200
    assert catalog(client)["download"]["status"] == "complete"


def test_unsupported_platform_disables_local_models(client, fake_services):
    fake_services.assistant_models.available = False
    result = catalog(client)
    assert not result["available"]
    assert "not supported" in result["error"]
    assert client.post("/api/assistant/models/download", json={"model": "qwen3.5:9b"}).status_code == 503


@pytest.mark.parametrize("model", ["qwen3.5:cloud", "other-model", "../../model", "gemma-4-e2b-it"])
def test_only_catalog_models_can_be_downloaded(client, model):
    assert client.post("/api/assistant/models/download", json={"model": model}).status_code == 422


def test_removed_selected_model_is_rejected_without_saving_a_run(client, test_state, fake_services):
    from tests.test_assistant import request_body, saved_run

    test_state.state.app_settings.assistant_provider = "local"
    fake_services.assistant_models.models.clear()
    body = request_body("Hello")
    response = client.post("/api/assistant/turn", json=body)
    assert response.status_code == 409
    assert response.json()["code"] == "ASSISTANT_LOCAL_MODEL_MISSING"
    assert saved_run(test_state, body["request_id"]) is None
