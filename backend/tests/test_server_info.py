"""Backend capability handshake tests."""


def test_managed_local_server_info(client) -> None:
    response = client.get("/api/server-info")
    assert response.status_code == 200
    assert response.json() == {
        "api_version": 2,
        "deployment_mode": "managed_local",
        "capabilities": {
            "media_ids": True,
            "artifact_downloads": True,
            "legacy_path_inputs": True,
            "models_dir_editable": True,
        },
    }


def test_standalone_server_info(client, test_state) -> None:
    test_state.config.deployment_mode = "standalone"
    test_state.config.allow_legacy_path_inputs = False
    test_state.config.models_dir_editable = False
    response = client.get("/api/server-info")
    assert response.status_code == 200
    assert response.json()["deployment_mode"] == "standalone"
    assert response.json()["capabilities"]["legacy_path_inputs"] is False
