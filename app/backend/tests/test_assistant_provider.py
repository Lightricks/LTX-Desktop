import asyncio
import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from pydantic_ai import Agent, BinaryContent, ToolReturn

from services.assistant_provider import DesktopAssistantProvider
from tests.fakes.assistant import FakeLlamaRuntime


@contextmanager
def local_server(status=200):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers["Authorization"], payload))
            message = {"role": "assistant", "content": "Finished"}
            if len(requests) == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "read_value", "arguments": '{}'}},
                ]}
            data = json.dumps({"id": "local-1", "object": "chat.completion", "created": 1, "model": "qwen3.5:9b",
                               "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if len(requests) == 1 else "stop"}]}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_local_model_stays_resident_until_the_tool_loop_finishes(tmp_path):
    with local_server() as (url, requests):
        runtime = FakeLlamaRuntime(tmp_path, url)
        provider = DesktopAssistantProvider(runtime, lambda _: tmp_path / "model.gguf", projector_path=lambda _: tmp_path / "mmproj.gguf")
        calls = []

        async def check():
            async with provider.create_model(provider="local", model="qwen3.5:9b", api_key="never-forward-this-key") as model:
                agent = Agent(model)

                @agent.tool_plain
                def read_value() -> ToolReturn:
                    assert runtime.active
                    calls.append(True)
                    return ToolReturn(return_value=42, content=[BinaryContent(data=b"preview", media_type="image/jpeg")])

                result = await agent.run("Read the value.")
                assert result.output == "Finished"

        asyncio.run(check())
        assert calls == [True]
        assert not runtime.active
        assert all(path == "/v1/chat/completions" and auth == "Bearer test-local-key" for path, auth, _ in requests)
        assert "never-forward-this-key" not in json.dumps(requests)
        assert requests[0][2]["tools"][0]["function"]["name"] == "read_value"
        assert any(message["role"] == "tool" for message in requests[1][2]["messages"])
        images = [part for message in requests[1][2]["messages"] if isinstance(message.get("content"), list)
                  for part in message["content"] if part["type"] == "image_url"]
        assert images == [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,cHJldmlldw=="}}]



def test_local_runtime_is_released_on_model_error(tmp_path):
    with local_server(status=400) as (url, _):
        runtime = FakeLlamaRuntime(tmp_path, url)

        async def check():
            async with DesktopAssistantProvider(runtime, lambda _: tmp_path / "model.gguf", projector_path=lambda _: tmp_path / "mmproj.gguf").create_model(
                provider="local", model="qwen3.5:9b", api_key="",
            ) as model:
                await Agent(model).run("Hello")

        with pytest.raises(Exception):
            asyncio.run(check())
        assert not runtime.active


def test_assistant_settings_survive_reload_and_unrelated_updates(client, test_state, default_app_settings):
    response = client.post("/api/settings", json={
        "assistantProvider": "local", "assistantLocalModel": "qwen3.5:9b",
    })
    assert response.status_code == 200, response.text
    test_state.settings.load_settings(default_app_settings)
    response = client.get("/api/settings")
    assert response.json()["assistantProvider"] == "local"
    assert response.json()["assistantLocalModel"] == "qwen3.5:9b"
    assert client.post("/api/settings", json={"promptCacheSize": 5}).status_code == 200
    assert client.get("/api/settings").json()["assistantProvider"] == "local"


def test_cloud_model_tags_are_not_a_local_option(client):
    response = client.post("/api/settings", json={"assistantLocalModel": "qwen3.5:cloud"})
    assert response.status_code == 422
