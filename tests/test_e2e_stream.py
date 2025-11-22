from fastapi.testclient import TestClient
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_service import app
import json
import time

from raglite.api.endpoints import rag_service as api_rag_service


class FakeLLMClient:
    def generate(self, model, prompt, options=None, stream=False):
        # If not streaming, return final dict
        if not stream:
            return {'response': 'This is a final response.'}

        # Yield some chunks that include 'thinking' and 'response'
        yield {'thinking': 'thinking chunk 1'}
        time.sleep(0.01)
        yield {'thinking': 'thinking chunk 2'}
        time.sleep(0.01)
        yield {'response': 'response part 1'}
        time.sleep(0.01)
        yield {'response': 'response part 2.'}
        time.sleep(0.01)
        yield {'done': True, 'done_reason': 'complete'}


# Patch dataset and search behaviors for E2E testing
def setup_service_for_test(monkeypatch):
    # No-op connect_* methods
    monkeypatch.setattr(api_rag_service, 'connect_embedding_server', lambda *args, **kwargs: True)
    monkeypatch.setattr(api_rag_service, 'connect_llm_server', lambda *args, **kwargs: True)
    monkeypatch.setattr(api_rag_service, 'connect_dataset_server', lambda *args, **kwargs: True)

    # Replace list_available_datasets with a dataset including 'test_index'
    monkeypatch.setattr(api_rag_service, 'list_available_datasets', lambda: [{'index_name': 'test_index'}])

    # Replace semantic_search to return a single hit
    monkeypatch.setattr(api_rag_service, 'semantic_search', lambda *args, **kwargs: {'hits': {'hits': [{'_source': {'content': 'doc 1 content'}}]}})
    monkeypatch.setattr(api_rag_service, 'rerank_search_results', lambda *args, **kwargs: args[1])

    # Inject the fake LLM client
    api_rag_service.llm_client = FakeLLMClient()


def parse_sse_lines(lines):
    # Return event list with (event_type, data)
    events = []
    event = None
    data_buf = []

    for line in lines:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        line = line.strip('\r')
        if not line:
            if event:
                events.append((event, '\n'.join(data_buf)))
            event = None
            data_buf = []
            continue
        if line.startswith('event:'):
            event = line.split(':', 1)[1].strip()
        elif line.startswith('data:'):
            data_buf.append(line.split(':', 1)[1].strip())

    # Append if final
    if event:
        events.append((event, '\n'.join(data_buf)))
    return events


def test_e2e_sse_excludes_thinking(monkeypatch):
    setup_service_for_test(monkeypatch)
    client = TestClient(app)

    payload = {
        'query': 'test',
        'index_name': 'test_index',
        'include_thinking': False,
        'stream': True
    }

    response = client.post('/rag/stream', json=payload)
    assert response.status_code == 200
    lines = list(response.iter_lines())

    events = parse_sse_lines(lines)
    # Make a lookup of event types
    event_types = {e: d for (e, d) in events}

    # There should be token events (response) but not 'thinking' text inside (we check text)
    all_text = '\n'.join([d for _, d in events])
    assert 'thinking chunk 1' not in all_text
    assert 'response part 1' in all_text
    assert 'event: end' not in all_text  # We assert that 'end' payload is not present verbatim in data because SSE uses an 'end' event; parse above returns 'end' event separately


def test_e2e_sse_includes_thinking(monkeypatch):
    setup_service_for_test(monkeypatch)
    client = TestClient(app)

    payload = {
        'query': 'test',
        'index_name': 'test_index',
        'include_thinking': True,
        'stream': True
    }

    response = client.post('/rag/stream', json=payload)
    assert response.status_code == 200
    lines = list(response.iter_lines())

    events = parse_sse_lines(lines)
    all_text = '\n'.join([d for _, d in events])
    assert 'thinking chunk 1' in all_text
    assert 'response part 1' in all_text
    # end event should be present (verify last event type 'end' exists)
    assert any(e == 'end' for e, _ in events)
