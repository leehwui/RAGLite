import json
import asyncio

from raglite.core.rag_service import RAGService

class FakeLLMClient:
    def __init__(self, chunks):
        self._chunks = chunks

    def generate(self, model, prompt, options=None, stream=False):
        if stream:
            # Return an iterator/generator
            for c in self._chunks:
                yield c
        else:
            # Return a dict-like response
            return self._chunks[-1] if isinstance(self._chunks[-1], dict) else {'response': str(self._chunks[-1])}


def test_generate_sse_stream_skips_thinking():
    rs = RAGService()
    # Create fake search_results
    search_results = {'hits': {'hits': [{'_source': {'content': 'doc1 content'}}]}}

    # chunks where first yields thinking tokens, then response tokens, and done
    chunks = [
        {'thinking': 'thinking1'},
        {'thinking': 'thinking2'},
        {'response': 'Hello world.'},
        {'done': True, 'done_reason': 'length'}
    ]

    rs.llm_client = FakeLLMClient(chunks)

    async def run_test():
        collected = []
        async for s in rs.generate_sse_stream('query', search_results, 'llm', 128, include_thinking=False):
            collected.append(s)
        # Ensure no thinking tokens were included; tokens should only be the response and end payload
        data = ''.join(collected)
        assert 'thinking1' not in data
        assert 'thinking2' not in data
        assert 'Hello world.' in data

    asyncio.run(run_test())


def test_generate_sse_stream_includes_thinking():
    rs = RAGService()
    search_results = {'hits': {'hits': [{'_source': {'content': 'doc1 content'}}]}}
    chunks = [
        {'thinking': 'thinking1'},
        {'response': 'Hello world.'},
        {'done': True, 'done_reason': 'length'}
    ]
    rs.llm_client = FakeLLMClient(chunks)

    async def run_test():
        collected = []
        async for s in rs.generate_sse_stream('query', search_results, 'llm', 128, include_thinking=True):
            collected.append(s)
        data = ''.join(collected)
        assert 'thinking1' in data
        assert 'Hello world.' in data

    asyncio.run(run_test())
