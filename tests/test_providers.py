"""Unit tests for ProviderResponse token usage tracking."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from agentapi.providers.base import ProviderResponse
from agentapi.providers.openai_compatible import OpenAICompatibleProvider
from agentapi.providers.gemini import GeminiProvider
from agentapi.providers.anthropic import AnthropicProvider


def test_openai_compatible_usage_extraction():
    async def _test():
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
        )

        mock_json = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello world",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            },
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_json)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await provider.chat([{"role": "user", "content": "Hi"}])

            assert isinstance(res, ProviderResponse)
            assert res.content == "Hello world"
            assert res.usage == {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            }

    asyncio.run(_test())


def test_openai_compatible_usage_missing():
    async def _test():
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
        )

        mock_json = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                    }
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_json)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await provider.chat([{"role": "user", "content": "Hi"}])

            assert isinstance(res, ProviderResponse)
            assert res.usage is None

    asyncio.run(_test())


def test_gemini_usage_extraction():
    async def _test():
        provider = GeminiProvider(
            api_key="test-key",
            model="gemini-2.5-flash",
        )

        mock_json = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini answer"}]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150,
            },
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_json)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await provider.chat([{"role": "user", "content": "Hi"}])

            assert isinstance(res, ProviderResponse)
            assert res.content == "Gemini answer"
            assert res.usage == {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }

    asyncio.run(_test())


def test_gemini_usage_missing():
    async def _test():
        provider = GeminiProvider(
            api_key="test-key",
            model="gemini-2.5-flash",
        )

        mock_json = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini answer"}]
                    }
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_json)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await provider.chat([{"role": "user", "content": "Hi"}])

            assert isinstance(res, ProviderResponse)
            assert res.usage is None

    asyncio.run(_test())


def test_anthropic_usage_extraction():
    async def _test():
        with patch("agentapi.providers.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="test-key", model="claude-3-5-sonnet-20241022")

            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "Claude response"

            mock_usage = MagicMock()
            mock_usage.input_tokens = 42
            mock_usage.output_tokens = 18

            mock_response = MagicMock()
            mock_response.content = [mock_block]
            mock_response.usage = mock_usage
            mock_response.model_dump = MagicMock(return_value={
                "content": [{"type": "text", "text": "Claude response"}],
                "usage": {"input_tokens": 42, "output_tokens": 18}
            })

            provider.client.messages.create = AsyncMock(return_value=mock_response)

            res = await provider.chat([{"role": "user", "content": "Hi"}])

            assert isinstance(res, ProviderResponse)
            assert res.content == "Claude response"
            assert res.usage == {
                "prompt_tokens": 42,
                "completion_tokens": 18,
                "total_tokens": 60,
            }

    asyncio.run(_test())
