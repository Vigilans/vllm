# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.speech_to_text.transcription.api_router import (
    router as transcription_router,
)
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionResponse,
)
from vllm.entrypoints.speech_to_text.translation.api_router import (
    router as translation_router,
)
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationResponse


class TestSpeechToTextResponses(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(transcription_router)
        app.include_router(translation_router)
        self.transcription = SimpleNamespace(create_transcription=AsyncMock())
        self.translation = SimpleNamespace(create_translation=AsyncMock())
        app.state.openai_serving_transcription = self.transcription
        app.state.openai_serving_translation = self.translation
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.endpoints = (
            (
                "/v1/audio/transcriptions",
                self.transcription.create_transcription,
                TranscriptionResponse,
            ),
            (
                "/v1/audio/translations",
                self.translation.create_translation,
                TranslationResponse,
            ),
        )

    def post(self, path, response_format, **fields):
        return self.client.post(
            path,
            data={"model": "asr", "response_format": response_format, **fields},
            files={"file": ("sample.wav", b"audio", "audio/wav")},
        )

    def test_text_returns_utf8_plain_text(self):
        for path, create, response_cls in self.endpoints:
            for text in ("Hello world.", '你好，"世界"。\n第二行。', ""):
                with self.subTest(path=path, text=text):
                    create.return_value = response_cls(
                        text=text, usage={"type": "duration", "seconds": 1}
                    )
                    response = self.post(path, "text")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(
                        response.headers["content-type"], "text/plain; charset=utf-8"
                    )
                    self.assertEqual(response.content, text.encode("utf-8"))

    def test_json_includes_text_and_usage(self):
        for path, create, response_cls in self.endpoints:
            with self.subTest(path=path):
                result = response_cls(
                    text="Hello world.", usage={"type": "duration", "seconds": 1}
                )
                create.return_value = result
                response = self.post(path, "json")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "application/json")
                self.assertEqual(response.json(), result.model_dump())

    def test_text_request_errors_return_json(self):
        error = ErrorResponse(
            error=ErrorInfo(message="Invalid audio", type="BadRequestError", code=400)
        )
        for path, create, _ in self.endpoints:
            with self.subTest(path=path):
                create.return_value = error
                response = self.post(path, "text")
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.headers["content-type"], "application/json")
                self.assertEqual(response.json(), error.model_dump())

    def test_streaming_returns_sse(self):
        async def chunks():
            yield 'data: {"text":"hello"}\n\n'
            yield "data: [DONE]\n\n"

        for path, create, _ in self.endpoints:
            for response_format in ("json", "text"):
                with self.subTest(path=path, response_format=response_format):
                    create.return_value = chunks()
                    response = self.post(path, response_format, stream="true")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(
                        response.headers["content-type"],
                        "text/event-stream; charset=utf-8",
                    )
                    self.assertEqual(
                        response.text, 'data: {"text":"hello"}\n\ndata: [DONE]\n\n'
                    )
