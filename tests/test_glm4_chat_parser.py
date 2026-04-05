import pytest
from transformers import AutoTokenizer

from rllm.parser import ChatTemplateParser, GLM4ChatTemplateParser
from rllm.parser.utils import PARSER_TEST_MESSAGES


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("THUDM/GLM-4-9B-0414", trust_remote_code=True)


@pytest.fixture(scope="module")
def parser(tokenizer):
    return GLM4ChatTemplateParser(tokenizer)


class TestGLM4ChatTemplateParser:
    def test_equivalence(self, parser):
        """Batch parsing must equal concatenated individual parsing."""
        assert parser.verify_equivalence(PARSER_TEST_MESSAGES)

    def test_factory_returns_glm4_parser(self, tokenizer):
        """get_parser() should return GLM4ChatTemplateParser for GLM models."""
        p = ChatTemplateParser.get_parser(tokenizer)
        assert isinstance(p, GLM4ChatTemplateParser)

    def test_parse_basic(self, parser):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = parser.parse(messages, is_first_msg=True, add_generation_prompt=True)
        assert result == "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\nHello<|assistant|>"

    def test_parse_no_system(self, parser):
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = parser.parse(messages, is_first_msg=True, add_generation_prompt=True)
        assert result == "[gMASK]<sop><|user|>\nHello<|assistant|>"

    def test_parse_multi_turn(self, parser):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        result = parser.parse(messages, is_first_msg=True, add_generation_prompt=True)
        assert result == "[gMASK]<sop><|user|>\nHello<|assistant|>\nHi<|user|>\nHow are you?<|assistant|>"

    def test_parse_matches_tokenizer_template(self, tokenizer, parser):
        """Parser output should match the tokenizer's built-in apply_chat_template."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        expected = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        result = parser.parse(messages, is_first_msg=True, add_generation_prompt=False)
        assert result == expected

    def test_parse_matches_tokenizer_with_gen_prompt(self, tokenizer, parser):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        expected = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = parser.parse(messages, is_first_msg=True, add_generation_prompt=True)
        assert result == expected

    def test_parse_tool_observation(self, parser):
        messages = [
            {"role": "user", "content": "Search for Python."},
            {"role": "assistant", "content": "Searching..."},
            {"role": "tool", "content": "Python is a programming language."},
        ]
        result = parser.parse(messages, is_first_msg=True)
        assert "<|observation|>\nPython is a programming language." in result

    def test_bos_only_on_first_msg(self, parser):
        messages = [{"role": "user", "content": "Hello"}]
        with_bos = parser.parse(messages, is_first_msg=True)
        without_bos = parser.parse(messages, is_first_msg=False)
        assert with_bos.startswith("[gMASK]<sop>")
        assert not without_bos.startswith("[gMASK]<sop>")

    def test_parse_completion_plain(self, parser):
        text = "Hello, how can I help you?"
        ids = parser.tokenizer.encode(text, add_special_tokens=False)
        result = parser.parse_completion(ids)
        assert result["content"] == text
        assert result["reasoning"] == ""
        assert result["tool_calls"] == []

    def test_parse_completion_strips_user_token(self, parser):
        text = "Hello, how can I help you?<|user|>"
        ids = parser.tokenizer.encode(text, add_special_tokens=False)
        result = parser.parse_completion(ids)
        assert result["content"] == "Hello, how can I help you?"

    def test_parse_completion_strips_endoftext(self, parser):
        text = "Hello<|endoftext|>"
        ids = parser.tokenizer.encode(text, add_special_tokens=False)
        result = parser.parse_completion(ids)
        assert result["content"] == "Hello"

    def test_parse_completion_strips_observation(self, parser):
        text = "Let me search for that.<|observation|>"
        ids = parser.tokenizer.encode(text, add_special_tokens=False)
        result = parser.parse_completion(ids)
        assert result["content"] == "Let me search for that."

    def test_generation_prompt(self, parser):
        assert parser.generation_prompt == "<|assistant|>"
