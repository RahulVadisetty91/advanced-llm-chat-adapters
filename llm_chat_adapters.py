from functools import cache, lru_cache
from typing import Dict, List, Tuple, Type
from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType
from dbgpt.model.llm.conversation import Conversation, get_conv_template
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure all chat adapters are defined correctly
class BaseChatAdapter:
    """Base class for chat with LLM models. It matches the model and fetches output from the model."""

    def match(self, model_path: str) -> bool:
        return False

    def get_generate_stream_func(self, model_path: str):
        """Return the generate stream handler function."""
        from dbgpt.model.llm.inference import generate_stream
        return generate_stream

    def get_conv_template(self, model_path: str) -> Conversation:
        return None

    @cache
    def _cached_get_conv_template(self, model_path: str) -> Conversation:
        """Cache the conversation template to improve performance."""
        return get_conv_template(model_path)

    def model_adaptation(
        self, params: Dict, model_path: str, prompt_template: str = None
    ) -> Tuple[Dict, Dict]:
        """Params adaptation with AI-driven prompt template selection."""
        conv = self._cached_get_conv_template(model_path)
        messages = params.get("messages")
        model_context = {"prompt_echo_len_char": -1}

        if messages:
            messages = [
                m if isinstance(m, ModelMessage) else ModelMessage(**m)
                for m in messages
            ]
            params["messages"] = messages

        if prompt_template:
            logger.info(f"Using prompt template {prompt_template} from config")
            conv = get_conv_template(prompt_template)

        if not conv or not messages:
            logger.warning(f"No conversation template for model_path {model_path} or no messages in params, {self}")
            return params, model_context
        
        conv = conv.copy()
        system_messages = []
        for message in messages:
            role, content = (message.role, message.content) if isinstance(message, ModelMessage) else (message["role"], message["content"])
            if role == ModelMessageRoleType.SYSTEM:
                system_messages.append(content)
            elif role == ModelMessageRoleType.HUMAN:
                conv.append_message(conv.roles[0], content)
            elif role == ModelMessageRoleType.AI:
                conv.append_message(conv.roles[1], content)
            else:
                raise ValueError(f"Unknown role: {role}")
        
        if system_messages:
            conv.update_system_message("".join(system_messages))
        conv.append_message(conv.roles[1], None)
        new_prompt = conv.get_prompt()
        prompt_echo_len_char = len(new_prompt.replace("</s>", "").replace("<s>", ""))
        model_context.update({
            "prompt_echo_len_char": prompt_echo_len_char,
            "echo": params.get("echo", True)
        })
        params.update({
            "prompt": new_prompt,
            "stop": conv.stop_str
        })

        return params, model_context

# Ensure all chat adapter classes are defined correctly
class VicunaChatAdapter(BaseChatAdapter):
    """Model chat Adapter for Vicuna"""
    def _is_llama2_based(self, model_path: str):
        return "v1.5" in model_path.lower()

    def match(self, model_path: str) -> bool:
        return "vicuna" in model_path.lower()

    def get_conv_template(self, model_path: str) -> Conversation:
        if self._is_llama2_based(model_path):
            return get_conv_template("vicuna_v1.1")
        return None

    def get_generate_stream_func(self, model_path: str):
        from dbgpt.model.llm_out.vicuna_base_llm import generate_stream
        if self._is_llama2_based(model_path):
            return super().get_generate_stream_func(model_path)
        return generate_stream

class ChatGLMChatAdapter(BaseChatAdapter):
    """Model chat Adapter for ChatGLM"""
    def match(self, model_path: str) -> bool:
        return "chatglm" in model_path

    def get_generate_stream_func(self, model_path: str):
        from dbgpt.model.llm_out.chatglm_llm import chatglm_generate_stream
        return chatglm_generate_stream

class FalconChatAdapter(BaseChatAdapter):
    """Model chat adapter for Falcon"""
    def match(self, model_path: str) -> bool:
        return "falcon" in model_path

    def get_generate_stream_func(self, model_path: str):
        from dbgpt.model.llm_out.falcon_llm import falcon_generate_output
        return falcon_generate_output

class GorillaChatAdapter(BaseChatAdapter):
    """Model chat adapter for Gorilla"""
    def match(self, model_path: str) -> bool:
        return "gorilla" in model_path

    def get_generate_stream_func(self, model_path: str):
        from dbgpt.model.llm_out.gorilla_llm import generate_stream
        return generate_stream

class GPT4AllChatAdapter(BaseChatAdapter):
    """Model chat adapter for GPT4All"""
    def match(self, model_path: str) -> bool:
        return "gptj-6b" in model_path

    def get_generate_stream_func(self, model_path: str):
        from dbgpt.model.llm_out.gpt4all_llm import gpt4all_generate_stream
        return gpt4all_generate_stream

class Llama2ChatAdapter(BaseChatAdapter):
    """Model chat adapter for Llama2"""
    def match(self, model_path: str) -> bool:
        return "llama-2" in model_path.lower()

    def get_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-2")

class CodeLlamaChatAdapter(BaseChatAdapter):
    """Model chat adapter for CodeLlama"""
    def match(self, model_path: str) -> bool:
        return "codellama" in model_path.lower()

    def get_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("codellama")

class BaichuanChatAdapter(BaseChatAdapter):
    """Model chat adapter for Baichuan"""
    def match(self, model_path: str) -> bool:
        return "baichuan" in model_path.lower()

    def get_conv_template(self, model_path: str) -> Conversation:
        if "chat" in model_path.lower():
            return get_conv_template("baichuan-chat")
        return get_conv_template("zero_shot")

class WizardLMChatAdapter(BaseChatAdapter):
    """Model chat adapter for WizardLM"""
    def match(self, model_path: str) -> bool:
        return "wizardlm" in model_path.lower()

    def get_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("vicuna_v1.1")

class LlamaCppChatAdapter(BaseChatAdapter):
    """Model chat adapter for LlamaCpp"""
    def match(self, model_path: str) -> bool:
        from dbgpt.model.adapter.old_adapter import LlamaCppAdapter
        if "llama-cpp" == model_path:
            return True
        is_match, _ = LlamaCppAdapter._parse_model_path(model_path)
        return is_match

    def get_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-2")

    def get_generate_stream_func(self, model_path: str):
        from dbgpt.model.llm_out.llama_cpp_llm import generate_stream
        return generate_stream

class InternLMChatAdapter(BaseChatAdapter):
    """Model chat adapter for InternLM"""
    def match(self, model_path: str) -> bool:
        return "internlm" in model_path.lower()

    def get_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("internlm-chat")

def register_llm_model_chat_adapter(cls: Type[BaseChatAdapter]):
    """Register a chat adapter."""
    llm_model_chat_adapters.append(cls())

# Register all adapters
register_llm_model_chat_adapter(VicunaChatAdapter)
register_llm_model_chat_adapter(ChatGLMChatAdapter)
register_llm_model_chat_adapter(FalconChatAdapter)
register_llm_model_chat_adapter(GorillaChatAdapter)
register_llm_model_chat_adapter(GPT4AllChatAdapter)
register_llm_model_chat_adapter(Llama2ChatAdapter)
register_llm_model_chat_adapter(CodeLlamaChatAdapter)
register_llm_model_chat_adapter(BaichuanChatAdapter)
register_llm_model_chat_adapter(WizardLMChatAdapter)
register_llm_model_chat_adapter(LlamaCppChatAdapter)
register_llm_model_chat_adapter(InternLMChatAdapter)

# Uncomment if ProxyllmChatAdapter is needed
# register_llm_model_chat_adapter(ProxyllmChatAdapter)
