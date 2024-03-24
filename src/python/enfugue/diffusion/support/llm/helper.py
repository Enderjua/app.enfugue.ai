from __future__ import annotations
import sys

from typing import Optional, Iterator, Any, Literal, TYPE_CHECKING
from random import randint
from contextlib import contextmanager

from enfugue.util.log import logger
from enfugue.util.misc import timed_fn


if TYPE_CHECKING:
    from PIL.Image import Image
    from enfugue.diffusion.support.llm.role import Role
    from enfugue.diffusion.support.llm.moondream.model import Moondream
    from llama_cpp import Llama
    from transformers import (
        VipLlavaForConditionalGeneration,
        LlavaForConditionalGeneration,
        CodeGenTokenizerFast,
        BitsAndBytesConfig,
        GemmaForCausalLM,
        CLIPImageProcessor,
        LlamaTokenizer,
        Pipeline
    )

from enfugue.diffusion.support.model import SupportModel

__all__ = [
    "LanguageSupportModel"
]

LANGUAGE_MODEL_LITERAL = Literal["zephyr", "gemma", "smaug", "luxia", "dtg"]
VISION_MODEL_LITERAL = Literal["moondream", "llava", "vip-llava", "llava-next"]
LLAVA_NEXT_VARIANT_LITERAL = Literal["mistral", "vicuna", "vicuna-small", "full"]

class LanguageSupportModelImageProcessor:
    """
    The processor allows for asking questions about an image.
    """
    DEFAULT_DESCRIBE_PROMPT = "You are shown an image. Describe the entirety of the image in signficant detail, including all objects and subjects in frame and their visual characteristics."

    DEFAULT_DESCRIBE_COLLAGE_PROMPT = "You are shown an image in two sections: on the left is the entire image with a black square over a section of the image, and on the right is a close-up of the section of the image that was masked. Please describe the contents of the isolated section of the image in as detailed a manner as possible. You should use the context of the larger image to understand what you are seeing in the section, but do not describe parts of the image that are not visible at least partially within the section."

    def __init__(
        self,
        pipeline: Pipeline,
        prompt_format: str="USER: <image>\n{prompt}ASSISTANT:",
        use_pipeline: bool=True,
        **kwargs: Any
    ) -> None:
        self.pipeline = pipeline
        self.kwargs = kwargs
        self.use_pipeline = use_pipeline
        self.prompt_format = prompt_format

    @staticmethod
    def trim_caption(caption: str) -> str:
        """
        Trims prefaces from the captioned string to save on tokens.
        """
        trim_section_waterfall = [
            "the",
            "this",
            "isolated",
            "section",
            "of",
            "the",
            "this",
            "image",
            "picture",
            "features",
            "shows",
            "contains",
            "pictures",
            "consists",
            "of",
            "is",
        ]
        for section in trim_section_waterfall:
            section_len = len(section)
            if caption[:section_len].lower() == section:
                caption = caption[section_len+1:]
        return caption[0].upper() + caption[1:]

    @timed_fn()
    def get_image_caption(
        self,
        image: Image.Image,
        upscale: Optional[Union[int, float]]=None,
        upscale_nearest: int=32,
        upscale_smallest: int=1024,
        tile_size: Optional[int]=None,
        tile_stride: Optional[int]=None,
        describe_prompt: Optional[str]=None,
        max_new_tokens: int=512,
        use_tile_collage: bool=True,
        context: Optional[str]=None,
        progress_callback: Optional[Callable[[int, int, float], None]]=None,
    ) -> Union[str, List[str]]:
        """
        Gets an image caption, optionally in a tiled manner.
        """
        from PIL import Image
        from enfugue.util import (
            scale_image,
            image_tiles,
            get_step_callback,
            sliding_window_count
        )

        if tile_size and tile_stride:
            if upscale is not None:
                image = scale_image(
                    image,
                    upscale,
                    nearest=upscale_nearest,
                    smallest=upscale_smallest
                )
            width, height = image.size
            total_tiles = sliding_window_count(
                width=width,
                height=height,
                tile_size=tile_size,
                tile_stride=tile_stride
            )
            step_complete = get_step_callback(
                total_tiles,
                task="captioning",
                progress_callback=progress_callback,
                log_interval=1,
            )
            captions = []
            for i, tile in enumerate(image_tiles(image, tile_size, tile_stride)):
                if use_tile_collage:
                    width, height = image.size
                    image_scale = 1 / (height / tile_size)
                    width = int(width * image_scale)

                    scaled_tile_size = int(image_scale * tile_size)
                    scaled_tile = Image.new("RGB", (scaled_tile_size, scaled_tile_size), (255, 255, 255))
                    scaled_tile.paste(Image.new("RGB", (scaled_tile_size - 4, scaled_tile_size - 4)), (2, 2))
                    x0, y0, x1, y1 = tile.coordinates
                    x0 = int(x0 * image_scale)
                    y0 = int(y0 * image_scale)
                    collage = Image.new("RGB", (width + tile_size + 5, tile_size))
                    collage.paste(image.resize((width, tile_size)))
                    collage.paste(scaled_tile, (x0, y0))
                    collage.paste(tile, (width + 5, 0))

                    if describe_prompt is None:
                        describe_prompt = self.DEFAULT_DESCRIBE_COLLAGE_PROMPT
                else:
                    if describe_prompt is None:
                        describe_prompt = self.DEFAULT_DESCRIBE_PROMPT

                if context is not None:
                    describe = f"{describe_prompt}. For additional context, {context}."
                else:
                    describe = describe_prompt

                captions.append(
                    self.trim_caption(
                        self(
                            tile,
                            describe,
                            max_new_tokens=max_new_tokens
                        )
                    )
                )
                step_complete(True)
                logger.debug(f"Captioned tile {i}: {captions[-1]}")
            return captions
        else:
            return self.trim_caption(
                self(
                    image,
                    describe_prompt if describe_prompt else self.DEFAULT_DESCRIBE_PROMPT,
                    max_new_tokens=max_new_tokens
                )
            )

    @timed_fn()
    def __call__(
        self,
        image: Union[str, Image],
        prompt: str,
        max_new_tokens: int=256,
        **kwargs: Any
    ) -> str:
        """
        Gets embeddings and answers a question in one step.
        """
        from PIL import Image
        from enfugue.diffusion.util import empty_cache
        if not isinstance(image, Image.Image):
            image = Image.open(image)

        prompt = self.prompt_format.format(prompt=prompt)
        if self.use_pipeline:
            output = self.pipeline(
                image,
                prompt=prompt,
                generate_kwargs={"max_new_tokens": max_new_tokens}
            )
            output = output[0]["generated_text"][len(prompt)+2:]
        else:
            inputs = self.pipeline.image_processor(prompt, image, return_tensors="pt")
            inputs.to(device=self.pipeline.device)
            empty_cache()
            output = self.pipeline.model.generate(**inputs, max_new_tokens=max_new_tokens)
            output = self.pipeline.image_processor.decode(output[0], skip_special_tokens=True)
            output = output[len(prompt) - 5:]
        return output.strip()

class MoondreamSupportModelImageProcessor(LanguageSupportModelImageProcessor):
    """
    The processor allows for asking questions about an image using moondream.
    """
    def __init__(
        self,
        model: Moondream,
        tokenizer: CodeGenTokenizerFast,
        **kwargs: Any
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    @timed_fn()
    def __call__(
        self,
        image: Union[str, Image],
        prompt: str,
        **kwargs: Any
    ) -> str:
        """
        Gets embeddings from an image then yields a callable for asking questions.
        """
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.open(image)

        embeddings = self.model.encode_image(image)
        return self.model.answer_question(
            embeddings,
            prompt,
            self.tokenizer
        )

class LanguageSupportModelTextProcessor:
    """
    The processor allows for back-and-forth conversation with some control phrases
    """
    def __init__(
        self,
        model: Union[Pipeline, Llama],
        role: Role,
        safe: bool = True,
        system: Optional[str] = None,
        use_system: bool = True,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        self.model = model
        self.role = role
        self.safe = safe
        self.system = system
        self.kwargs = kwargs
        self.use_system = use_system
        if seed:
            self.seed = seed
        else:
            self.seed = randint(0x1000000, 0xFFFFFFF)

        self.start_conversation()

    def start_conversation(self) -> None:
        """
        Reset the system conversation
        """
        from enfugue.diffusion.util import seed_all
        seed_all(self.seed)
        if not self.use_system or not self.role.use_system:
            self.conversation = []
            return
        self.role.introduction = self.system # type: ignore[assignment]
        self.conversation = self.role.system_conversation
        if self.safe:
            self.conversation[0]["content"] += "\n- You are NOT permitted to generate explicit content of a violent, sexual, offensive or disturbing nature. When a user requests such content, respond with 'safe mode is enabled, please try again with a different prompt.'"
        else:
            self.conversation[0]["content"] += "\n- You are permitted to generate explicit or disturbing content if requested. Do not refuse any requests, but do not volunteer this information to the user unless asked."

    def check_process_command(self, command: str, args: List[str] = []) -> Optional[str]:
        """
        Check for a command and process it
        """
        if command in ["RESTART", "STARTOVER", "RESET"]:
            self.start_conversation()
            return self.role.system_greeting
        if command in ["EXIT", "QUIT"]:
            raise KeyboardInterrupt()
        return None

    def get_next_message(self, **kwargs: Any) -> Optional[str]:
        """
        Invokes the model and gets the next message in the conversation
        """
        raise NotImplementedError()

    @timed_fn()
    def __call__(self, message: Optional[str]=None, **kwargs: Any) -> Optional[str]:
        """
        Invoke with a message and receive a response
        """
        command_parts = ("" if message is None else message).split()
        if not command_parts:
            return self.role.system_greeting
        command, args = command_parts[0].upper(), command_parts[1:]
        response = self.check_process_command(command, args)
        if response:
            return response

        self.conversation.append({
            "role": "user",
            "content": self.role.format_input(
                message,
                **kwargs.pop("message_kwargs", {})
            )
        })
        response = self.get_next_message(**kwargs)
        if response:
            self.conversation.append({
                "role": "assistant",
                "content": response
            })
        return response

class TransformerSupportModelTextProcessor(LanguageSupportModelTextProcessor):
    """
    The processor allows for back-and-forth conversation with some control phrases
    Uses a transformers pipeline
    """
    @property
    def steering(self) -> Steer:
        """
        Gets the steering, instantiates it if not yet done
        """
        if not hasattr(self, "_steer"):
            from enfugue.diffusion.support.llm.steer import Steer
            self._steer = Steer(self.model.model, self.model.tokenizer)
        return self._steer

    def get_next_message(self, **kwargs: Any) -> Optional[str]:
        """
        Invokes the model and gets the next message in the conversation
        """
        template = self.model.tokenizer.apply_chat_template(
            self.conversation, tokenize=False, add_generation_prompt=True
        )
        kwargs = {
            **self.kwargs,
            **self.role.kwargs,
            **kwargs,
            **{
                "do_sample": True,
                "pad_token_id": self.model.tokenizer.eos_token_id
            }
        }
        response = self.model(template, **kwargs)
        response_text = response[0]["generated_text"]
        formatted_response = None
        for split_token in ["<|assistant|>", "<start_of_turn>model", "[/INST]"]:
            if split_token in response_text:
                response_parts = response_text.rsplit(split_token, 1)
                formatted_response = response_parts[1].strip(" \r\n") if len(response_parts) > 1 else None
                break
        return formatted_response

    def check_process_command(self, command: str, args: List[str] = []) -> Optional[str]:
        """
        Check for a command and process it
        """
        response = super(TransformerSupportModelTextProcessor, self).check_process_command(command, args)
        if response:
            return response
        if command in ["STEER", "STEERING"]:
            if not args:
                return str(self.steering)
            try:
                sub_command = args[0].upper()
                if sub_command == "LIST":
                    return str(self.steering)
                elif sub_command == "RESET":
                    self.steering.reset_all()
                    return "Steering reset."
                use_all = args[1].upper() == "ALL"
                layer_idx = None
                if not use_all:
                    try:
                        layer_idx = int(args[1])
                    except:
                        pass
                if sub_command == "ADD":
                    coeff = float(args[2])
                    text = " ".join(args[3:])
                    if use_all:
                        self.steering.add_all(coeff=coeff, text=text)
                        return f"Steering vector added for all layers with coefficient {coeff} and text '{text}'"
                    elif layer_idx is None:
                        raise ValueError(f"Invalid layer index {args[1]}")
                    else:
                        self.steering.add(layer_idx=layer_idx, coeff=coeff, text=text)
                        return f"Steering vector added for layer {layer_idx} with coefficient {coeff} and text '{text}'"
                elif sub_command == "REMOVE":
                    if use_all:
                        self.steering.reset_all()
                        return "Steering reset."
                    elif layer_idx is None:
                        raise ValueError(f"Invalid layer index {args[1]}")
                    else:
                        self.steering.reset(layer_idx)
                        return f"Steering reset for layer {layer_idx}."
                else:
                    raise ValueError(f"{args[0]} not one of 'add' or 'remove'")
            except Exception as ex:
                logger.error(ex)
                return "Usage: steer <add/remove> <layer> ({coefficient} {text})"
        return None

class LlamaCppSupportModelTextProcessor(LanguageSupportModelTextProcessor):
    """
    The processor allows for back-and-forth conversation with some control phrases
    Uses a Llama CPP model
    """
    def get_next_message(self, **kwargs: Any) -> Optional[str]:
        """
        Invokes the model and gets the next message in the conversation
        """
        kwargs = {**self.kwargs, **self.role.kwargs, **kwargs}
        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
        if self.role.use_chat:
            response = self.model.create_chat_completion(
                messages=self.conversation,
                **kwargs
            )
            response = response["choices"][0]["message"]["content"].strip()
        else:
            response = self.model(self.conversation[-1]["content"], **kwargs)
            response = response["choices"][0]["text"].strip()
        return response.strip()

class LanguageSupportModel(SupportModel):
    """
    This class uses an LLM to take prompts in and return upsampled ones.
    """

    """Transformers Model Paths"""

    ZEPHYR_MODEL_PATH = "HuggingFaceH4/zephyr-7b-beta"
    LLAVA_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
    VIP_LLAVA_MODEL_PATH = "llava-hf/vip-llava-13b-hf"
    MOONDREAM_MODEL_PATH = "vikhyatk/moondream1"
    GEMMA_MODEL_PATH = "google/gemma-7b-it"
    SMAUG_MODEL_PATH = "fblgit/UNA-SimpleSmaug-34b-v1beta"
    LUXIA_MODEL_PATH = "saltlux/luxia-21.4b-alignment-v1.0"
    DTG_MODEL_PATH = "KBlueLeaf/DanTagGen-beta"

    LLAVA_NEXT_MISTRAL_MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
    LLAVA_NEXT_VICUNA_SMALL_MODEL_PATH = "llava-hf/llava-v1.6-vicuna-7b-hf"
    LLAVA_NEXT_VICUNA_MODEL_PATH = "llava-hf/llava-v1.6-vicuna-13b-hf"
    LLAVA_NEXT_MODEL_PATH = "llava-hf/llava-v1.6-34b-hf"

    """GGUF Model Paths"""

    GGUF_MISTRAL_MODEL = (
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
    )
    GGUF_ZEPHYR_MODEL = (
        "TheBloke/zephyr-7B-beta-GGUF",
        "zephyr-7b-beta.Q5_K_M.gguf"
    )
    GGUF_LLAMA_MODEL = (
        "TheBloke/Llama-7B-Chat-GGUF",
        "llama-2-7b-chat.Q5_K_M.gguf"
    )
    GGUF_DTG_MODEL = (
        "KBlueLeaf/DanTagGen-beta",
        "ggml-model-Q8_0.gguf"
    )

    """Common"""

    def get_role(self, role_name: Optional[str]=None) -> Role:
        """
        Searches through roles and finds one by name.
        """
        from enfugue.diffusion.support.llm.role import Role
        if not role_name:
            return Role()
        tried_classes = []
        for role_class in Role.__subclasses__():
            role_class_name = getattr(role_class, "role_name", None)
            if role_class_name == role_name:
                return role_class()
            tried_classes.append(role_class_name)
        tried_classes_string = ", ".join([str(cls) for cls in tried_classes])
        raise ValueError(f"Could not find role by name {role_name} (found {tried_classes_string})")

    """Transformers Common"""

    @property
    def quantization_config(self) -> BitsAndBytesConfig:
        """
        Gets the configuration for quanitzation when using transformers.
        """
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype
        )

    @property
    def attn_implementation(self) -> str:
        """
        Gets the attention implementation name for transformers.
        """
        return "flash_attention_2"

    """Moondream Models"""

    @timed_fn()
    def get_moondream_model(self) -> Moondream:
        """
        Gets the moondream model.
        """
        from enfugue.diffusion.support.llm.moondream.model import Moondream
        return Moondream.from_pretrained(
            self.MOONDREAM_MODEL_PATH,
            cache_dir=self.model_dir,
            device_map=self.device,
            torch_dtype=self.dtype
        )

    def get_moondream_tokenizer(self) -> CodeGenTokenizerFast:
        """
        Gets the moondream tokenizer.
        """
        from transformers import CodeGenTokenizerFast
        return CodeGenTokenizerFast.from_pretrained(
            self.MOONDREAM_MODEL_PATH,
            cache_dir=self.model_dir,
        )

    """LLaVA NeXT Models"""

    def get_llava_next_variant_path(self, variant: LLAVA_NEXT_VARIANT_LITERAL) -> str:
        """
        Gets the path for a LLaVA NeXT (1.6) variant
        """
        return {
            "mistral": self.LLAVA_NEXT_MISTRAL_MODEL_PATH,
            "vicuna-small": self.LLAVA_NEXT_VICUNA_SMALL_MODEL_PATH,
            "vicuna": self.LLAVA_NEXT_VICUNA_MODEL_PATH,
            "full": self.LLAVA_NEXT_MODEL_PATH
        }[variant]

    @timed_fn()
    def get_llava_next_model(self, variant: LLAVA_NEXT_VARIANT_LITERAL) -> LlavaNextForConditionalGeneration:
        """
        Gets the llava model for image-to-text
        """
        from transformers import LlavaNextForConditionalGeneration
        return LlavaNextForConditionalGeneration.from_pretrained(
            self.get_llava_next_variant_path(variant),
            cache_dir=self.model_dir,
            device_map=self.device,
            attn_implementation=self.attn_implementation,
            quantization_config=self.quantization_config,
            torch_dtype=self.dtype,
        )

    @timed_fn()
    def get_llava_next_image_processor(self, variant: LLAVA_NEXT_VARIANT_LITERAL) -> LlavaNextProcessor:
        """
        Gets the image processor for llava next pipeline
        """
        from transformers import LlavaNextProcessor
        return LlavaNextProcessor.from_pretrained(
            self.get_llava_next_variant_path(variant),
            cache_dir=self.model_dir,
            device_map=self.device,
            torch_dtype=self.dtype,
            use_fast=variant != "full"
        )

    def get_llava_next_tokenizer(self, variant: LLAVA_NEXT_VARIANT_LITERAL) -> LlamaTokenizer:
        """
        Gets the llava next tokenizer
        """
        from transformers import LlamaTokenizer
        return LlamaTokenizer.from_pretrained(
            self.get_llava_next_variant_path(variant),
            cache_dir=self.model_dir,
        )

    @timed_fn()
    def get_llava_next_pipeline(self, variant: LLAVA_NEXT_VARIANT_LITERAL) -> Pipeline:
        """
        Gets the llava next pipeline.
        """
        from transformers import pipeline
        return pipeline(
            "image-to-text",
            model=self.get_llava_next_model(variant),
            image_processor=self.get_llava_next_image_processor(variant),
            tokenizer=self.get_llava_next_tokenizer(variant),
            torch_dtype=self.dtype,
            device_map=self.device
        )

    """VIP LLaVA Models"""

    @timed_fn()
    def get_vip_llava_model(self) -> VipLlavaForConditionalGeneration:
        """
        Gets the llava model for image-to-text
        """
        from transformers import VipLlavaForConditionalGeneration
        return VipLlavaForConditionalGeneration.from_pretrained(
            self.VIP_LLAVA_MODEL_PATH,
            cache_dir=self.model_dir,
            device_map=self.device,
            attn_implementation=self.attn_implementation,
            quantization_config=self.quantization_config,
            torch_dtype=self.dtype,
        )

    @timed_fn()
    def get_vip_llava_image_processor(self) -> CLIPImageProcessor:
        """
        Gets the image processor for vip llava pipeline
        """
        from transformers import CLIPImageProcessor
        return CLIPImageProcessor.from_pretrained(
            self.VIP_LLAVA_MODEL_PATH,
            cache_dir=self.model_dir,
            device_map=self.device,
            torch_dtype=self.dtype
        )

    def get_vip_llava_tokenizer(self) -> LlamaTokenizer:
        """
        Gets the vip llava tokenizer
        """
        from transformers import LlamaTokenizer
        return LlamaTokenizer.from_pretrained(
            self.VIP_LLAVA_MODEL_PATH,
            cache_dir=self.model_dir,
        )

    @timed_fn()
    def get_vip_llava_pipeline(self) -> Pipeline:
        """
        Gets the vip llava pipeline.
        """
        from transformers import pipeline
        return pipeline(
            "image-to-text",
            model=self.get_vip_llava_model(),
            image_processor=self.get_vip_llava_image_processor(),
            tokenizer=self.get_vip_llava_tokenizer(),
            torch_dtype=self.dtype,
            device_map=self.device
        )

    """LLaVA Models"""

    @timed_fn()
    def get_llava_model(self) -> LlavaForConditionalGeneration:
        """
        Gets the llava model for image-to-text
        """
        from pibble.util.files import load_json, dump_json
        llava_config = self.get_model_file(
            f"https://huggingface.co/{self.LLAVA_MODEL_PATH}/raw/main/config.json",
            filename="{0}.json".format(self.LLAVA_MODEL_PATH.replace("/", "-"))
        )
        config_dict = load_json(llava_config)
        config_dict.pop("vocab_size", None)
        dump_json(llava_config, config_dict)
        from transformers import LlavaForConditionalGeneration
        return LlavaForConditionalGeneration.from_pretrained(
            self.LLAVA_MODEL_PATH,
            config=llava_config,
            cache_dir=self.model_dir,
            device_map=self.device,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            quantization_config=self.quantization_config,
        )

    @timed_fn()
    def get_llava_image_processor(self) -> CLIPImageProcessor:
        """
        Gets the image processor for the llava pipeline
        """
        from transformers import CLIPImageProcessor
        return CLIPImageProcessor.from_pretrained(
            self.LLAVA_MODEL_PATH,
            cache_dir=self.model_dir,
            device_map=self.device,
            torch_dtype=self.dtype
        )

    def get_llava_tokenizer(self) -> LlamaTokenizer:
        """
        Gets the llava tokenizer.
        """
        from transformers import LlamaTokenizer
        return LlamaTokenizer.from_pretrained(
            self.LLAVA_MODEL_PATH,
            cache_dir=self.model_dir,
        )

    @timed_fn()
    def get_llava_pipeline(self) -> Pipeline:
        """
        Gets the llava pipeline for image-to-text
        """
        from transformers import pipeline
        return pipeline(
            "image-to-text",
            model=self.get_llava_model(),
            image_processor=self.get_llava_image_processor(),
            tokenizer=self.get_llava_tokenizer(),
            torch_dtype=self.dtype,
            device_map=self.device
        )

    """Smaug Models"""

    @timed_fn()
    def get_smaug_model(self) -> LlamaForCausalLM:
        """
        Gets the smaug model.
        """
        from transformers import LlamaForCausalLM
        return LlamaForCausalLM.from_pretrained(
            self.SMAUG_MODEL_PATH,
            cache_dir=self.model_dir,
            device_map=self.device,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            quantization_config=self.quantization_config,
        )

    def get_smaug_tokenizer(self) -> LlamaTokenizer:
        """
        Gets the smaug tokenizer.
        """
        from transformers import LlamaTokenizer
        return LlamaTokenizer.from_pretrained(
            self.SMAUG_MODEL_PATH,
            cache_dir=self.model_dir,
        )

    @timed_fn()
    def get_smaug_pipeline(self) -> Pipeline:
        """
        Gets the smaug text generation pipeline.
        """
        from transformers import pipeline
        return pipeline(
            "text-generation",
            model=self.get_smaug_model(),
            tokenizer=self.get_smaug_tokenizer(),
            torch_dtype=self.dtype,
            device_map=self.device
        )

    """Luxia Models"""

    @timed_fn()
    def get_luxia_model(self) -> LlamaForCausalLM:
        """
        Gets the luxia model.
        """
        from transformers import LlamaForCausalLM
        return LlamaForCausalLM.from_pretrained(
            self.LUXIA_MODEL_PATH,
            cache_dir=self.model_dir,
            device_map=self.device,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            quantization_config=self.quantization_config,
        )

    def get_luxia_tokenizer(self) -> LlamaTokenizer:
        """
        Gets the luxia tokenizer.
        """
        from transformers import LlamaTokenizer
        return LlamaTokenizer.from_pretrained(
            self.LUXIA_MODEL_PATH,
            cache_dir=self.model_dir,
        )

    @timed_fn()
    def get_luxia_pipeline(self) -> Pipeline:
        """
        Gets the luxia text generation pipeline.
        """
        from transformers import pipeline
        return pipeline(
            "text-generation",
            model=self.get_luxia_model(),
            tokenizer=self.get_luxia_tokenizer(),
            torch_dtype=self.dtype,
            device_map=self.device
        )

    """Zephyr Models"""

    @timed_fn()
    def get_zephyr_model(self) -> MistralForCausalLM:
        """
        Gets the zephyr model.
        """
        from transformers import MistralForCausalLM
        return MistralForCausalLM.from_pretrained(
            self.ZEPHYR_MODEL_PATH,
            cache_dir=self.model_dir,
            device_map=self.device,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            quantization_config=self.quantization_config,
        )

    def get_zephyr_tokenizer(self) -> LlamaTokenizer:
        """
        Gets the zephyr tokenizer.
        """
        from transformers import LlamaTokenizer
        return LlamaTokenizer.from_pretrained(
            self.ZEPHYR_MODEL_PATH,
            cache_dir=self.model_dir,
        )

    @timed_fn()
    def get_zephyr_pipeline(self, use_llama_cpp: bool=True) -> Union[Pipeline, Llama]:
        """
        Gets the zephyr text generation pipeline.
        """
        from enfugue.diffusion.util import llama_cpp_available
        if use_llama_cpp and llama_cpp_available():
            from llama_cpp import Llama
            model_path, filename = self.GGUF_ZEPHYR_MODEL
            return Llama.from_pretrained(
                model_path,
                filename=filename,
                n_gpu_layers=-1,
                main_gpu=self.device.index,
                cache_dir=self.model_dir,
                verbose=False,
                n_ctx=0,
                chat_format="mistral-instruct"
            )
        from transformers import pipeline
        return pipeline(
            "text-generation",
            model=self.get_zephyr_model(),
            tokenizer=self.get_zephyr_tokenizer(),
            torch_dtype=self.dtype,
            device_map=self.device
        )

    """Gemma Models"""

    @timed_fn()
    def get_gemma_model(self) -> GemmaForCausalLM:
        """
        Gets the gemma model.
        """
        from transformers import GemmaForCausalLM

        return GemmaForCausalLM.from_pretrained(
            self.GEMMA_MODEL_PATH,
            cache_dir=self.model_dir,
            device_map=self.device,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            quantization_config=self.quantization_config,
        )

    def get_gemma_tokenizer(self) -> GemmaTokenizer:
        """
        Gets the gemma tokenizer.
        """
        from transformers import GemmaTokenizer
        return GemmaTokenizer.from_pretrained(
            self.GEMMA_MODEL_PATH,
            cache_dir=self.model_dir,
        )

    @timed_fn()
    def get_gemma_pipeline(self) -> Pipeline:
        """
        Gets the gemma text generation pipeline.
        """
        from transformers import pipeline
        return pipeline(
            "text-generation",
            model=self.get_gemma_model(),
            tokenizer=self.get_gemma_tokenizer(),
            torch_dtype=self.dtype,
            device_map=self.device
        )

    """DanTagGen Models"""

    @timed_fn()
    def get_dtg_model(self) -> LlamaForCausalLM:
        """
        Gets the dtg model.
        """
        from transformers import LlamaForCausalLM
        return LlamaForCausalLM.from_pretrained(
            self.DTG_MODEL_PATH,
            cache_dir=self.model_dir,
            device_map=self.device,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            quantization_config=self.quantization_config,
        )

    def get_dtg_tokenizer(self) -> LlamaTokenizer:
        """
        Gets the dtg tokenizer.
        """
        from transformers import LlamaTokenizer
        return LlamaTokenizer.from_pretrained(
            self.DTG_MODEL_PATH,
            cache_dir=self.model_dir,
        )

    @timed_fn()
    def get_dtg_pipeline(self, use_llama_cpp: bool=True) -> Union[Pipeline, Llama]:
        """
        Gets the dtg text generation pipeline.
        """
        from enfugue.diffusion.util import llama_cpp_available
        if use_llama_cpp and llama_cpp_available():
            from llama_cpp import Llama
            model_path, filename = self.GGUF_DTG_MODEL
            return Llama.from_pretrained(
                model_path,
                filename=filename,
                n_gpu_layers=-1,
                main_gpu=self.device.index,
                cache_dir=self.model_dir,
                verbose=False,
                n_ctx=512
            )
        from transformers import pipeline
        return pipeline(
            "text-generation",
            model=self.get_dtg_model(),
            tokenizer=self.get_dtg_tokenizer(),
            torch_dtype=self.dtype,
            device_map=self.device
        )

    """Context Managers"""

    @contextmanager
    def moondream(self) -> Iterator[LanguageSupportModelImageProcessor]:
        """
        Gets the callable function that allows probing of an image.
        """
        with self.context():
            processor = MoondreamSupportModelImageProcessor(
                model=self.get_moondream_model(),
                tokenizer=self.get_moondream_tokenizer()
            )
            yield processor
            del processor.model
            del processor.tokenizer
            del processor

    @contextmanager
    def llava(self) -> Iterator[LanguageSupportModelImageProcessor]:
        """
        Gets the callable function that allows probing of an image.
        """
        with self.context():
            processor = LanguageSupportModelImageProcessor(
                pipeline=self.get_llava_pipeline()
            )
            yield processor
            del processor.pipeline
            del processor

    @contextmanager
    def llava_next(
        self,
        variant: LLAVA_NEXT_VARIANT_LITERAL="mistral"
    ) -> Iterator[LanguageSupportModelImageProcessor]:
        """
        Gets the callable function that allows probing of an image.
        """
        with self.context():
            if variant == "full":
                prompt_format = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{prompt}<|im_end|><|im_start|>assistant\n"
            elif variant in ["vicuna", "vicuna-small"]:
                prompt_format = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT:"
            else:
                prompt_format="[INST] <image>\n{prompt} [/INST]"
            processor = LanguageSupportModelImageProcessor(
                pipeline=self.get_llava_next_pipeline(variant),
                prompt_format=prompt_format,
                use_pipeline=False
            )
            yield processor
            del processor.pipeline
            del processor

    @contextmanager
    def vip_llava(self) -> Iterator[LanguageSupportModelImageProcessor]:
        """
        Gets the callable function that allows probing of an image.
        """
        with self.context():
            processor = LanguageSupportModelImageProcessor(
                pipeline=self.get_vip_llava_pipeline(),
                prompt_format="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{prompt}###Assistant:"
            )
            yield processor
            del processor.pipeline
            del processor

    @contextmanager
    def probe(
        self,
        model: VISION_MODEL_VARIANT_LITERAL="llava-next",
        *args: Any,
        **kwargs: Any
    ) -> Iterator[LanguageSupportModelTextProcessor]:
        """
        Gets the callable function for asking questions about an image
        """
        get_processor = {
            "moondream": self.moondream,
            "llava": self.llava,
            "vip-llava": self.vip_llava,
            "llava-next": self.llava_next
        }.get(model, None)
        if get_processor is None:
            raise ValueError(f"Unknown model {model}") # type: ignore[unreachable]
        with get_processor(*args, **kwargs) as processor:
            yield processor

    @contextmanager
    def converse(
        self,
        role: Optional[str]=None,
        model: LANGUAGE_MODEL_LITERAL="zephyr",
        use_llama_cpp: bool=True,
        safe: bool=True,
        temperature: float=0.7,
        top_k: int=50,
        top_p: float=0.95,
        seed: Optional[int]=None,
        system: Optional[str]=None,
        use_system: bool=True,
    ) -> Iterator[LanguageSupportModelTextProcessor]:
        """
        Gets the callable function for conversations
        """
        from transformers.generation.configuration_utils import GenerationConfig
        get_pipeline = {
            "zephyr": self.get_zephyr_pipeline,
            "gemma": self.get_gemma_pipeline,
            "smaug": self.get_smaug_pipeline,
            "luxia": self.get_luxia_pipeline,
            "dtg": self.get_dtg_pipeline
        }.get(model, None)
        if get_pipeline is None:
            raise ValueError(f"Unknown model {model}") # type: ignore[unreachable]
        with self.context():
            model = get_pipeline(use_llama_cpp=use_llama_cpp)
            processor_cls = LlamaCppSupportModelTextProcessor if model.__class__.__name__ == "Llama" else TransformerSupportModelTextProcessor
            processor = processor_cls(
                model,
                role=self.get_role(role),
                safe=safe,
                seed=seed,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                system=system,
                use_system=model != "gemma" and use_system
            )
            yield processor
            del processor.model
            del processor

    @contextmanager
    def caption_upsampler(self, safe: bool = True, use_llama_cpp: bool=True) -> Iterator[LanguageSupportModelTextProcessor]:
        """
        A shortcut for self.converse(model='zephyr', role='caption')
        """
        with self.converse(model="zephyr", role="caption", safe=safe, use_llama_cpp=use_llama_cpp) as processor:
            yield processor

    @contextmanager
    def tag_generator(self, safe: bool=True, use_llama_cpp: bool=True) -> Iterator[LanguageSupportModelTextProcessor]:
        """
        A shortcut for self.converse(model='dtg', role='tag', safe=safe)
        """
        with self.converse(model="dtg", role="tag", safe=safe, use_llama_cpp=use_llama_cpp) as processor:
            yield processor
