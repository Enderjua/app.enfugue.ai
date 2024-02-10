from __future__ import annotations

import os

from enfugue.diffusion.support.model import SupportModel, SupportModelProcessor
from enfugue.util import logger

from typing import Any, Union, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from enfugue.diffusion.audio.fam.llm.sample import ( # type: ignore
        Model as MetaVoiceModel
    )
    from enfugue.diffusion.audio.fam.quantiser.audio.speaker_encoder.model import ( # type: ignore
        SpeakerEncoder
    )
    from torch import Tensor

from contextlib import contextmanager

__all__ = [
    "AudioSupportModel",
    "TextToAudioProcessor",
    "MetaVoiceProcessor",
]

class TextToAudioProcessor(SupportModelProcessor):
    pass

class MetaVoiceProcessor(TextToAudioProcessor):
    def __init__(
        self,
        first_stage_model: MetaVoiceModel,
        second_stage_model: MetaVoiceModel,
        encoder: SpeakerEncoder,
        default_embedding: Union[Tensor, str],
        **kwargs: Any
    ) -> None:
        super(MetaVoiceProcessor, self).__init__(**kwargs)
        self.first_stage_model = first_stage_model
        self.second_stage_model = second_stage_model
        self.encoder = encoder
        self.default_embedding = default_embedding

    def __call__(
        self,
        texts: Union[str, List[str]],
        embeddings: List[Union[str, Tensor]] = [],
        enhancer: Optional[Literal["df"]] = "df",
        guidance_scale: Optional[float] = 3.0,
        max_new_tokens: int = 864,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        temperature: float = 1.0,
        batch_size: int=1,
    ) -> List[str]:
        """
        Samples the audio model
        """
        import torch
        from enfugue.diffusion.util import load_state_dict
        speaker_embeddings: List[torch.Tensor] = []
        if embeddings is None or (isinstance(embeddings, list) and not embeddings):
            embeddings = [self.default_embedding]
        for embedding in embeddings:
            if isinstance(embedding, torch.Tensor):
                speaker_embeddings.append(embedding)
            elif isinstance(embedding, str):
                basename, ext = os.path.splitext(os.path.basename(embedding))
                if ext in [".bin", ".pt", ".safetensors"]:
                    logger.debug(f"Loading embeddings from {basename}")
                    speaker_embeddings.append(
                        load_state_dict(embedding).to(
                            self.first_stage_model.config.device
                        )
                    )
                else:
                    logger.debug(f"Extracting embeddings from {basename}")
                    speaker_embeddings.append(
                        self.encoder.embed_utterance_from_file(
                            embedding,
                            numpy=False
                        ).unsqueeze(0)
                    )
            else:
                raise IOError(f"Can't handle embedding of type {type(embeddings)}") # type: ignore[unreachable]
        speaker_embeddings = torch.cat(speaker_embeddings, dim=0)
        if not isinstance(texts, list): 
            texts = [texts]
        tokens = self.first_stage_model(
            texts=texts,
            speaker_embs=speaker_embeddings,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        wav_files = [
            f"{wav_file}.wav" for wav_file in self.second_stage_model(
                texts=texts,
                encodec_tokens=tokens,
                speaker_embs=speaker_embeddings,
                batch_size=batch_size,
                guidance_scale=None,
                top_p=None,
                top_k=top_k,
                temperature=temperature,
                max_new_tokens=None
            )
        ]

        if enhancer:
            from enfugue.diffusion.audio.fam.llm.enhancers import get_enhancer
            enhancer_model = get_enhancer(enhancer)
            output_dir = self.second_stage_model.config.output_dir
            for i, wav_file in enumerate(wav_files):
                basename, ext = os.path.splitext(os.path.basename(wav_file))
                enhanced_file = os.path.join(os.path.dirname(wav_file), f"{basename}_enhanced.wav")
                enhancer_model(wav_file, enhanced_file)
                wav_files[i] = enhanced_file
                try:
                    os.remove(wav_file)
                except:
                    pass

        return wav_files

class AudioSupportModel(SupportModel):
    METAVOICE_FIRST_STAGE_PATH = "https://huggingface.co/metavoiceio/metavoice-1B-v0.1/resolve/main/first_stage.pt"
    METAVOICE_SECOND_STAGE_PATH = "https://huggingface.co/metavoiceio/metavoice-1B-v0.1/resolve/main/second_stage.pt"
    METAVOICE_DEFAULT_EMBEDDING_PATH = "https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3"

    @contextmanager
    def metavoice(self) -> Iterator[MetaVoiceProcessor]:
        from enfugue.diffusion.audio.fam.llm.sample import ( # type: ignore
            Model,
            InferenceConfig
        )
        from enfugue.diffusion.audio.fam.llm.decoders import ( # type: ignore
            EncodecDecoder
        )
        from enfugue.diffusion.audio.fam.quantiser.audio.speaker_encoder.model import ( # type: ignore
            SpeakerEncoder
        )
        from enfugue.diffusion.audio.fam.quantiser.text.tokenise import ( # type: ignore
            TrainedBPETokeniser
        )
        from enfugue.diffusion.audio.fam.llm.adapters import ( # type: ignore
            FlattenedInterleavedEncodec2Codebook,
            TiltedEncodec,
        )
        from pibble.util.files import TempfileContext
        first_stage_path = self.get_model_file(
            self.METAVOICE_FIRST_STAGE_PATH
        )
        second_stage_path = self.get_model_file(
            self.METAVOICE_SECOND_STAGE_PATH
        )
        default_embedding_path = self.get_model_file(
            self.METAVOICE_DEFAULT_EMBEDDING_PATH
        )
        with self.context():
            tempfile_context = TempfileContext()
            with tempfile_context:
                encoder = SpeakerEncoder(device=self.device, eval=True)
                default_embedding = encoder.embed_utterance_from_file(
                    default_embedding_path,
                    numpy=False
                ).unsqueeze(0)
                first_stage_config = InferenceConfig(
                    ckpt_path=first_stage_path,
                    output_dir=tempfile_context.directory,
                    device=str(self.device),
                    dtype=str(self.dtype).split(".")[1],
                    init_from="resume"
                )
                first_stage_adapter = FlattenedInterleavedEncodec2Codebook(
                    end_of_audio_token=1024
                )
                first_stage = Model(
                    first_stage_config,
                    TrainedBPETokeniser,
                    EncodecDecoder,
                    data_adapter_fn=first_stage_adapter.decode,
                )
                second_stage_config = InferenceConfig(
                    ckpt_path=second_stage_path,
                    output_dir=tempfile_context.directory,
                    device=str(self.device),
                    dtype=str(self.dtype).split(".")[1],
                    init_from="resume"
                )
                second_stage_adapter = TiltedEncodec(end_of_audio_token=1024)
                second_stage = Model(
                    second_stage_config,
                    TrainedBPETokeniser,
                    EncodecDecoder,
                    data_adapter_fn=second_stage_adapter.decode,
                )
                processor = MetaVoiceProcessor(
                    first_stage_model=first_stage,
                    second_stage_model=second_stage,
                    encoder=encoder,
                    default_embedding=default_embedding
                )
                yield processor
                del processor
                del second_stage
