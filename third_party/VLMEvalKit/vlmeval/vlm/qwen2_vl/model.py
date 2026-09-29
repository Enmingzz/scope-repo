from __future__ import annotations

import os
import sys
import warnings
import math
import logging
import re
import time

import torch
from policy.policy_model import PolicyModel
from transformers import StoppingCriteria

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_gpu_memory, listinstr
from ...dataset import DATASET_MODALITY

VLLM_MAX_IMAGE_INPUT_NUM = 24


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def create_image_content(image_path, min_pixels, max_pixels):
    base64_image, mime_type = encode_image(image_path)
    return {
        "type": "image",
        "image": f"data:{mime_type};base64,{base64_image}",
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    }


def encode_image(image_path, max_side=None):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

    from PIL import Image
    image = Image.open(image_path)
    # Handle the alpha channel
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    if max_side:
        image = _resize_image(image, max_side)
    encoded_image = _encode_image(image, image_format)

    return encoded_image, mime_type


def _encode_image(image, image_format):
    from io import BytesIO
    with BytesIO() as output:
        image.convert("RGB").save(output, format=image_format)
        import base64
        base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_encoded_data


def _rgba_to_rgb(image):
    from PIL import Image
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _resize_image(image, max_side):
    resize_scale = max_side / max(image.size)
    new_size = (
        int(image.size[0] * resize_scale),
        int(image.size[1] * resize_scale),
    )
    return image.resize(new_size)


def process_video(video_path, num_frames, min_pixels, max_pixels):
    import cv2
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # the sampling rate using max number of frames
    sampling_gap_maxframe = (
        1 if not num_frames else math.ceil(frame_count / num_frames)
    )
    sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)

    frame_number = 0
    images = []

    while True:
        import tempfile
        success, frame = cap.read()
        if not success:
            break
        # Sample frames based on the dynamic sampling rate
        if frame_number % sampling_gap == 0:
            # Create a temporary file for the frame
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp_frame:
                cv2.imwrite(temp_frame.name, frame)
                images.append(create_image_content(temp_frame.name, min_pixels, max_pixels))
                os.remove(temp_frame.name)
        frame_number += 1
    if frame_number == 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")
    logging.info(
        f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
    )
    cap.release()
    return images


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"  # noqa: E501

UNTIL = ["<|diff_marker|>"]


class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_kv_cache: bool = True,

        enable_visionzip=False,
        visionzip_ratio=0.0,

        enable_divprune=False,
        divprune_ratio=0.0,

        enable_kdvz=False,
        kdvz_ratio=0.0,

        enable_kd_tokens=False,
        tokens_anchor='all',
        tokens_ratio=0.0,
        tokens_prune_layers='7,15,23',

        num_return_sequences=1,
        
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        use_audio_in_video: bool = False,
        enable_policy = False,
        policy_path = None,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        if self.total_pixels and self.total_pixels > 24576 * 28 * 28:
            print('The total number of video tokens might become too large, resulting in an overly long input sequence. We recommend lowering **total_pixels** to below **24576 × 28 × 28**.')  # noqa: E501

        if temperature > 0.01 and num_return_sequences > 1:
            do_sample=True
            top_p = 0.9
            top_k = None
        else:
            do_sample=False
            top_p = 0.001
            top_k = 1

        if enable_policy: enable_kdvz=True

        self.total_inference_time_in_seconds = 0
        self.num_total_dataset_generated_tokens = 0
        self.total_initial_input_visual_tokens = 0
        self.total_input_text_tokens = 0
        self.enable_thinking = kwargs.get('enable_thinking', True)
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            use_cache=use_kv_cache,

            enable_policy=enable_policy,

            enable_visionzip=enable_visionzip,
            visionzip_ratio=visionzip_ratio,

            enable_divprune=enable_divprune,
            divprune_ratio=divprune_ratio,
            
            enable_kdvz=enable_kdvz,
            kdvz_ratio=kdvz_ratio,
            
            enable_kd_tokens=enable_kd_tokens,
            tokens_anchor=tokens_anchor,
            tokens_ratio=tokens_ratio,
            tokens_prune_layers=tokens_prune_layers,
            
            num_return_sequences=num_return_sequences
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.adapter_path = kwargs.get('adapter_path', '') or ''
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        if self.fps is None and self.nframe is None:
            print("Warning: fps and nframe are both None, \
                  using default nframe/fps setting in qwen-vl-utils/qwen-omni-utils, \
                  the fps/nframe setting in video dataset is omitted")
        self.use_audio_in_video = use_audio_in_video
        self.FRAME_FACTOR = 2
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None

        from transformers import AutoConfig
        # Composite configs can forward instance attributes to the text config.
        model_type = type(AutoConfig.from_pretrained(model_path)).model_type
        if model_type == 'qwen2_5_omni':
            try:
                from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356")  # noqa: E501
                raise err
            MODEL_CLS = Qwen2_5OmniForConditionalGeneration
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        elif model_type == 'qwen2_5_vl':
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        elif model_type == 'qwen2_vl':
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)
        else:
            raise ValueError(f'Unsupported model_type for this Qwen2/2.5 evaluator: {model_type}')

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0
        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag between `use_vllm` and `use_lmdeploy` to True"  # noqa: E501

        if self.use_vllm:
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 8:
                tp_size = 8
            elif gpu_count >= 4:
                tp_size = 4
            elif gpu_count >= 2:
                tp_size = 2
            else:
                tp_size = 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            import os
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    'VLLM_WORKER_MULTIPROC_METHOD is not set to spawn.'
                    'Use \'export VLLM_WORKER_MULTIPROC_METHOD=spawn\' to avoid potential multi-process issues'
                )
            self.llm = LLM(
                model=self.model_path,
                model_impl="transformers",
                max_num_seqs=5,
                max_model_len=32768,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
                enforce_eager=True,
            )

        elif self.use_lmdeploy:
            from lmdeploy import TurbomindEngineConfig, pipeline, ChatTemplateConfig
            num_gpus = torch.cuda.device_count()
            self.model = pipeline(
                model_path,
                backend_config=TurbomindEngineConfig(session_len=32768, cache_max_entry_count=0.1, tp=num_gpus),
                chat_template_config=ChatTemplateConfig(model_name='qwen2d5-vl'))
            torch.cuda.set_device(0)
            self.device = 'cuda'
        else:
            if listinstr(['2.5', '2_5', 'qwen25'], model_path.lower()):
                self.model = MODEL_CLS.from_pretrained(
                    model_path, torch_dtype='auto', device_map="auto", attn_implementation='flash_attention_2'
                )
                self.model.eval()
            else:
                self.model = MODEL_CLS.from_pretrained(
                    model_path, dtype='bfloat16', device_map="auto", attn_implementation='flash_attention_2'
                )
                self.model.eval()
            if self.adapter_path:
                from peft import PeftModel

                print('Loading PEFT adapter:', self.adapter_path)
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
                self.model.eval()

        from pprint import pprint
        print("Generate kwargs:")
        pprint(self.generate_kwargs)
        self.enable_policy = enable_policy
        if enable_policy:
            assert policy_path is not None and policy_path != ''
            print('Policy is enabled. Init policy model with checkpoint', policy_path)
            self.policy_model = PolicyModel(policy_path, device=self.model.device)
            self.policy_model.eval()
        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        image_count = sum(1 for s in inputs if s['type'] == 'image')
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                elif self._use_total_image_budget(dataset, image_count):
                    min_pixels, max_pixels = self._per_image_pixel_budget(image_count)
                    item['min_pixels'] = min_pixels
                    item['max_pixels'] = max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': ensure_video_url(s['value'])
                }
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            elif s['type'] == 'audio':
                item = {'type':'audio','audio':s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _use_total_image_budget(self, dataset: str | None, image_count: int) -> bool:
        if image_count <= 1 or dataset is None:
            return False
        return 'MMMU_PRO' in dataset.upper()

    def _per_image_pixel_budget(self, image_count: int) -> tuple[int, int]:
        token_area = 28 * 28
        total_min_tokens = int(os.getenv('MMMU_PRO_TOTAL_MIN_IMAGE_TOKENS', self.min_pixels // token_area if self.min_pixels else 1280))
        total_max_tokens = int(os.getenv('MMMU_PRO_TOTAL_MAX_IMAGE_TOKENS', self.max_pixels // token_area if self.max_pixels else 4096))
        per_min_tokens = max(1, total_min_tokens // image_count)
        per_max_tokens = max(per_min_tokens, total_max_tokens // image_count)
        return per_min_tokens * token_area, per_max_tokens * token_area

    def _prepare_content_vllm(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        video_inputs = [s for s in inputs if s['type'] == 'video']
        video_count = len(video_inputs)
        image_count = sum(1 for s in inputs if s['type'] == 'image')
        cur_image_count = 0
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                elif self._use_total_image_budget(dataset, image_count):
                    min_pixels, max_pixels = self._per_image_pixel_budget(image_count)
                    item['min_pixels'] = min_pixels
                    item['max_pixels'] = max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if cur_image_count < self.limit_mm_per_prompt:
                    content.append(item)
                    cur_image_count += 1
                else:
                    logging.warning(
                        f"Number of images exceeds the limit of {self.limit_mm_per_prompt}. "
                        f"Only the first {self.limit_mm_per_prompt} images will be used."
                    )
            elif s['type'] == 'video':
                if video_count > 1:
                    logging.warning(
                        "Multiple videos detected. Using video frames for each video"
                    )
                    if dataset == 'OCRBench':
                        min_pixels = 10 * 10 * 28 * 28
                        warnings.warn(f"OCRBench dataset uses custom min_pixels={min_pixels}")
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    else:
                        if self.min_pixels is not None:
                            min_pixels = self.min_pixels
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()

                    frames_per_video = max(1, self.limit_mm_per_prompt // video_count)
                    content.append({"type": "text", "text": "<video frames start>"})
                    content.extend(process_video(s['value'], frames_per_video, min_pixels, max_pixels))
                    content.append({"type": "text", "text": "<video frames end>"})

                else:
                    item = {
                        'type': 'video',
                        'video': ensure_video_url(s['value'])
                    }
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                    if self.total_pixels is not None:
                        item['total_pixels'] = self.total_pixels
                    if self.fps is not None:
                        item['fps'] = self.fps
                    elif self.nframe is not None:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
                    content.append(item)
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
                content.append(item)
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        return content

    def extract_answer_from_thinking_response(self, response):
        # Use regular expression to find all content between <answer> and </answer> tags
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(answer_pattern, response, re.DOTALL)
        
        # If there are multiple matches, concatenate all of them into a single string
        if matches:
            return ''.join(matches)
        
        # If there is no match, return the original response as a fallback
        return response

    def generate_inner_transformers(self, message, dataset=None):
        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        if listinstr(['omni'], self.model_path.lower()):
            audios, images, videos = process_mm_info([messages], use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(text=text, images=images,audio=audios, videos=videos, padding=True, return_tensors='pt',use_audio_in_video=self.use_audio_in_video)  # noqa: E501
        else:
            images, videos = process_vision_info([messages])
            inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')  # noqa: E501
        inputs = inputs.to('cuda')

        sample_initial_visual_tokens = 0
        if 'video_grid_thw' in inputs:
            sample_initial_visual_tokens = inputs.video_grid_thw.prod().item() // 4
        elif 'image_grid_thw' in inputs:
            sample_initial_visual_tokens = inputs.image_grid_thw.prod().item() // 4
        else:
            raise ValueError("Either image or video must be passed")

        sample_text_tokens = inputs.input_ids.shape[1] - sample_initial_visual_tokens

        self.total_initial_input_visual_tokens += sample_initial_visual_tokens
        self.total_input_text_tokens += sample_text_tokens

        if listinstr(['omni'], self.model_path.lower()):
            self.generate_kwargs['use_audio_in_video'] = self.use_audio_in_video
            self.generate_kwargs['return_audio'] = False

        if self.enable_policy:
            # breakpoint()
            if 'pixel_values' in inputs:
                embeds, _, _ = self.model.model.get_image_features(
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs.get("image_grid_thw"),
                    enable_kdvz=self.generate_kwargs.get("enable_kdvz", False),
                    enable_visionzip=self.generate_kwargs.get("enable_visionzip", False),
                )
            elif 'pixel_values_videos' in inputs:
                embeds, _, _ = self.model.model.get_image_features(
                    pixel_values=inputs["pixel_values_videos"],
                    image_grid_thw=inputs.get("image_grid_thw"),
                    enable_kdvz=self.generate_kwargs.get("enable_kdvz", False),
                    enable_visionzip=self.generate_kwargs.get("enable_visionzip", False),
                )
            else:
                raise ValueError()
            embeds = torch.stack(embeds, dim=0)
            K = self.policy_model(embeds)
            # with open('/users/minh.le1/policy_log.txt', 'a', encoding='utf-8') as f: f.write(f"Policy K: {str(int(K))}\n")
            self.generate_kwargs['num_return_sequences'] = K
            if K > 1:
                self.generate_kwargs['temperature'] = 0.7
                self.generate_kwargs['do_sample'] = True
                self.generate_kwargs['top_p'] = 0.9
                self.generate_kwargs['top_k'] = None
            else:
                self.generate_kwargs['temperature'] = 0.01
                self.generate_kwargs['do_sample'] = False
                self.generate_kwargs['top_p'] = 0.001
                self.generate_kwargs['top_k'] = 1

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids.repeat(generated_ids.shape[0],1), generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # import pdb; pdb.set_trace()

        responses = []
        for response in out:
            if self.post_process:
                resp = response.split('\\boxed{')[-1]
                lt = len(resp)
                counter, end = 1, None
                for i in range(lt):
                    if resp[i] == '{':
                        counter += 1
                    elif resp[i] == '}':
                        counter -= 1
                    if counter == 0:
                        end = i
                        break
                    elif i == lt - 1:
                        end = lt
                        break
                if end is not None:
                    response = resp[:end]

            if self.verbose:
                print(f'\033[32m{response}\033[0m')
            responses.append(response)
        return responses

    def generate_inner_lmdeploy(self, message, dataset=None):
        from lmdeploy import GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            top_p=self.generate_kwargs['top_p'],
            top_k=self.generate_kwargs['top_k'],
            temperature=self.generate_kwargs['temperature'],
            repetition_penalty=self.generate_kwargs['repetition_penalty'],
        )
        gen_config.random_seed = None
        messages_list = self.message_to_lmdeploy(message, system_prompt=self.system_prompt)
        assert len(messages_list) == 1
        response = self.model(messages_list, gen_config=gen_config)[0]
        response = response.text
        return response

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams

        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content_vllm(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if listinstr(['omni'], self.model_path.lower()):
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            images, videos = process_vision_info(messages)
        print('finishing process vision info in vllm.')

        if DATASET_MODALITY(dataset) == 'VIDEO' and 'megabench' not in dataset.lower():
            assert len(videos) == 1
            videos_nd = [videos[0].detach().cpu().numpy().transpose(0, 2, 3, 1)]

            video_inputs = {
                "prompt": text[0],
                "multi_modal_data": {"video": videos_nd[0]},
                "mm_processor_kwargs":{}
            }
            if self.use_audio_in_video:
                import vllm
                assert not vllm.envs.VLLM_USE_V1, ("V1 does not support use_audio_in_video. Please launch this example with `VLLM_USE_V1=0`.")  # noqa: E501
                video_inputs["multi_modal_data"]["audio"] = audios[0]
                video_inputs['mm_processor_kwargs']['use_audio_in_video'] = True
            if videos_nd[0].shape[0] > VLLM_MAX_IMAGE_INPUT_NUM:
                print('video input sequence may be too long for vllm, Maybe cannot generate response for VLLM')
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=self.max_new_tokens, stop_token_ids=None
        )
        if images:
            outputs = self.llm.generate(
                {
                    "prompt": 1,
                    "multi_modal_data": {"image": images},
                },
                sampling_params=sampling_params,
            )
        elif videos_nd:
            outputs = self.llm.generate(
                video_inputs,
                sampling_params=sampling_params,
            )
        else:
            outputs = self.llm.generate(
                {
                    "prompt": text,
                },
                sampling_params=sampling_params,
            )

        responses = []
        for o in outputs:
            for i in range(len(o.outputs)):
                generated_text = o.outputs[i].text

                if self.post_process:
                    resp = generated_text.split('\\boxed{')[-1]
                    lt = len(resp)
                    counter, end = 1, None
                    for i in range(lt):
                        if resp[i] == '{':
                            counter += 1
                        elif resp[i] == '}':
                            counter -= 1
                        if counter == 0:
                            end = i
                            break
                        elif i == lt - 1:
                            end = lt
                            break
                    if end is not None:
                        generated_text = resp[:end]

                if self.verbose:
                    print(f'\033[32m{generated_text}\033[0m')
                responses.append(generated_text)
        return responses

    def generate_inner(self, message, dataset=None):
        # import pdb; pdb.set_trace()
        if any(item['type'] == 'video' for item in message):
            video_path = None
            for i in range(len(message)):
                if not (message[i]['type'] == 'text'):
                    video_path = message[i]['value']
                    break

            if 'mvbench' in video_path.lower():
                cleaned_messages = []
                cleaned_messages.append(message[1])
                cleaned_messages.append(message[2])

            elif 'tempcompass' in video_path.lower():
                cleaned_messages = []
                cleaned_messages.append(message[0])
                cleaned_messages.append(message[1])

            elif "q-bench" in video_path.lower():
                cleaned_messages = []
                cleaned_messages.append(message[1])
                cleaned_messages.append(message[2])

            elif "videommmu" in video_path.lower():
                cleaned_messages = []
                cleaned_messages.append(message[0])
                text_prompt = message[2]
                removal = text_prompt['value'].find('Add `Answer:')
                text_prompt['value'] = text_prompt['value'][:removal]
                cleaned_messages.append(text_prompt)

            else: # video-MME or Video-TT
                cleaned_messages = []
                for i in range(len(message)):
                    if not (message[i]['type'] == 'text'):
                        cleaned_messages.append(message[i])
                    if message[i]['type'] == 'text' and i == len(message) - 1:

                        cleaned_messages.append(message[i])

            message = cleaned_messages

        # import pdb; pdb.set_trace()
        if self.enable_thinking:
            for i in range(len(message)):
                if message[i]['type'] == 'text':
                    message[i]['value'] = message[i]['value'].replace('\nAnswer:', '')
                    message[i]['value'] += "\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
                    break

        start_time = time.time()

        if self.use_vllm:
            responses = self.generate_inner_vllm(message, dataset=dataset)
        elif self.use_lmdeploy:
            responses = self.generate_inner_lmdeploy(message, dataset=dataset)
        else:
            responses = self.generate_inner_transformers(message, dataset=dataset)
        
        # with open('/users/minh.le1/policy_log.txt', 'a', encoding='utf-8') as f: f.write(f"Responses: {str(responses)}\n\n")

        self.total_inference_time_in_seconds += time.time() - start_time
        final_answers = []
        # import pdb; pdb.set_trace()
        for response in responses:
            raw_response = response
            num_generated_tokens = self.processor.tokenizer(response, return_tensors="pt").input_ids.shape[1]
            self.num_total_dataset_generated_tokens += num_generated_tokens
            if self.enable_thinking:
                response = self.extract_answer_from_thinking_response(response)
            parsed_response = response.replace('\n', ' ').strip()
            if os.environ.get('VLMEVAL_SAVE_RAW_PREDICTION', '1') != '0':
                final_answers.append({
                    'prediction': parsed_response,
                    'extra_records': {
                        'raw_prediction': raw_response,
                        'parsed_prediction': parsed_response,
                        'generated_token_len': int(num_generated_tokens),
                    },
                })
            else:
                final_answers.append(parsed_response)
        
        return final_answers[0] if len(final_answers) == 1 else final_answers


class Qwen2VLChatAguvis(Qwen2VLChat):
    def __init__(self, mode=None, **kwargs):
        self.mode = mode
        super().__init__(**kwargs)
        self.processor.max_pixels = self.max_pixels
        self.processor.min_pixels = self.min_pixels

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )
            raise err

        messages = []
        user_message = []
        for item in message:
            if "role" in item.keys():
                if item["role"] == "system":
                    self.system_prompt = item["value"]
                else:
                    item.pop("role")
                    user_message.append(item)
            else:
                user_message.append(item)
        message = user_message

        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {"role": "user", "content": self._prepare_content(message, dataset=dataset)}
        )
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            chat_template=CHAT_TEMPLATE,
        )
        # TODO: provide current action's low-level instruction
        # if False:
        #     # If low-level instruction is provided
        #     # We enforce using "Action: {low_level_instruction} to guide generation"
        #     recipient_text = f"<|im_start|>assistant<|recipient|>all\nAction: {low_level_instruction}\n"
        if self.mode == "force-plan":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nThought: "
        elif self.mode == "force-plan-l1":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nAction: "
        elif self.mode == "force-plan-l3":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nObservation: "
        elif self.mode == "grounding":
            recipient_text = "<|im_start|>assistant<|recipient|>os\n"
        elif self.mode == "force-plan-free":
            recipient_text = "<|im_start|>assistant<|recipient|>all\n"
        elif self.mode == "self-plan":
            recipient_text = "<|im_start|>assistant<|recipient|>"
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        text += recipient_text
        # print(text)

        images, videos = process_vision_info([messages])
        inputs = self.processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # stop_str = "<|diff_marker|>"
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(
        #     keywords, self.processor.tokenizer, inputs.input_ids
        # )

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
            # stopping_criteria=[stopping_criteria],
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        # for term in UNTIL:
        #     if len(term) > 0:
        #         response = response.split(term)[0]

        if self.post_process:
            resp = response.split("\\boxed{")[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == "{":
                    counter += 1
                elif resp[i] == "}":
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response
