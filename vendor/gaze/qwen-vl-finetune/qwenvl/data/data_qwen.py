import os
import copy
import json
import random
import logging
import time
import math
import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
# from torchcodec.decoders import VideoDecoder
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def read_mask_supervision_jsonl(path):
    fields = ("sample_id", "mask_token_labels", "mask_loss_valid", "bbox_budget_mask")
    rows = []
    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            rows.append({field: record[field] for field in fields})
    return rows


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        # Only insert if we still have an image length available; otherwise drop extra <image> tags
                        if grid_thw_image and visual_replicate_index_image < len(grid_thw_image):
                            image_len = grid_thw_image[visual_replicate_index_image]
                            visual_replicate_index_image += 1
                            replacement = (
                                "<|vision_start|>"
                                + f"<|image_pad|>" * image_len
                                + "<|vision_end|>"
                            )
                            new_parts.append(replacement)
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|video_pad|>"
                            * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def alpaca_to_conversations(sample: Dict) -> Optional[List[Dict[str, str]]]:
    """
    Convert Alpaca-style sample (instruction/input/output) into the conversation
    format expected by preprocess_qwen_2_visual.

    Expected fields:
      - instruction: str (often contains "<image>\\n...")
      - input: str (optional, may be empty)
      - output: str
    """
    instruction = sample.get("instruction")
    output = sample.get("output")
    if instruction is None or output is None:
        return None

    user_text = str(instruction)
    input_text = sample.get("input")
    if input_text:
        input_text = str(input_text)
        if input_text.strip():
            sep = "" if user_text.endswith("\n") else "\n"
            user_text = f"{user_text}{sep}{input_text}"

    # If this is an image sample but the prompt forgot to include <image>, add it.
    if (("image" in sample) or ("images" in sample)) and "<image>" not in user_text:
        user_text = f"<image>\n{user_text}"

    return [
        {"from": "human", "value": user_text},
        {"from": "gpt", "value": str(output)},
    ]


def get_sample_conversations(sample: Dict) -> List[Dict]:
    conversations = sample.get("conversations")
    if isinstance(conversations, list) and conversations:
        return conversations
    alpaca_convs = alpaca_to_conversations(sample)
    return alpaca_convs if alpaca_convs is not None else []


def pixel_bboxs_to_ref_token_mask(
    bboxs: Optional[List[List[float]]],
    width: Optional[float],
    height: Optional[float],
    grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int = 2,
) -> Optional[torch.Tensor]:
    """
    Build a token-grid mask (after spatial merge) from pixel-space bboxes.

    Args:
        bboxs: List of [x1,y1,x2,y2] in pixels (may be float).
        width/height: Original image size used by bboxs.
        grid_thw: Patch grid before merge: [T,H,W] from image_processor.
        spatial_merge_size: Merge size (Qwen2.5-VL default 2).

    Returns:
        BoolTensor of shape [H_tokens, W_tokens] (union of all bboxes), or None.
    """
    if bboxs is None or len(bboxs) == 0:
        return None
    if width is None or height is None:
        return None
    if grid_thw is None or not isinstance(grid_thw, torch.Tensor) or grid_thw.numel() != 3:
        return None

    try:
        width_f = float(width)
        height_f = float(height)
    except (TypeError, ValueError):
        return None
    if width_f <= 0 or height_f <= 0:
        return None

    t, h, w = grid_thw.to(dtype=torch.long).tolist()
    h_tokens = h // spatial_merge_size
    w_tokens = w // spatial_merge_size
    if h_tokens <= 0 or w_tokens <= 0:
        return None

    mask = torch.zeros((h_tokens, w_tokens), dtype=torch.bool)
    num_valid = 0

    for bbox in bboxs:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError):
            continue

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        x1 = max(0.0, min(1.0, x1 / width_f))
        x2 = max(0.0, min(1.0, x2 / width_f))
        y1 = max(0.0, min(1.0, y1 / height_f))
        y2 = max(0.0, min(1.0, y2 / height_f))

        x1_grid = int(x1 * w_tokens)
        y1_grid = int(y1 * h_tokens)
        x2_grid = min(int(x2 * w_tokens), w_tokens - 1)
        y2_grid = min(int(y2 * h_tokens), h_tokens - 1)

        x1_grid = max(0, min(w_tokens - 1, x1_grid))
        y1_grid = max(0, min(h_tokens - 1, y1_grid))
        if x2_grid < x1_grid:
            x2_grid = x1_grid
        if y2_grid < y1_grid:
            y2_grid = y1_grid

        mask[y1_grid : y2_grid + 1, x1_grid : x2_grid + 1] = True
        num_valid += 1

    if num_valid == 0:
        return None
    if not mask.any():
        return None
    return mask


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            mask_supervision_path = data.get("mask_supervision_path")
            if mask_supervision_path is not None:
                mask_supervision = read_mask_supervision_jsonl(mask_supervision_path)
                if len(mask_supervision) != len(annotations):
                    raise ValueError(
                        f"Mask supervision length mismatch: {len(mask_supervision)} != {len(annotations)}"
                    )
                for row_idx, (annotation, supervision) in enumerate(
                    zip(annotations, mask_supervision), start=1
                ):
                    if annotation.get("sample_id") != supervision["sample_id"]:
                        raise ValueError(
                            f"Mask supervision sample_id mismatch at row {row_idx}: "
                            f"{annotation.get('sample_id')} != {supervision['sample_id']}"
                        )
                    annotation.update(supervision)
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if (("image" in sample) or ("images" in sample)) else 0
            conversations = get_sample_conversations(sample)
            if conversations:
                length_list.append(
                    sum(len((conv.get("value") or conv.get("content") or "").split()) for conv in conversations)
                    + img_tokens
                )
            else:
                length_list.append(img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            conversations = get_sample_conversations(sample)
            cur_len = sum(
                len((conv.get("value") or conv.get("content") or "").split()) for conv in conversations
            )
            cur_len = (
                cur_len if (("image" in sample) or ("images" in sample) or ("video" in sample)) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 3
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(video_file)
                return decord_video
                if decord_video:
                    break
            except Exception as e:
                print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        torchcodec_video = None
        # try:
        #     torchcodec_video = self.video_torchcodec(video_file)
        #     return torchcodec_video
        # except Exception as e:
        #     print(f"torchcodec attempt failed: {e}")
        print(f"torchcodec attempt failed: {e}")

    def video_decord(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    # def video_torchcodec(self, video_file):
    #     device = "cpu"  # or e.g. "cuda"
    #     decoder = VideoDecoder(video_file, device=device)
    #     total_frames = decoder.metadata.num_frames
    #     avg_fps = decoder.metadata.average_fps
    #     video_length = total_frames / avg_fps
    #     interval = getattr(self.data_args, "base_interval", 4)

    #     num_frames_to_sample = round(video_length / interval)
    #     video_min_frames = getattr(self.data_args, "video_min_frames", 4)
    #     video_max_frames = getattr(self.data_args, "video_max_frames", 8)

    #     target_frames = min(
    #         max(num_frames_to_sample, video_min_frames), video_max_frames
    #     )
    #     frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    #     frame_idx = np.unique(frame_idx)
    #     frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
    #     video = frame_batch.data.cpu().numpy()
    #     return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None
        # print(sources[0])
        ref_token_masks: Optional[List[Optional[torch.Tensor]]] = None
        mask_token_labels: Optional[List[torch.Tensor]] = None
        mask_loss_valid: Optional[List[bool]] = None

        if ("image" in sources[0]) or ("images" in sources[0]):
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i].get("image")
            if image_file is None:
                image_file = self.list_data_dict[i].get("images")
            if isinstance(image_file, List):
                # Use the first image for training.
                image_file = image_file[:1]
                image_file = image_file[0]

            clean_path = str(image_file)
            # Remove 'cot/' prefix if present (dataset uses this prefix, actual images don't)
            if clean_path.startswith("cot/"):
                clean_path = clean_path.replace("cot/", "", 1)

            image_file = os.path.join(image_folder, clean_path)
            image, grid_thw = self.process_image_unified(image_file)
            image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]

            expected_tokens = int(grid_thw_merged[0])
            bbox_budget = self.list_data_dict[i]["bbox_budget_mask"]
            one_mask_labels = self.list_data_dict[i]["mask_token_labels"]
            if len(bbox_budget) != expected_tokens or len(one_mask_labels) != expected_tokens:
                raise ValueError(
                    f"{self.list_data_dict[i]['sample_id']}: token supervision length mismatch "
                    f"(bbox={len(bbox_budget)}, mask={len(one_mask_labels)}, expected={expected_tokens})"
                )
            ref_token_masks = [torch.tensor(bbox_budget, dtype=torch.bool)]
            mask_token_labels = [torch.tensor(one_mask_labels, dtype=torch.int8)]
            mask_loss_valid = [bool(self.list_data_dict[i]["mask_loss_valid"])]
        if "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, video_grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, video_grid_thw, second_per_grid_ts = self.process_video(
                        video_file
                    )
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, video_grid_thw, second_per_grid_ts = self.process_video(
                    video_file
                )
                video = [video]
            video_grid_thw_merged = copy.deepcopy(video_grid_thw)
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw_merged = [video_grid_thw_merged]
                video_grid_thw = [video_grid_thw]
            video_grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in video_grid_thw_merged
            ]
        chat_sources = [copy.deepcopy(get_sample_conversations(self.list_data_dict[i]))]
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        if ("image" not in sources[0] and "images" not in sources[0]) and "video" not in sources[0]:
            # print(sources[0])
            grid_thw_merged = None
            sources = copy.deepcopy([get_sample_conversations(e) for e in sources])
            # data_dict = preprocess_qwen_2_visual(
            #     sources, self.tokenizer, grid_thw=grid_thw_merged
            # )
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if ("image" in self.list_data_dict[i]) or ("images" in self.list_data_dict[i]):
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw], dim=0
            )
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
            data_dict["video_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
            )

        # BBox budget masks and three-value Mask ranking labels are loaded from the aligned sidecar.
        data_dict["ref_token_masks"] = ref_token_masks
        data_dict["mask_token_labels"] = mask_token_labels
        data_dict["mask_loss_valid"] = mask_loss_valid

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids

        # Batch token-level reference masks (aligned with concatenated images)
        if concat_images is not None:
            ref_token_masks_batch: List[Optional[torch.Tensor]] = []
            any_ref = False
            for instance in instances:
                if "pixel_values" not in instance:
                    continue
                if "image_grid_thw" in instance and instance["image_grid_thw"] is not None:
                    num_imgs = int(instance["image_grid_thw"].shape[0])
                else:
                    masks = instance.get("ref_token_masks")
                    if isinstance(masks, list):
                        num_imgs = len(masks)
                    elif isinstance(masks, torch.Tensor):
                        num_imgs = 1
                    else:
                        num_imgs = 0
                masks = instance.get("ref_token_masks")
                if masks is None:
                    ref_token_masks_batch.extend([None] * num_imgs)
                    continue
                if isinstance(masks, torch.Tensor):
                    masks_list = [masks]
                else:
                    masks_list = list(masks)
                if len(masks_list) < num_imgs:
                    masks_list = masks_list + [None] * (num_imgs - len(masks_list))
                masks_list = masks_list[:num_imgs]
                if any(m is not None for m in masks_list):
                    any_ref = True
                ref_token_masks_batch.extend(masks_list)
            batch["ref_token_masks"] = ref_token_masks_batch if any_ref else None
        else:
            batch["ref_token_masks"] = None

        if concat_images is not None:
            batch["mask_token_labels"] = list(
                itertools.chain.from_iterable(instance["mask_token_labels"] for instance in instances)
            )
            batch["mask_loss_valid"] = list(
                itertools.chain.from_iterable(instance["mask_loss_valid"] for instance in instances)
            )
        else:
            batch["mask_token_labels"] = None
            batch["mask_loss_valid"] = None

        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        # Batch token-level reference masks (aligned with concatenated images)
        if concat_images is not None:
            ref_token_masks_batch: List[Optional[torch.Tensor]] = []
            any_ref = False
            for instance in instances:
                if "pixel_values" not in instance:
                    continue
                if "image_grid_thw" in instance and instance["image_grid_thw"] is not None:
                    num_imgs = int(instance["image_grid_thw"].shape[0])
                else:
                    masks = instance.get("ref_token_masks")
                    if isinstance(masks, list):
                        num_imgs = len(masks)
                    elif isinstance(masks, torch.Tensor):
                        num_imgs = 1
                    else:
                        num_imgs = 0
                masks = instance.get("ref_token_masks")
                if masks is None:
                    ref_token_masks_batch.extend([None] * num_imgs)
                    continue
                if isinstance(masks, torch.Tensor):
                    masks_list = [masks]
                else:
                    masks_list = list(masks)
                if len(masks_list) < num_imgs:
                    masks_list = masks_list + [None] * (num_imgs - len(masks_list))
                masks_list = masks_list[:num_imgs]
                if any(m is not None for m in masks_list):
                    any_ref = True
                ref_token_masks_batch.extend(masks_list)
            batch["ref_token_masks"] = ref_token_masks_batch if any_ref else None
        else:
            batch["ref_token_masks"] = None

        if concat_images is not None:
            batch["mask_token_labels"] = list(
                itertools.chain.from_iterable(instance["mask_token_labels"] for instance in instances)
            )
            batch["mask_loss_valid"] = list(
                itertools.chain.from_iterable(instance["mask_loss_valid"] for instance in instances)
            )
        else:
            batch["mask_token_labels"] = None
            batch["mask_loss_valid"] = None

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
