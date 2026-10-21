# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm, Qwen2_5_VLVisionFlashAttention2, Qwen2_5_VLVisionBlock
import torch.nn as nn
from compression_method.selector_scorer import TransformerScorer
from compression_method.selector_model import (
   qwen25vl_vision_tower_forward_selector,
   qwen25vl_generation_forward_selector 
)
import types
import random
import numpy as np

local_rank = None

class ScheduledWeightTrainer(Trainer):
    """
    自定义训练器，支持动态调整正则化权重。
    
    该训练器继承自Hugging Face的Trainer类，主要功能：
    1. 在训练过程中动态调整正则化权重（从reg_weight_start到reg_weight_end）
    2. 处理边界框相关字段的过滤
    """
    def __init__(self, *args, reg_weight_start=0.1, reg_weight_end=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        # 正则化权重起始值
        self.reg_weight_start = reg_weight_start
        # 正则化权重结束值
        self.reg_weight_end = reg_weight_end

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写损失计算方法，动态调整正则化权重并处理边界框相关字段。
        
        参数:
            model: 训练模型
            inputs: 输入数据批次
            return_outputs: 是否返回输出
            num_items_in_batch: 批次中的项目数量
        
        功能:
        1. 根据训练进度动态计算正则化权重
        2. 过滤掉不需要传递给模型的边界框字段
        3. 调用父类的损失计算方法
        """
        total_steps = self.state.max_steps
        current_step = self.state.global_step

        if total_steps > 0:
            # 计算训练进度，确保不超过1.0
            progress = min(current_step / total_steps, 1.0)
            # 线性插值计算当前正则化权重
            current_weight = self.reg_weight_start + (self.reg_weight_end - self.reg_weight_start) * progress
        else:
            # 如果总步数尚未确定，使用起始权重
            current_weight = self.reg_weight_start

        # 将计算出的权重设置到实际模型上
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model.regularization_weight = current_weight
        actual_model.bbox_loss_weight = getattr(self.args, "bbox_loss_weight", 1.0)

        # 清理遗留字段：当前BBox监督走ref_token_masks，不再使用对话bbox相关字段
        inputs.pop("image_bboxes", None)
        inputs.pop("has_bbox", None)
        inputs.pop("conversation_bboxes", None)

        # 只在优化器步骤记录日志；compute_loss在每个微步骤运行，避免每个微步骤记录日志

        # 调用父类的损失计算方法
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

def set_seed(seed: int):
    """
    设置随机种子以确保实验的可重现性。
    
    参数:
        seed: 随机种子值
    """
    random.seed(seed)                # Python随机数种子
    np.random.seed(seed)              # NumPy随机数种子
    torch.manual_seed(seed)           # PyTorch CPU随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)          # 当前GPU随机数种子
        torch.cuda.manual_seed_all(seed)      # 所有GPU随机数种子
    torch.backends.cudnn.deterministic = True  # 确保CNN算法确定性
    torch.backends.cudnn.benchmark = False     # 关闭自动优化以保证可重现性


def rank0_print(*args):
    """
    只在主进程（rank 0）打印信息，避免多进程训练时重复输出。
    
    参数:
        *args: 要打印的参数
    """
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    安全地保存模型状态字典到磁盘，兼容DeepSpeed和普通训练。
    
    参数:
        trainer: Hugging Face训练器实例
        output_dir: 输出目录路径
    """
    if trainer.deepspeed:
        # DeepSpeed情况下的特殊处理
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    # 普通情况下的模型保存
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        # 将参数转移到CPU以节省显存
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    """
    根据训练参数设置模型不同组件的梯度更新状态。
    
    参数:
        model_args: 模型参数配置
        model: 要配置的模型实例
    """
    # 设置视觉编码器的可训练性
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True   # 启用视觉编码器参数更新
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False  # 冻结视觉编码器参数
    
    # 设置视觉-语言融合层（merger）的可训练性
    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True   # 启用融合层参数更新
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False  # 冻结融合层参数
    
    # 设置语言模型主体的可训练性
    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True   # 启用语言模型参数更新
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = True   # 启用语言模型头部参数更新
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False  # 冻结语言模型参数
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = False  # 冻结语言模型头部参数
    
    # -------------------------添加压缩器调优设置---------------------------------
    # 设置重要性评分器的可训练性
    if model_args.tune_compressor:
        for n, p in model.visual.importance_scorer.named_parameters():
            p.requires_grad = True   # 启用重要性评分器参数更新
    else:
        for n, p in model.visual.importance_scorer.named_parameters():
            p.requires_grad = False  # 冻结重要性评分器参数
    # -------------------------------------------------------------------------------

def train(attn_implementation="flash_attention_2"):
    """
    主训练函数，负责模型加载、配置和训练流程的初始化。
    
    参数:
        attn_implementation: 注意力实现方式，默认使用flash_attention_2
    """
    global local_rank

    # 解析命令行参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # 保留自定义字段（如ref_token_masks）在批次中
    training_args.remove_unused_columns = False

    # 为预算设置合理的默认值，避免配置/检查点中出现None值
    if model_args.budget is None:
        model_args.budget = 0.2

    spatial_ratio = getattr(model_args, "spatial_ratio", None)
    if spatial_ratio is None:
        spatial_ratio = getattr(model_args, "contextual_ratio", None)
    spatial_ratio = float(0.1 if spatial_ratio is None else spatial_ratio)
    model_args.spatial_ratio = spatial_ratio
    model_args.contextual_ratio = spatial_ratio

    print('seed:', training_args.seed)
    print('data_seed:', training_args.data_seed)
    set_seed(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # 加载模型和配置
    if "qwen2.5" in model_args.model_name_or_path.lower():
        # 首先加载配置以获取LLM隐藏层大小
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        # 将预算持久化到视觉配置中，以便检查点携带这些标志
        config.vision_config.budgets = model_args.budget
        config.vision_config.spatial_ratio = spatial_ratio
        config.vision_config.contextual_ratio = spatial_ratio
        config.vision_config.cosine_penalty_weight = float(
            getattr(model_args, "cosine_penalty_weight", 0.15)
        )
        config.vision_config.cosine_penalty_threshold = float(
            getattr(model_args, "cosine_penalty_threshold", 0.85)
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor

        data_args.model_type = "qwen2.5vl"
    else:
        raise ValueError("Model not currently supported")
        
    # ----------------------------添加压缩器设置-------------------------------
    model.visual.budgets = model_args.budget
    model.visual.config.budgets = model_args.budget
    model.visual.spatial_ratio = spatial_ratio
    model.visual.contextual_ratio = spatial_ratio
    model.visual.config.spatial_ratio = spatial_ratio
    model.visual.config.contextual_ratio = spatial_ratio
    model.visual.cosine_penalty_weight = max(
        0.0, float(getattr(model_args, "cosine_penalty_weight", 0.15))
    )
    model.visual.config.cosine_penalty_weight = model.visual.cosine_penalty_weight
    model.visual.cosine_penalty_threshold = max(
        -1.0, min(1.0, float(getattr(model_args, "cosine_penalty_threshold", 0.85)))
    )
    model.visual.config.cosine_penalty_threshold = model.visual.cosine_penalty_threshold
    # 替换视觉塔的前向传播方法为支持选择器的版本
    model.visual.forward = types.MethodType(qwen25vl_vision_tower_forward_selector, model.visual)
    
    # 初始化 Scorer（纯视觉）
    if "3b" in model_args.model_name_or_path.lower():
        print("Initializing Scorer for 3B ...")
        # 确保使用正确的视觉维度
        # 通常3B使用小型ViT，但如果可能应该使用配置值
        # 如果配置检查失败，硬编码2048作为原始脚本中的值，但理想情况下从vision_config读取
        vis_dim = 2048 
        if hasattr(model.visual.config, "out_hidden_size"):
             vis_dim = model.visual.config.out_hidden_size

        # 初始化3B模型的TransformerScorer
        model.visual.importance_scorer = TransformerScorer(
            in_features=vis_dim,                    # 视觉特征维度
            hidden_dim=vis_dim // 2                 # 隐藏层维度（视觉维度的一半）
        )
    elif "7b" in model_args.model_name_or_path.lower():   
        print("Initializing Scorer for 7B ...")
        vis_dim = 3584
        if hasattr(model.visual.config, "out_hidden_size"):
             vis_dim = model.visual.config.out_hidden_size

        # 初始化7B模型的TransformerScorer
        model.visual.importance_scorer = TransformerScorer(
            in_features=vis_dim,                    # 视觉特征维度
            hidden_dim=vis_dim // 2                 # 隐藏层维度（视觉维度的一半）
        )
    else:
        raise ValueError("Model not currently supported")

    # 替换模型的前向传播方法为支持选择器的版本
    model.forward = types.MethodType(qwen25vl_generation_forward_selector, model)

    # 设置边界框监督损失权重
    model.bbox_loss_weight = getattr(training_args, 'bbox_loss_weight', 1.0)
    # -------------------------------------------------------------------------------

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False


    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    # --- 打印可训练参数 ---
    if local_rank == 0: 
        print("="*80)
        print("Printing trainable parameters...")
        trainable_param_names = []
        
        # 收集所有可训练参数的名称
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_param_names.append(name)
        
        # 打印所有可训练参数名称
        for name in trainable_param_names:
            print(f"- {name}")

        print("="*80)
    # -----------------------------------

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = ScheduledWeightTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        reg_weight_start=training_args.reg_weight_start,  # Set starting weight
        reg_weight_end=training_args.reg_weight_end,    # Set ending weight
        **data_module
    )


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # 训练完成后，复制预处理器配置和聊天模板文件
    if local_rank == 0:
        source_dir = pathlib.Path(model_args.model_name_or_path)    # 源模型目录
        dest_dir = pathlib.Path(training_args.output_dir)          # 输出目录
        
        preprocessor_file = "preprocessor_config.json"             # 预处理器配置文件
        chat_template_file = "chat_template.json"                  # 聊天模板文件
        
        # 复制预处理器配置文件
        source_preprocessor_path = source_dir / preprocessor_file
        dest_preprocessor_path = dest_dir / preprocessor_file
        if source_preprocessor_path.exists():
            shutil.copy(source_preprocessor_path, dest_preprocessor_path)
            rank0_print(f"Copied {source_preprocessor_path} to {dest_preprocessor_path}")
        else:
            rank0_print(f"Warning: {source_preprocessor_path} not found.")

        # 复制聊天模板文件
        source_chat_template_path = source_dir / chat_template_file
        dest_chat_template_path = dest_dir / chat_template_file
        if source_chat_template_path.exists():
            shutil.copy(source_chat_template_path, dest_chat_template_path)
            rank0_print(f"Copied {source_chat_template_path} to {dest_chat_template_path}")
        else:
            rank0_print(f"Warning: {source_chat_template_path} not found.")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
