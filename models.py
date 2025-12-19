"""
Implements Facebook's RoBERTa model for question answering and
JetBrains' Mellum-4b-base model for Python code generation
from Hugging Face with LoRA fine-tuning.
"""

import evaluate
import os
# import json
import torch
import time

from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, AutoModelForQuestionAnswering, default_data_collator, \
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

from load_data import rebuild_splits, prepare_train_features, prepare_eval_features, \
    postprocess_qa_predictions, prepare_prompt_completion, tokenize


class RoBERTaQA:
    def __init__(self, epochs: int = 3, lr: float = 2e-5,
                 batch_size: int = 32, lora: bool = False,
                 train_size: float = 1.0, lora_rank: int = 1) -> None:
        self.model_name = 'FacebookAI/roberta-base'
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.data_source = 'rajpurkar/squad_v2'
        self.data_collator = default_data_collator
        self.train_size = train_size
        self.lora_rank = lora_rank
        self.dataset = None
        self.train_features = None
        self.dev_features = None
        self.test_features = None
        self.trainer = None
        self.test_mode = False

        self.output_dir = f'{self.model_name}-base'

        if lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.QUESTION_ANS,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=lora_rank*2,
                lora_dropout=0.1,
                target_modules=["query", "value"],
            )
            self.model = get_peft_model(self.model, self.peft_config)
            self.output_dir = f'{self.model_name}-lora'

        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            log_level='info',
            logging_strategy="epoch",
            load_best_model_at_end=True,
        )

    def prepare_data(self):
        data = load_dataset(self.data_source, split='train+validation')
        self.dataset = rebuild_splits(data, self.train_size)
        self.train_features = self.dataset['train'].map(
            prepare_train_features,
            batched=True,
            remove_columns=self.dataset['train'].column_names,
            fn_kwargs={'tokenizer': self.tokenizer}
        )
        self.dev_features = self.dataset['dev'].map(
            prepare_train_features,
            batched=True,
            remove_columns=self.dataset['dev'].column_names,
            fn_kwargs={'tokenizer': self.tokenizer}
        )
        self.test_features = self.dataset['test'].map(
            prepare_train_features,
            batched=True,
            remove_columns=self.dataset['test'].column_names,
            fn_kwargs={'tokenizer': self.tokenizer}
        )

    def compute_metrics(self, p):
        metric = evaluate.load('squad_v2')
        features = self.dataset['dev'].map(
            prepare_eval_features,
            batched=True,
            remove_columns=self.dataset['dev'].column_names,
            fn_kwargs={'tokenizer': self.tokenizer}
        )
        pred = postprocess_qa_predictions(self.dataset['dev'], features, p.predictions, self.tokenizer)
        ref = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.dataset['dev']]

        if self.test_mode:
            features = self.dataset['test'].map(
                prepare_eval_features,
                batched=True,
                remove_columns=self.dataset['test'].column_names,
                fn_kwargs={'tokenizer': self.tokenizer}
            )
            pred = postprocess_qa_predictions(self.dataset['test'], features, p.predictions, self.tokenizer)
            ref = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.dataset['test']]

        # print(f"\npred: {pred[:10]}", flush=True)
        # print(f"\nref: {ref[:10]}", flush=True)
        return metric.compute(predictions=pred, references=ref)

    def train(self):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_features,
            eval_dataset=self.dev_features,
            data_collator=self.data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        self.trainer.can_return_loss = True
        self.test_mode = False
        self.trainer.train()
        print(flush=True)

    def eval(self):
        self.test_mode = True
        results = self.trainer.evaluate(self.test_features)
        print(results, flush=True)


class PyCodeModel:
    def __init__(self, epochs: int = 3, lr: float = 2e-5,
                 batch_size: int = 32, lora: bool = False,
                 train_size: float = 1.0, lora_rank: int = 8) -> None:
        self.model_name = 'JetBrains/Mellum-4b-base'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.data_source = 'flytech/python-codes-25k'
        self.epochs = epochs
        self.lr = lr
        self.train_size = train_size
        self.lora_rank = lora_rank
        self.dataset = None
        self.test = None
        self.peft_config = None
        self.trainer = None

        self.output_dir = f'{self.model_name}-base'
        self.data_dir = '/data/ldial/gen_output'
        bnb_config = None

        if lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=lora_rank*2,
                lora_dropout=0.1,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            )
            self.output_dir = f'{self.model_name}-lora'
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.output_dir = self.output_dir + f'-{time.strftime("%H%M%S")}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          dtype=torch.bfloat16,
                                                          quantization_config=bnb_config,
                                                          trust_remote_code=True)
        self.model = prepare_model_for_kbit_training(self.model)

        self.sft_args = SFTConfig(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            pad_token=self.tokenizer.eos_token,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            max_length=128,
            packing=False,
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            num_train_epochs=epochs,
            weight_decay=0.01,
            log_level='info',
            logging_strategy="epoch",
            load_best_model_at_end=True,
            activation_offloading=True,
            optim="paged_adamw_32bit",
            use_liger_kernel=True,
            completion_only_loss=True,
            eval_accumulation_steps=30,
        )

    def prepare_data(self):
        data = load_dataset(self.data_source)
        data = Dataset.from_generator(prepare_prompt_completion, gen_kwargs={'examples': data['train']})
        dataset = rebuild_splits(data, self.train_size)
        self.test = dataset['test']
        self.dataset = dataset.map(tokenize, batched=True, remove_columns=dataset['train'].column_names, fn_kwargs={'tokenizer': self.tokenizer})

    def compute_metrics(self, p):
        pass

    # def output_writer(self, output):
    #     lora_type = f'lora-{self.lora_rank}' if self.peft_config else 'base'
    #     output_file = os.path.join(self.data_dir, f'ts-{self.train_size}_{lora_type}_test.json')
    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         json.dump(output, f, ensure_ascii=False, indent=2)
    #     return output_file

    def train(self):
        trainer = SFTTrainer(
            model=self.model,
            args=self.sft_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['dev'],
            processing_class=self.tokenizer,
            peft_config=self.peft_config,
        )
        trainer.train()
        trainer.save_model(self.output_dir)
        print(flush=True)

    def eval(self):
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            base_model,
            self.output_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.output_dir,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model = model.merge_and_unload()
        model.save_pretrained(
            f"{self.output_dir}/model_merged",
            safe_serialization=True,
        )

        def tokenize_prompts(examples, tkn):
            return tkn(examples['prompt'], return_tensors='pt', padding=True)

        self.test.set_format(type='torch')
        inputs = self.test.map(tokenize_prompts, batched=True, fn_kwargs={'tkn': tokenizer})
        inputs = inputs[:]['input_ids'].to('cuda')
        model = AutoModelForCausalLM.from_pretrained(
            f"{self.output_dir}/model_merged",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        outputs = model.generate(inputs,
                                 max_new_tokens=512,
                                 repetition_penalty=1.1,
                                 stop_strings=["<|endoftext|>", "\n\n"],
                                 eos_token_id=[tokenizer.eos_token_id],
                                 tokenizer=tokenizer,
                                 do_sample=True,)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.test = self.test.add_column(name='pred', column=preds)
        self.test.to_json(f"{self.data_dir}/test.jsonl")
        print(f"Output file: {self.output_dir}/test.jsonl", flush=True)
