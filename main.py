"""
LLM training and evaluation runner.
"""

import argparse

from models import RoBERTaQA, PyCodeModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # task options
    parser.add_argument(
        '--task',
        type=str,
        choices=['qa', 'gen'],
        required=True,
    )
    # epochs
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
    )
    # batch size
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
    )
    # learning rate
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5,
    )
    # pretrained vs finetuned
    parser.add_argument(
        '--tuning',
        type=bool,
        default=False,
    )
    # training size
    parser.add_argument(
        '--train_size',
        type=float,
        default=1.0,
    )
    # lora rank
    parser.add_argument(
        '--rank',
        type=int,
        default=8,
    )

    args = parser.parse_args()

    if args.task == 'qa':
        print(80 * '=', flush=True)
        print('Question Answering', flush=True)
        print(80 * '=', flush=True)
        task_model = RoBERTaQA(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, lora=args.tuning,
                             train_size=args.train_size, lora_rank=args.rank)

    elif args.task == 'gen':
        print(80 * '=', flush=True)
        print('Python Code Generation', flush=True)
        print(80 * '=', flush=True)
        task_model = PyCodeModel(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, lora=args.tuning,
                             train_size=args.train_size, lora_rank=args.rank)

    print(task_model.model.config, flush=True)
    task_model.prepare_data()
    task_model.train()
    if args.task == 'qa':
        task_model.model.save_pretrained(task_model.output_dir)
    task_model.eval()