#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small standalone chat REPL for qualitatively testing a base model with an
optional LoRA adapter on top (e.g. one of the Pareto-optimal adapters saved
by --save-pareto-adapters-dir). Plain transformers/peft, deliberately not
heretic's own Model class -- that class sets up its own fresh LoRA structure
for abliteration, which isn't what you want when loading a specific saved
adapter checkpoint back for testing.

Usage:
    python chat_test.py --base-model /path/to/base [--adapter /path/to/adapter]
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to a LoRA adapter directory. Omit to chat with the base model as-is.",
    )
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Loading base model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype="auto", device_map="auto"
    )

    if args.adapter:
        print(f"Loading LoRA adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    chat = [{"role": "system", "content": args.system_prompt}]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print()
    print("Chat ready. Empty message or Ctrl+C/Ctrl+D to exit.")
    print()

    while True:
        try:
            message = input("User: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not message.strip():
            break

        chat.append({"role": "user", "content": message})
        inputs = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        print("Assistant: ", end="")
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                streamer=streamer,
            )
        response = tokenizer.decode(
            output[0][inputs.shape[-1] :], skip_special_tokens=True
        )
        chat.append({"role": "assistant", "content": response})
        print()


if __name__ == "__main__":
    main()
