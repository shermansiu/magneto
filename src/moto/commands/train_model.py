from __future__ import annotations
import functools
import typing as tp
import datasets
import transformers
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


increment_en = [
    {"input": "One", "target": "Two"},
    {"input": "Three", "target": "Four"},
    {"input": "Five", "target": "Six"},
    {"input": "Seven", "target": "Eight"},
    {"input": "Nine", "target": "Ten"},
]
increment_en = increment_en * 100


def lod_to_dol(list_of_dicts: tp.List[tp.Dict[str, tp.Any]]) -> tp.Dict[str, list]:
    dict_of_lists = {
        key: [dct[key] for dct in list_of_dicts] for key in list_of_dicts[0]
    }
    return dict_of_lists


increment_en = lod_to_dol(increment_en)


def preprocess_function_(
    examples,
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    max_target_length: int,
):
    inputs = examples["input"]
    targets = examples["target"]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    tokenizer = transformers.T5TokenizerFast.from_pretrained("google/flan-t5-small")
    model = transformers.T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    args = Seq2SeqTrainingArguments(
        "script_debug",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        fp16=False,
        push_to_hub=False,
        # sharded_ddp=["zero_dp_3"],
        max_steps=10000,
        logging_steps=5000,
        save_steps=5000
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    dataset = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(increment_en),
            "test": datasets.Dataset.from_dict(increment_en),
        }
    )

    preprocess_function = functools.partial(
        preprocess_function_,
        tokenizer=tokenizer,
        max_input_length=512,
        max_target_length=512
    )

    processed_ds = dataset.map(preprocess_function, batched=True)
    processed_ds.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=processed_ds["train"],
        eval_dataset=processed_ds["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
