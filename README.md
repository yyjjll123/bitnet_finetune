# bitnet_finetune

# 项目内容：
本项目为复现1.58bit的微调代码，本复现代码基于https://github.com/microsoft/BitNet；

本项目微调使用的数据集为alpaca数据集


# 使用方法：

## 1.下载模型并复现GPU推理

下载https://github.com/microsoft/BitNet的代码，并根据代码的readme文件复现GPU推理

## 2.数据集下载

terminal输入：

```
wget https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/raw/main/alpaca_data.json -O /root/（your path）/BitNet-main/gpu/checkpoints/alpaca_data.json
```

## 3.开始调试

将finetune_bitnet.py放入BitNet-main/gpu/文件夹中，随后在BitNet-main/gpu的地址下输入下面代码：

```
python finetune_bitnet.py
--model_path /root/lanyun-tmp/BitNet-main/gpu/checkpoints/model_state_int2.pt
--dataset_path /root/lanyun-tmp/BitNet-main/gpu/checkpoints/alpaca_data.json
--tokenizer_path /root/lanyun-tmp/BitNet-main/gpu/tokenizer.model
--save_path /root/lanyun-tmp/BitNet-main/gpu/checkpoints/finetuned_bitnet_with_eval.pt
--epochs 1
--batch_size 1
--seq_len 512
--learning_rate 5e-6
--gradient_accumulation_steps 8
--num_workers 4
--dataset_fraction 0.5
--validation_split_fraction 0.3
--eval_interval 200
--max_grad_norm 1.0
--early_stopping_patience 3
--early_stopping_min_delta 0.001
--early_stopping_patience 2
```

微调代码解释：

