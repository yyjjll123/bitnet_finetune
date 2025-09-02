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
```
--model_path：                  预训练模型的权重文件路径。用于加载已有模型参数。
--dataset_path：                训练用的数据集（JSON格式）路径。
--tokenizer_path：              分词器模型路径。用于文本分词。
--save_path：                   微调后模型保存路径。
--epochs：                      训练轮数（epoch），即所有数据集被训练的次数。
--batch_size：                  每次训练的样本数量。
--seq_len：                     最大序列长度，输入文本被截断或填充到该长度。
--learning_rate：               优化器的学习率。
--gradient_accumulation_steps： 梯度累积步数，累计一定步数后再更新参数，适合显存有限时使用。
--num_workers：                 用于数据加载的子进程数量。
--dataset_fraction：            使用数据集的比例（如0.5表示只用一半数据）。
--validation_split_fraction：   验证集占总数据的比例。
--eval_interval：               每隔多少步进行一次验证。
--max_grad_norm：               梯度裁剪的最大范数，防止梯度爆炸。
--early_stopping_patience：     早停策略的耐心值，验证损失多次未提升则提前停止训练。
--early_stopping_min_delta：    验证损失提升的最小阈值，低于该值不算提升。
```

## 注意：

本代码在调试时发现可能是由于文本过于简单或者是预训练效果较好，学习率提高后容易出现梯度爆炸，且数据集验证成功率非常高，因此微调的学习率较低。
