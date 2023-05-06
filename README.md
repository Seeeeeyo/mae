## Masked Autoencoders: A PyTorch Implementation

## Steps



Use python 3.8 (should work).
I tried only with python 3.10 but had to to the following changes:


——------------ 
In “/usr/local/lib/python3.10/dist-packages/timm/models/layers/helpers.py”, Add 
```
Import torch 
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
```
——------------ 

In “/content/mae/util/misc.py”, change 
```from torch._six import inf```
By 
```from torch import inf```
——------------ 

-- Download the data from the drive 'data_60k' (full medmnist dataset) and 'data_sampled.zip' (fractions of medmnist dataset).
-- Download a medical classification dataset. I let you choose one, as you know these better than I do. Let's try to find one which is kind of similar to MedMnist to hopefully reach some performances. The data structure should be as follow: 
	- eval_data
		- train 
			-class1
				-img1
				-img2
				-...
			-class2
			-...

Then, 


1) 
```git clone https://github.com/Seeeeeyo/mae.git```
----

2) ```cd mae```

----

3) ```wget -nc https://dl.fbaipublicfiles.com/mae/finetune/mae_pretrain_vit_base.pth```

----

4) 
```
!pip install submitit
!pip install timm==0.3.2
```
----

5) Evaluate the mae_vit_base on eval_data
```python main_finetune.py --eval --resume mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 32 --data_path 'eval_data'```

----
6) FINETUNE

```
python main_finetune.py \
    --accum_iter 1 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune 'mae_pretrain_vit_base.pth' \
    --epochs 50 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path 'AGI/data_60k'    
```


 OR  (couldn't try it without cluster and multiple GPU):


```
python submitit_finetune.py \
	--job_dir ${JOB_DIR} \
    --nodes 4 \ TO CHANGE 
    --batch_size 32 \ 
    --model vit_base_patch14 \
    --finetune 'mae_pretrain_vit_base.pth' \
    --epochs 50 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.3 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path 'AGI/data_60k'
  ```



OR (if running on 1 node with 8 GPUs. Couldn't try it without cluster and multiple GPU): 

```
MP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune 'mae_pretrain_vit_base.pth' \
    --epochs 50 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path 'AGI/data_60k'
```

----

```
model_path = 'mae/output_dir/checkpoint-49.pth'
```

7) Evaluate the finetune mae_vit_base on eval_data

```
python main_finetune.py --eval --resume {model_path} --model vit_base_patch16 --batch_size TODO --data_path 'eval_data'
```

----

8) Repeat 6) and 7) for the different dataset sizes ('data_sampled_6k', 'data_sampled_36k', 'data_sampled_600').

----
IF NEEDED

9)a) In case the results are shitty, we might need to pretrain the model on MedMnist and then finetune on the eval_data.

```
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 8 \
    --use_volta32 \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path 'AGI/data_60k'
```


9)b)

```
python main_finetune.py \
    --accum_iter 1 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune 'mae_pretrain_vit_base.pth' \
    --epochs 50 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path 'eval_data'
```


--------



<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>


This is a PyTorch/GPU re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):
```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```

* The original implementation was in TensorFlow+TPU. This re-implementation is in PyTorch+GPU.

* This repo is a modification on the [DeiT repo](https://github.com/facebookresearch/deit). Installation and preparation follow that repo.

* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

### Catalog

- [x] Visualization demo
- [x] Pre-trained checkpoints + fine-tuning code
- [x] Pre-training code

### Visualization demo

Run our interactive visualization demo using [Colab notebook](https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb) (no GPU needed):
<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/147859292-77341c70-2ed8-4703-b153-f505dcb6f2f8.png" width="600">
</p>

### Fine-tuning with pre-trained checkpoints

The following table provides the pre-trained checkpoints used in the paper, converted from TF/TPU to PT/GPU:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Base</th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">ViT-Huge</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>8cad7c</tt></td>
<td align="center"><tt>b8b06e</tt></td>
<td align="center"><tt>9bdbb0</tt></td>
</tr>
</tbody></table>

The fine-tuning instruction is in [FINETUNE.md](FINETUNE.md).

By fine-tuning these pre-trained models, we rank #1 in these classification tasks (detailed in the paper):
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-B</th>
<th valign="bottom">ViT-L</th>
<th valign="bottom">ViT-H</th>
<th valign="bottom">ViT-H<sub>448</sub></th>
<td valign="bottom" style="color:#C0C0C0">prev best</td>
<!-- TABLE BODY -->
<tr><td align="left">ImageNet-1K (no external data)</td>
<td align="center">83.6</td>
<td align="center">85.9</td>
<td align="center">86.9</td>
<td align="center"><b>87.8</b></td>
<td align="center" style="color:#C0C0C0">87.1</td>
</tr>
<td colspan="5"><font size="1"><em>following are evaluation of the same model weights (fine-tuned in original ImageNet-1K):</em></font></td>
<tr>
</tr>
<tr><td align="left">ImageNet-Corruption (error rate) </td>
<td align="center">51.7</td>
<td align="center">41.8</td>
<td align="center"><b>33.8</b></td>
<td align="center">36.8</td>
<td align="center" style="color:#C0C0C0">42.5</td>
</tr>
<tr><td align="left">ImageNet-Adversarial</td>
<td align="center">35.9</td>
<td align="center">57.1</td>
<td align="center">68.2</td>
<td align="center"><b>76.7</b></td>
<td align="center" style="color:#C0C0C0">35.8</td>
</tr>
<tr><td align="left">ImageNet-Rendition</td>
<td align="center">48.3</td>
<td align="center">59.9</td>
<td align="center">64.4</td>
<td align="center"><b>66.5</b></td>
<td align="center" style="color:#C0C0C0">48.7</td>
</tr>
<tr><td align="left">ImageNet-Sketch</td>
<td align="center">34.5</td>
<td align="center">45.3</td>
<td align="center">49.6</td>
<td align="center"><b>50.9</b></td>
<td align="center" style="color:#C0C0C0">36.0</td>
</tr>
<td colspan="5"><font size="1"><em>following are transfer learning by fine-tuning the pre-trained MAE on the target dataset:</em></font></td>
</tr>
<tr><td align="left">iNaturalists 2017</td>
<td align="center">70.5</td>
<td align="center">75.7</td>
<td align="center">79.3</td>
<td align="center"><b>83.4</b></td>
<td align="center" style="color:#C0C0C0">75.4</td>
</tr>
<tr><td align="left">iNaturalists 2018</td>
<td align="center">75.4</td>
<td align="center">80.1</td>
<td align="center">83.0</td>
<td align="center"><b>86.8</b></td>
<td align="center" style="color:#C0C0C0">81.2</td>
</tr>
<tr><td align="left">iNaturalists 2019</td>
<td align="center">80.5</td>
<td align="center">83.4</td>
<td align="center">85.7</td>
<td align="center"><b>88.3</b></td>
<td align="center" style="color:#C0C0C0">84.1</td>
</tr>
<tr><td align="left">Places205</td>
<td align="center">63.9</td>
<td align="center">65.8</td>
<td align="center">65.9</td>
<td align="center"><b>66.8</b></td>
<td align="center" style="color:#C0C0C0">66.0</td>
</tr>
<tr><td align="left">Places365</td>
<td align="center">57.9</td>
<td align="center">59.4</td>
<td align="center">59.8</td>
<td align="center"><b>60.3</b></td>
<td align="center" style="color:#C0C0C0">58.0</td>
</tr>
</tbody></table>

### Pre-training

The pre-training instruction is in [PRETRAIN.md](PRETRAIN.md).

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
