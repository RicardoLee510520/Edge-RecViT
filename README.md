# Edge-RecViT CVPR 2026 Poster

**Efficient Vision Transformer via Semantic-Refined Dynamic Recursion** 


## Abstract
Vision Transformers (ViTs) have achieved remarkable progress in visual and multimodal tasks, yet their deployment remains costly. Token-adaptive methods reduce FLOPs through dynamic depth computation, but they face two limitations: (1) Global attention overemphasizes highly similar foreground regions, causing token-adaptive modules to assign the deepest computation to semantically weak foreground tokens while prematurely exiting edge tokens rich in structural cues (as shown in Fig. 1); (2) Although token-adaption lowers FLOPs, it still relies on large parameter sets, and deep-layer weights remain underutilized due to early token exit. Parameter sharing could address redundancy but is difficult to apply in ViTs, where hierarchical abstraction typically requires diverse transformations. To address these issues, we propose Edge-RecViT,an Edge-Adaptive Dynamic Recursive Vision Transformer that integrates an edge-aware token-adaptive ranker with a recursive transformer using fully shared parameters in its hidden layers. Edge-RecViT dynamically allocates computation based on semantic richness: structurally informative edge tokens receive deeper refinement, whereas redundant low-information tokens exit early. Extensive experiments show that Edge-RecViT provides an excellent tradeoff among accuracy, FLOPs, and parameter efficiency. On imageNet-1K, it matches DeiT within 0.3% Top-1 accuracy
while reducing FLOPs by 30.5% (35.1 → 24.39 GFLOPs).At the Base level, parameter drops from 86M to 23.21M
with higher accuracy than ViT-Base; compared with ViTLarge, parameters are reduced by 93% while maintaining superior accuracy. 




**The code has been tested and should generally work as expected; minor fixes may be needed to get it running in your specific environment.**


## Repository Structure

```
optimum-main/
├── optimum/
│   └── mor_vit/                # Core Edge-RecViT (MoR-ViT) model implementation
│       ├── modeling_mor_vit.py       # Model, config, and recursive backbone
│       ├── token_choice_router.py    # Per-token dynamic recursion-depth router
│       └── load_pretrained.py        # Utilities for loading pretrained weights
├── examples/
│   ├── train_mor_vit_cifar10.py      # End-to-end training example (CIFAR-10)
│   ├── run_single_gpu_training.sh    # Single-GPU launch script
│   └── run_ddp_training.sh           # Multi-GPU / DDP launch script
├── docs/                        # Design notes, paper guide, implementation comparisons
├── tests/                       # Unit tests for the MoR-ViT model
├── best_model/                  # Example config for a trained checkpoint
└── install.sh / requirements.txt
```

## Training

```bash
cd examples
python train_mor_vit_cifar10.py \
    --data_dir /path/to/cifar-10-batches-py \
    --batch_size 256 \
    --epochs 100 \
    --lr 6e-5 \
    --use_pretrained \
    --enable_dora
```

Single-GPU and multi-GPU (DDP) launch scripts are provided in `examples/run_single_gpu_training.sh` and `examples/run_ddp_training.sh`.

## License

This project is released under the [Apache License 2.0](LICENSE).


## Citation

```bibtex
@inproceedings{li2026edge,
  title={Edge-RecViT: Efficient Vision Transformer via Semantic-Refined Dynamic Recursion},
  author={Li, YiZhou and Xu, Jinyi and Yin, Mingyu and Zhao, Xianyi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12987--12996},
  year={2026}
}
```
