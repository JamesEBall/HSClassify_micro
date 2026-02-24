---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:9829
- loss:MultipleNegativesRankingLoss
base_model: intfloat/multilingual-e5-small
widget:
- source_sentence: 'query: DAIRY PRODUCE; CHEESE (NOT GRATED, POWDERED OR PROCESSED),
    N.E.C. IN HEADING NO. 0406 POWDERED IN VACUUM PACKS 14290 PCS'
  sentences:
  - 'passage: Tôm đông lạnh, sơ chế, bỏ đầu bỏ vỏ, để xuất khẩu theo điều kiện thương
    mại tiêu chuẩn, điều kiện giao hàng FOB'
  - 'passage: Phô mai loại khác, để thông quan và khai báo nhập khẩu, kèm hóa đơn
    thương mại và phiếu đóng gói'
  - 'passage: Organic fresh tomatoes, hydroponic, for bulk procurement program, palletized
    for container shipment'
- source_sentence: 'query: Tôm thẻ chân trắng đông lạnh xuất khẩu'
  sentences:
  - 'passage: Red Delicious apples, fresh, for export'
  - 'passage: Cá nước ngọt đông lạnh, đóng thùng'
  - 'passage: กุ้งแช่แข็ง IQF ส่งออก สำหรับการขนส่งข้ามพรมแดน เงื่อนไขการขนส่ง CIF'
- source_sentence: 'query: 新鲜脐橙 加州进口，用于国际批发分销，托盘装集装箱运输'
  sentences:
  - 'passage: VEGETABLES; TOMATOES, FRESH OR CHILLED SIZE 72MM IN REEFER CONTAINER'
  - 'passage: CONVENTIONAL FRUIT, EDIBLE; ORANGES, FRESH OR DRIED IN BULK BAGS, for
    industrial procurement contract, shipping term FOB'
  - 'passage: Thịt bò đông lạnh không xương, Halal'
- source_sentence: 'query: MEAT; OF BOVINE ANIMALS, BONELESS CUTS, FRESH OR CHILLED
    IN CONTAINER, for cross-border shipment, shipping term FOB'
  sentences:
  - 'passage: Fresh plum tomatoes for Italian cooking, for bulk procurement program,
    palletized for container shipment'
  - 'passage: Boneless beef sirloin, fresh, not frozen, for bonded warehouse delivery,
    palletized for container shipment'
  - 'passage: ORGANIC VEGETABLES, ALLIACEOUS; ONIONS AND SHALLOTS, FRESH OR CHILLED
    WHITE ONION VARIETY IN CARTONS'
- source_sentence: 'query: CRUSTACEANS; FROZEN, SHRIMPS AND PRAWNS, EXCLUDING COLD-WATER
    VARIETIES, IN SHELL OR NOT, SMOKED, COOKED OR NOT BEFORE OR DURING SMOKING; IN
    SHELL, COOKED BY STEAMING OR BY BOILING IN WATER 21/25 COUNT IN SACKS 8576.9 KG'
  sentences:
  - 'passage: กุ้งแช่แข็ง IQF ส่งออก สำหรับการขนส่งข้ามพรมแดน เงื่อนไขการขนส่ง CIF'
  - 'passage: DAIRY PRODUCE; MILK AND CREAM, CONCENTRATED OR CONTAINING ADDED SUGAR
    OR OTHER SWEETENING MATTER, IN POWDER, GRANULES OR OTHER SOLID FORMS, OF A FAT
    CONTENT NOT EXCEEDING 1.5% (BY WEIGHT) FAT CONTENT 3.5% IN VACUUM PACKS'
  - 'passage: CRUSTACEANS; FROZEN, SHRIMPS AND PRAWNS, EXCLUDING COLD-WATER VARIETIES,
    IN SHELL OR NOT, SMOKED, COOKED OR NOT BEFORE OR DURING SMOKING; IN SHELL, COOKED
    BY STEAMING OR BY BOILING IN WATER'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on intfloat/multilingual-e5-small

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) <!-- at revision c007d7ef6fd86656326059b28395a7a03a7c5846 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'query: CRUSTACEANS; FROZEN, SHRIMPS AND PRAWNS, EXCLUDING COLD-WATER VARIETIES, IN SHELL OR NOT, SMOKED, COOKED OR NOT BEFORE OR DURING SMOKING; IN SHELL, COOKED BY STEAMING OR BY BOILING IN WATER 21/25 COUNT IN SACKS 8576.9 KG',
    'passage: CRUSTACEANS; FROZEN, SHRIMPS AND PRAWNS, EXCLUDING COLD-WATER VARIETIES, IN SHELL OR NOT, SMOKED, COOKED OR NOT BEFORE OR DURING SMOKING; IN SHELL, COOKED BY STEAMING OR BY BOILING IN WATER',
    'passage: กุ้งแช่แข็ง IQF ส่งออก สำหรับการขนส่งข้ามพรมแดน เงื่อนไขการขนส่ง CIF',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9576, 0.7030],
#         [0.9576, 1.0000, 0.6773],
#         [0.7030, 0.6773, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 9,829 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                            | positive                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             |
  | details | <ul><li>min: 9 tokens</li><li>mean: 36.3 tokens</li><li>max: 114 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 34.27 tokens</li><li>max: 113 tokens</li></ul> |
* Samples:
  | anchor                                                               | positive                                                                                                                                                                 |
  |:---------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>query: Chilled beef tenderloin, boneless, vacuum packed</code> | <code>passage: Thịt bò không xương tươi cho nhà hàng, cho hợp đồng mua sắm công nghiệp, hàng lô hỗn hợp</code>                                                           |
  | <code>query: 优质鲜牛肉 无骨 出口级别</code>                                    | <code>passage: 优质鲜牛肉 无骨 出口级别，用于国际批发分销，装20尺集装箱</code>                                                                                                                     |
  | <code>query: 冷却去骨黄牛肉 真空包装</code>                                     | <code>passage: FROZEN MEAT; OF BOVINE ANIMALS, BONELESS CUTS, FRESH OR CHILLED SKIN-ON IN TINS 15204.2 KG, for industrial procurement contract, shipping term CIF</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 2
- `learning_rate`: 2e-05
- `warmup_steps`: 0.1
- `gradient_accumulation_steps`: 16
- `warmup_ratio`: 0.1

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 2
- `max_steps`: -1
- `learning_rate`: 2e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.1
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 16
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: no
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: 0.1
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0651 | 10   | 0.9040        |
| 0.1302 | 20   | 0.7323        |
| 0.1953 | 30   | 0.4439        |
| 0.2604 | 40   | 0.2618        |
| 0.3255 | 50   | 0.2630        |
| 0.3906 | 60   | 0.2398        |
| 0.4557 | 70   | 0.1878        |
| 0.5207 | 80   | 0.2271        |
| 0.5858 | 90   | 0.2237        |
| 0.6509 | 100  | 0.2180        |
| 0.7160 | 110  | 0.2125        |
| 0.7811 | 120  | 0.2067        |
| 0.8462 | 130  | 0.1925        |
| 0.9113 | 140  | 0.1952        |
| 0.9764 | 150  | 0.1932        |
| 1.0391 | 160  | 0.1368        |
| 1.1041 | 170  | 0.1737        |
| 1.1692 | 180  | 0.1815        |
| 1.2343 | 190  | 0.1724        |
| 1.2994 | 200  | 0.1525        |
| 1.3645 | 210  | 0.1699        |
| 1.4296 | 220  | 0.1592        |
| 1.4947 | 230  | 0.1661        |
| 1.5598 | 240  | 0.1606        |
| 1.6249 | 250  | 0.1218        |
| 1.6900 | 260  | 0.1586        |
| 1.7551 | 270  | 0.1517        |
| 1.8202 | 280  | 0.1458        |
| 1.8853 | 290  | 0.1550        |
| 1.9504 | 300  | 0.1352        |


### Framework Versions
- Python: 3.14.3
- Sentence Transformers: 5.2.3
- Transformers: 5.2.0
- PyTorch: 2.10.0
- Accelerate: 1.12.0
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->