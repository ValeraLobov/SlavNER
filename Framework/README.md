---

The system has been tested on Python 3.6.9 and Python 3.7.

Before launching in the directory /huawei_slavic_ner_models, you need to run `pip install ./huawei_slavic_ner_models`, then run
`main.py ` in the same directory.

The hyperparameters of the models, the backbone type of the model ('xlm' or 'rubert'), etc. are set in `config.yaml`. The startup parameters are described in `main.py `. Here are the startup examples:

- `python main.py -train -dataset s2` (train three WECHSEL models C1, C2, C3 on dataset S2)

- `python main.py -eval -eval_model_path /release_archive_bio_format/wechsel_ccl_perplexity_7_lang_mixed -eval_dataset_filename ./huawei_slavic_ner_models/data/custom_data/example_custom_dataset.csv`
(calculate validation metrics for the XLM-R model at `/release_archive_bio_format/ wechsel_ccl_perplexity_7_lang_mixed` on the user dataset at `./huawei_slavic_ner_models/data/custom_data/example_custom_dataset.csv`)

The user dataset looks like a csv file with two columns, 'sentences' and 'tags', which contain lists of tokens/tags in BIO format. An example is in the custom_data folder. If the directory of the custom dataset is not specified, WikiNER is used for validation.


In `src/utils.py ` you can specify the GPU number.

All training/validation logs are stored in the `huawei_slavic_ner_models_log'.