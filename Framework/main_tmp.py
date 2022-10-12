import argparse

from huawei_slavic_ner_models.src.utils import config, logging
from huawei_slavic_ner_models.src.evaluate import evaluate_models
from huawei_slavic_ner_models.src.train_xlm_model import train_wechsel_on_s1, train_wechsel_on_s2, \
    train_wechsel_on_s3, train_wechsel_on_slavicner


def main():
    parser = argparse.ArgumentParser(description='Training and evaluation params')
    parser.add_argument('-train', action='store_true', default=False, help="training mode")
    parser.add_argument(
        '-dataset', dest="train_dataset_type", default="s2", type=str,
        help="train dataset type: s1/s2/s3/slavicner, default is s2"
    )
    parser.add_argument('-eval', action='store_true', default=False, help="evaluation mode")
    parser.add_argument(
        '-eval_model_path', dest="eval_model_path", default="", type=str, help="model path for evaluation"
    )
    parser.add_argument('-eval_dataset_filename', dest="eval_dataset_filename", default="", type=str,
                        help="Path of dataset for evaluation. If no data provided, WikiNER is used as default.")
    args = parser.parse_args()

    if args.train:
        if args.train_dataset_type.lower() == 's1':
            logging.info("Training model on dataset S1")
            train_wechsel_on_s1(config.ner_task.language_order)
        elif args.train_dataset_type.lower() == 's2':
            logging.info("Training model on dataset S2")
            train_wechsel_on_s2(config.ner_task.language_order)
        elif args.train_dataset_type.lower() == 's3':
            logging.info("Training model on dataset S3")
            train_wechsel_on_s3(config.ner_task.language_order)
        elif args.train_dataset_type.lower() == 'slavicner':
            logging.info("Training model on dataset SlavicNER")
            train_wechsel_on_slavicner(config.ner_task.language_order)
        else:
            logging.info("Unknown dataset type")
            raise Exception("Unknown dataset type in parameter '-dataset'")
    elif args.eval:
        if args.eval_model_path:
            evaluate_models(args.eval_model_path, args.eval_dataset_filename)
        else:
            logging.info("Evaluation model path is incorrect")
            raise Exception("Evaluation model path is incorrect. Use param '-eval_model_path'")
    else:
        logging.info("Unknown mode")
        raise Exception("Unknown mode. Use -train or -eval")


if __name__ == '__main__':
    main()

