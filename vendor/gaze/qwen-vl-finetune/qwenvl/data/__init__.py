import re

# Define placeholders for dataset paths

OCR_VQA ={
    "annotation_path": "../datasets/ocr_vqa_cambrian.jsonl",
    "data_path": "../datasets",
}

COCO ={
    "annotation_path": "../datasets/coco_cambrian.jsonl",
    "data_path": "../datasets",
}

CHARTQA ={
    "annotation_path": "../datasets/chartqa_cambrian.jsonl",
    "data_path": "../datasets",
}

VISCOT_MKZ_BBOX_4K ={
    "annotation_path": "/home/bbhan/gaze/viscot_mix_20k_30-15-15-10-5-5-5-5-5_seed42.jsonl",
    "data_path": "/data/DataSet/vlm_dataset/cot_image_data/",
}

VISCOT_MASK_20K_EXACT = {
    "annotation_path": "/home/ruimeng/hierarchical-vis-pruning/outputs/mask_gaze_20k_exact/manifest_20k_exact.jsonl",
    "mask_supervision_path": "/home/ruimeng/hierarchical-vis-pruning/outputs/mask_gaze_20k_exact/token_supervision_20k_exact.jsonl",
    "data_path": "/data/DataSet/vlm_dataset/cot_image_data/",
}

data_dict = {
    "ocr_vqa": OCR_VQA,
    "coco": COCO,
    'chartqa': CHARTQA,
    "viscot_mkz_bbox_4k": VISCOT_MKZ_BBOX_4K,
    "viscot_mask_20k_exact": VISCOT_MASK_20K_EXACT,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["ocr_vqa"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
