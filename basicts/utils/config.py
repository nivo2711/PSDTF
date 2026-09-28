from typing import Dict


def get_dataset_name(cfg: Dict) -> str:

    dataset_cfg = cfg.get('DATASET')
    if dataset_cfg:
        assert isinstance(dataset_cfg, dict), "'DATASET' must be a dictionary."
        return dataset_cfg.get('NAME', 'Unknown Dataset')

    dataset_names = [
        cfg.get(section, {}).get('DATA', {}).get('DATASET', {}).get('NAME')
        for section in ['TRAIN', 'VAL', 'TEST']
    ]

    dataset_names = [name for name in dataset_names if name]
    dataset_names = list(set(dataset_names))

    return '_'.join(dataset_names) if dataset_names else 'Unknown Dataset'
