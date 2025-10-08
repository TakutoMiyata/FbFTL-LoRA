#!/usr/bin/env python3
"""
Generate BiT Non-IID experiment configuration files with alpha=0.5
Non-IID environment: data_split=non_iid, alpha=0.5 (moderate heterogeneity)
"""

import yaml
from pathlib import Path
from itertools import product

# Base configuration template
BASE_CONFIG = {
    'seed': 42,
    'use_gpu': True,
    'use_amp': False,
    'data': {
        'dataset_name': 'cifar100',
        'data_dir': './data',
        'num_clients': 10,
        'num_test_clients': 10,
        'batch_size': 64,
        'num_workers': 4,
        'pin_memory': True,
        'input_size': 224,
        'verbose': False,
        'imagenet_style': True,
        'data_split': 'non_iid',  # Non-IID固定
        'alpha': 0.5,  # 中程度のNon-IID
        'seed': 42,
        'augmentations': {
            'horizontal_flip': {'enabled': True, 'prob': 0.5},
            'random_rotation': {'enabled': True, 'degrees': 20},
            'random_resized_crop': {'enabled': True, 'scale_min': 0.4},
            'color_jitter': {'enabled': True, 'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3, 'hue': 0.1},
            'random_erasing': {'enabled': True, 'prob': 0.3},
            'mixup': {'enabled': False, 'alpha': 0.2, 'prob': 0.5},
            'cutmix': {'enabled': False, 'alpha': 1.0, 'prob': 0.5}
        }
    },
    'training': {
        'epochs': 3,
        'lr': 0.0001,
        'momentum': 0.9,
        'weight_decay': 0.001,
        'scheduler': 'cosine',
        'warmup_epochs': 0,
        'label_smoothing': 0.1,
        'gradient_clip': 1.0,
        'optimizer': 'sgd'
    },
    'federated': {
        'num_rounds': 100,
        'num_clients': 10,
        'client_fraction': 1.0,
        'aggregation_method': 'fedavg',
        'checkpoint_freq': 25,
        'exclude_bn_from_agg': True
    },
    'privacy': {
        'enable_privacy': False,
        'epsilon': 8.0,
        'delta': 1e-05,
        'max_grad_norm': 0.5,
        'noise_multiplier': 1.0,
        'target': 'lora_A',
        'use_opacus_accounting': True
    },
    'evaluation': {
        'eval_freq': 5,
        'metric': 'accuracy',
        'save_best_model': True
    },
    'experiment': {
        'name': 'BiT_NonIID_Experiment',
        'output_dir': 'experiments/bit_non_iid',
        'save_history': True,
        'save_model': True,
        'log_interval': 10,
        'use_wandb': False,
        'wandb_project': 'bit-non-iid',
        'wandb_entity': None
    },
    'reproducibility': {
        'deterministic': False
    },
    'advanced': {
        'personalized': False,
        'personalization_layers': [],
        'use_fbftl': False,
        'feature_extraction_rounds': 10,
        'server_lr': 1.0,
        'server_momentum': 0.9
    },
    'communication': {
        'compress': False,
        'compression_ratio': 0.1,
        'quantization_bits': None
    }
}

def generate_experiment_configs():
    """Generate all Non-IID (alpha=0.5) experiment configurations"""

    # Experiment parameters
    models = ['bit_s_r50x1', 'bit_s_r101x1']
    client_fractions = [1.0, 0.3]
    lora_ranks = [4, 8]
    epsilon_values = [4.0, 8.0]
    alpha = 0.5  # 中程度のNon-IID

    base_dir = Path('configs/experiment_configs_bit_non_iid')

    experiments = []

    # 1. FedAvg experiments (no LoRA)
    for model, fraction in product(models, client_fractions):
        config = BASE_CONFIG.copy()
        config['data'] = BASE_CONFIG['data'].copy()
        config['data']['alpha'] = alpha
        config['model'] = {
            'model_name': model,
            'num_classes': 100,
            'pretrained': True,
            'freeze_backbone': True,
            'lora': {'enabled': False}
        }
        config['federated'] = BASE_CONFIG['federated'].copy()
        config['federated']['client_fraction'] = fraction
        config['federated']['aggregation_method'] = 'fedavg'
        config['advanced'] = BASE_CONFIG['advanced'].copy()
        config['advanced']['personalized'] = False

        alpha_str = f"alpha{alpha}".replace('.', '')
        fraction_str = f"f{fraction:.1f}".replace('.', '')
        filename = f"{model}_{alpha_str}_{fraction_str}_fedavg.yaml"
        output_dir = base_dir / 'fedavg'

        config['experiment'] = BASE_CONFIG['experiment'].copy()
        config['experiment']['name'] = f"BiT_NonIID_{model}_alpha{alpha}_f{fraction:.1f}_FedAvg"
        config['experiment']['output_dir'] = f'experiments/bit_non_iid/fedavg/{model}_alpha{alpha}_f{fraction:.1f}'

        experiments.append({
            'config': config,
            'path': output_dir / filename,
            'type': 'FedAvg',
            'model': model,
            'fraction': fraction,
            'alpha': alpha
        })

    # 2. FedSA-LoRA experiments (no DP)
    for model, fraction, rank in product(models, client_fractions, lora_ranks):
        config = BASE_CONFIG.copy()
        config['data'] = BASE_CONFIG['data'].copy()
        config['data']['alpha'] = alpha
        config['model'] = {
            'model_name': model,
            'num_classes': 100,
            'pretrained': True,
            'freeze_backbone': True,
            'lora': {
                'enabled': True,
                'r': rank,
                'alpha': rank * 2,
                'dropout': 0.1
            }
        }
        config['federated'] = BASE_CONFIG['federated'].copy()
        config['federated']['client_fraction'] = fraction
        config['federated']['aggregation_method'] = 'fedsa'
        config['advanced'] = BASE_CONFIG['advanced'].copy()
        config['advanced']['personalized'] = True
        config['advanced']['personalization_layers'] = ['lora_B', 'head']

        alpha_str = f"alpha{alpha}".replace('.', '')
        fraction_str = f"f{fraction:.1f}".replace('.', '')
        filename = f"{model}_{alpha_str}_{fraction_str}_r{rank}_fedsa_lora.yaml"
        output_dir = base_dir / 'fedsa_lora'

        config['experiment'] = BASE_CONFIG['experiment'].copy()
        config['experiment']['name'] = f"BiT_NonIID_{model}_alpha{alpha}_f{fraction:.1f}_r{rank}_FedSA_LoRA"
        config['experiment']['output_dir'] = f'experiments/bit_non_iid/fedsa_lora/{model}_alpha{alpha}_f{fraction:.1f}_r{rank}'

        experiments.append({
            'config': config,
            'path': output_dir / filename,
            'type': 'FedSA-LoRA',
            'model': model,
            'fraction': fraction,
            'rank': rank,
            'alpha': alpha
        })

    # 3. FedSA-LoRA with DP experiments
    for model, fraction, rank, epsilon in product(models, client_fractions, lora_ranks, epsilon_values):
        config = BASE_CONFIG.copy()
        config['data'] = BASE_CONFIG['data'].copy()
        config['data']['alpha'] = alpha
        config['model'] = {
            'model_name': model,
            'num_classes': 100,
            'pretrained': True,
            'freeze_backbone': True,
            'lora': {
                'enabled': True,
                'r': rank,
                'alpha': rank * 2,
                'dropout': 0.2  # Higher dropout for DP
            }
        }
        config['federated'] = BASE_CONFIG['federated'].copy()
        config['federated']['client_fraction'] = fraction
        config['federated']['aggregation_method'] = 'fedsa_shareA_dp'
        config['privacy'] = BASE_CONFIG['privacy'].copy()
        config['privacy']['enable_privacy'] = True
        config['privacy']['epsilon'] = epsilon
        config['advanced'] = BASE_CONFIG['advanced'].copy()
        config['advanced']['personalized'] = True
        config['advanced']['personalization_layers'] = ['lora_B', 'head']

        alpha_str = f"alpha{alpha}".replace('.', '')
        fraction_str = f"f{fraction:.1f}".replace('.', '')
        filename = f"{model}_{alpha_str}_{fraction_str}_r{rank}_eps{int(epsilon)}_fedsa_lora_dp.yaml"
        output_dir = base_dir / 'fedsa_lora_dp'

        config['experiment'] = BASE_CONFIG['experiment'].copy()
        config['experiment']['name'] = f"BiT_NonIID_{model}_alpha{alpha}_f{fraction:.1f}_r{rank}_eps{int(epsilon)}_FedSA_LoRA_DP"
        config['experiment']['output_dir'] = f'experiments/bit_non_iid/fedsa_lora_dp/{model}_alpha{alpha}_f{fraction:.1f}_r{rank}_eps{int(epsilon)}'

        experiments.append({
            'config': config,
            'path': output_dir / filename,
            'type': 'FedSA-LoRA-DP',
            'model': model,
            'fraction': fraction,
            'rank': rank,
            'epsilon': epsilon,
            'alpha': alpha
        })

    return experiments

def save_experiments(experiments):
    """Save all experiment configurations to files"""

    print("=" * 80)
    print("Generating BiT Non-IID (alpha=0.5) Experiment Configurations")
    print("=" * 80)
    print(f"Non-IID parameter: α = 0.5 (moderate heterogeneity)")
    print("=" * 80)

    for exp in experiments:
        exp['path'].parent.mkdir(parents=True, exist_ok=True)

        with open(exp['path'], 'w') as f:
            yaml.dump(exp['config'], f, default_flow_style=False, sort_keys=False)

        print(f"✅ {exp['path']}")

    print("=" * 80)
    print(f"Total configurations generated: {len(experiments)}")

    # Summary by type
    by_type = {}
    for exp in experiments:
        exp_type = exp['type']
        by_type[exp_type] = by_type.get(exp_type, 0) + 1

    print("\nSummary:")
    for exp_type, count in by_type.items():
        print(f"  {exp_type}: {count} configs")

    print("=" * 80)

if __name__ == "__main__":
    experiments = generate_experiment_configs()
    save_experiments(experiments)
    print("\n🎉 Non-IID (alpha=0.5) experiment configurations generated successfully!")
