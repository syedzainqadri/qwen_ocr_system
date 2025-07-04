#!/usr/bin/env python3
"""
PaddleOCR Training System
Train PaddleOCR with custom samples for better accuracy.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaddleOCRTrainer:
    def __init__(self, training_dir: str = "training_data"):
        self.training_dir = Path(training_dir)
        self.images_dir = self.training_dir / "images"
        self.labels_dir = self.training_dir / "labels"
        self.config_dir = self.training_dir / "configs"
        
        # Create directories
        self.training_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        
        logger.info(f"PaddleOCR Trainer initialized at: {self.training_dir}")
    
    def add_training_sample(self, image_path: str, ground_truth_text: str, sample_name: str = None):
        """Add a training sample with image and ground truth text."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return False
        
        # Generate sample name if not provided
        if sample_name is None:
            sample_name = f"sample_{len(list(self.images_dir.glob('*'))) + 1:04d}"
        
        # Copy image to training directory
        image_ext = image_path.suffix
        new_image_path = self.images_dir / f"{sample_name}{image_ext}"
        shutil.copy2(image_path, new_image_path)
        
        # Create label file
        label_path = self.labels_dir / f"{sample_name}.txt"
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(ground_truth_text)
        
        logger.info(f"Added training sample: {sample_name}")
        logger.info(f"  Image: {new_image_path}")
        logger.info(f"  Text: {ground_truth_text[:50]}...")
        
        return True
    
    def create_training_samples(self):
        """Create training samples for common OCR scenarios."""
        samples = [
            {
                "name": "english_document",
                "text": "This is a sample English document with clear text for OCR training.",
                "description": "Clean English text sample"
            },
            {
                "name": "urdu_document", 
                "text": "یہ اردو زبان کا نمونہ ہے جو OCR کی تربیت کے لیے استعمال ہوگا۔",
                "description": "Urdu text sample"
            },
            {
                "name": "mixed_content",
                "text": "Invoice #12345\nDate: 2024-01-15\nAmount: $1,234.56\nThank you for your business!",
                "description": "Mixed content with numbers and symbols"
            },
            {
                "name": "handwritten_style",
                "text": "Handwritten notes can be challenging for OCR systems to process accurately.",
                "description": "Handwritten-style text"
            }
        ]
        
        logger.info("Creating synthetic training samples...")
        
        for sample in samples:
            # Create a simple text image (you would replace this with actual images)
            sample_file = self.training_dir / f"{sample['name']}_sample.txt"
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(f"Sample: {sample['name']}\n")
                f.write(f"Description: {sample['description']}\n")
                f.write(f"Ground Truth: {sample['text']}\n")
            
            logger.info(f"Created sample template: {sample['name']}")
    
    def generate_training_config(self):
        """Generate PaddleOCR training configuration."""
        config = {
            "Global": {
                "use_gpu": False,  # Set to True if you have GPU
                "epoch_num": 100,
                "log_smooth_window": 20,
                "print_batch_step": 10,
                "save_model_dir": "./output/",
                "save_epoch_step": 10,
                "eval_batch_step": [0, 2000],
                "cal_metric_during_train": True,
                "pretrained_model": None,
                "checkpoints": None,
                "save_inference_dir": None,
                "use_visualdl": False,
                "infer_img": None,
                "character_dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
                "max_text_length": 100,
                "infer_mode": False,
                "use_space_char": True,
                "distributed": False
            },
            "Optimizer": {
                "name": "Adam",
                "beta1": 0.9,
                "beta2": 0.999,
                "lr": {
                    "name": "Cosine",
                    "learning_rate": 0.001,
                    "warmup_epoch": 5
                },
                "regularizer": {
                    "name": "L2",
                    "factor": 1e-4
                }
            },
            "Architecture": {
                "model_type": "rec",
                "algorithm": "SVTR_LCNet",
                "Transform": None,
                "Backbone": {
                    "name": "SVTRNet",
                    "img_size": [32, 128],
                    "out_char_num": 25,
                    "out_channels": 192,
                    "patch_merging": "Conv",
                    "arch": "base",
                    "last_stage": True
                },
                "Head": {
                    "name": "CTCHead",
                    "fc_decay": 0.00001
                }
            },
            "Loss": {
                "name": "CTCLoss"
            },
            "PostProcess": {
                "name": "CTCLabelDecode"
            },
            "Metric": {
                "name": "RecMetric",
                "main_indicator": "acc"
            },
            "Train": {
                "dataset": {
                    "name": "SimpleDataSet",
                    "data_dir": str(self.images_dir),
                    "label_file_list": [str(self.training_dir / "train_list.txt")],
                    "transforms": [
                        {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                        {"RecAug": {}},
                        {"CTCLabelEncode": {}},
                        {"RecResizeImg": {"image_shape": [3, 32, 128]}},
                        {"KeepKeys": {"keep_keys": ["image", "label", "length"]}}
                    ]
                },
                "loader": {
                    "shuffle": True,
                    "batch_size_per_card": 256,
                    "drop_last": True,
                    "num_workers": 4
                }
            },
            "Eval": {
                "dataset": {
                    "name": "SimpleDataSet", 
                    "data_dir": str(self.images_dir),
                    "label_file_list": [str(self.training_dir / "val_list.txt")],
                    "transforms": [
                        {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                        {"CTCLabelEncode": {}},
                        {"RecResizeImg": {"image_shape": [3, 32, 128]}},
                        {"KeepKeys": {"keep_keys": ["image", "label", "length"]}}
                    ]
                },
                "loader": {
                    "shuffle": False,
                    "drop_last": False,
                    "batch_size_per_card": 256,
                    "num_workers": 4
                }
            }
        }
        
        config_path = self.config_dir / "rec_config.yml"
        
        # Convert to YAML format (simplified)
        yaml_content = self._dict_to_yaml(config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        logger.info(f"Training configuration saved to: {config_path}")
        return config_path
    
    def _dict_to_yaml(self, data, indent=0):
        """Convert dictionary to YAML format (simplified)."""
        yaml_str = ""
        for key, value in data.items():
            yaml_str += "  " * indent + f"{key}:\n"
            if isinstance(value, dict):
                yaml_str += self._dict_to_yaml(value, indent + 1)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yaml_str += "  " * (indent + 1) + "-\n"
                        yaml_str += self._dict_to_yaml(item, indent + 2)
                    else:
                        yaml_str += "  " * (indent + 1) + f"- {item}\n"
            else:
                yaml_str += "  " * (indent + 1) + f"{value}\n"
        return yaml_str
    
    def create_data_lists(self):
        """Create training and validation data lists."""
        image_files = list(self.images_dir.glob("*"))
        
        if not image_files:
            logger.warning("No training images found!")
            return
        
        # Split 80% train, 20% validation
        split_idx = int(len(image_files) * 0.8)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Create train list
        train_list_path = self.training_dir / "train_list.txt"
        with open(train_list_path, 'w', encoding='utf-8') as f:
            for img_file in train_files:
                label_file = self.labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    with open(label_file, 'r', encoding='utf-8') as lf:
                        label_text = lf.read().strip()
                    f.write(f"{img_file.name}\t{label_text}\n")
        
        # Create validation list
        val_list_path = self.training_dir / "val_list.txt"
        with open(val_list_path, 'w', encoding='utf-8') as f:
            for img_file in val_files:
                label_file = self.labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    with open(label_file, 'r', encoding='utf-8') as lf:
                        label_text = lf.read().strip()
                    f.write(f"{img_file.name}\t{label_text}\n")
        
        logger.info(f"Created data lists:")
        logger.info(f"  Training samples: {len(train_files)}")
        logger.info(f"  Validation samples: {len(val_files)}")
    
    def get_training_instructions(self):
        """Get instructions for training PaddleOCR."""
        instructions = f"""
🎯 PaddleOCR Training Instructions
================================

1. 📁 Training Data Structure:
   {self.training_dir}/
   ├── images/          # Training images
   ├── labels/          # Ground truth text files
   ├── configs/         # Training configuration
   ├── train_list.txt   # Training data list
   └── val_list.txt     # Validation data list

2. 📝 Add Training Samples:
   trainer.add_training_sample("path/to/image.jpg", "Ground truth text")

3. 🚀 Start Training:
   python -m paddle.distributed.launch \\
       --gpus '0' \\
       tools/train.py \\
       -c {self.config_dir}/rec_config.yml

4. 📊 Monitor Training:
   - Check logs for accuracy improvements
   - Training will save models in ./output/
   - Use best model for inference

5. 🔄 Use Trained Model:
   - Replace default PaddleOCR model
   - Update model path in paddle_ocr.py
   - Test with your specific data

📋 Next Steps:
1. Add your training images and labels
2. Run trainer.create_data_lists()
3. Run trainer.generate_training_config()
4. Start training with PaddleOCR tools
"""
        return instructions

def main():
    """Main training setup function."""
    print("🎯 PaddleOCR Training System")
    print("=" * 50)
    
    trainer = PaddleOCRTrainer()
    
    # Create sample training data structure
    trainer.create_training_samples()
    trainer.generate_training_config()
    
    print("\n" + trainer.get_training_instructions())
    
    print("\n🎉 Training system ready!")
    print("📝 Add your training images and labels, then follow the instructions above.")

if __name__ == "__main__":
    main()
