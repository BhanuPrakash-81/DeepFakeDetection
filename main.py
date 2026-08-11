import os
import sys
import argparse
import logging
from typing import Optional

from utils.paths import load_config, get_base_output_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MainEntrypoint")

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Deepfake Detection Pipeline Entrypoint (Dynamic Environment-Agnostic MLOps)"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
        help="Pipeline Stage: 1=Feature Extraction, 2=Model Training, 3=Live Inference, 4=Academic Metrics"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Input dataset directory")
    parser.add_argument("--features", type=str, default=None, help="Path to .npz dataset")
    parser.add_argument("--save_path", type=str, default=None, help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout probability")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 weight decay")
    parser.add_argument("--source", type=str, default="0", help="Webcam ID (e.g. 0) or path to video file for live inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML configuration")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = get_base_output_dir()
    logger.info(f"Initialized Pipeline. Resolved Root Output Directory: {output_dir}")

    if args.stage == 1:
        logger.info("--- STAGE 1: FEATURE EXTRACTION ---")
        from extract_features import extract_dataset
        extract_dataset(data_dir=args.data_dir, output_path=args.features)

    elif args.stage == 2:
        logger.info("--- STAGE 2: MODEL TRAINING ---")
        from model_and_train import train_adapter
        train_adapter(
            npz_path=args.features,
            save_path=args.save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            dropout=args.dropout,
            weight_decay=args.weight_decay
        )

    elif args.stage == 3:
        logger.info("--- STAGE 3: LIVE VIDEO INFERENCE ---")
        from live_inference import run_live_inference
        src = int(args.source) if args.source.isdigit() else args.source
        run_live_inference(source=src, model_path=args.save_path)

    elif args.stage == 4:
        logger.info("--- STAGE 4: ACADEMIC METRICS EVALUATION ---")
        from evaluate_metrics import evaluate_model
        evaluate_model(npz_path=args.features, model_path=args.save_path)

if __name__ == "__main__":
    main()
