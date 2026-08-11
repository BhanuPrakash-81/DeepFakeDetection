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
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--source", type=str, default="0", help="Webcam ID (e.g. 0) or path to video file for live inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML configuration")
    args = parser.parse_args()

    # Load configuration cleanly
    config = load_config(args.config)
    output_dir = get_base_output_dir()
    logger.info(f"Initialized Pipeline. Resolved Root Output Directory: {output_dir}")

    if args.stage == 1:
        logger.info("--- STAGE 1: FEATURE EXTRACTION ---")
        from extract_features import extract_dataset
        extract_dataset(data_dir=args.data_dir)

    elif args.stage == 2:
        logger.info("--- STAGE 2: MODEL TRAINING ---")
        from model_and_train import train_adapter
        train_adapter(epochs=args.epochs, batch_size=args.batch_size)

    elif args.stage == 3:
        logger.info("--- STAGE 3: LIVE VIDEO INFERENCE ---")
        from live_inference import run_live_inference
        src = int(args.source) if args.source.isdigit() else args.source
        run_live_inference(source=src)

    elif args.stage == 4:
        logger.info("--- STAGE 4: ACADEMIC METRICS EVALUATION ---")
        from evaluate_metrics import evaluate_model
        evaluate_model()

if __name__ == "__main__":
    main()
