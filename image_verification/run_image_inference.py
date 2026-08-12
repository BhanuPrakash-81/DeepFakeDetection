import os
import sys
import argparse
import logging
import cv2

# Add parent directory to sys.path for importing project modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from image_verification.image_detector import ImageDeepfakeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RunImageInference")


def main():
    parser = argparse.ArgumentParser(description="Image Deepfake & AI-Generated Image Verification Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to local image file or image URL")
    parser.add_argument("--model", type=str, default=None, help="Path to trained attention adapter checkpoint")
    parser.add_argument("--out_dir", type=str, default="outputs/image_results", help="Directory to save annotated result image")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    detector = ImageDeepfakeDetector(model_path=args.model)
    logger.info(f"Processing image for deepfake verification: '{args.image}'")

    result = detector.analyze_image(args.image)

    # Save output annotated image
    if isinstance(args.image, str) and not (args.image.startswith("http://") or args.image.startswith("https://")):
        base_name = os.path.splitext(os.path.basename(args.image))[0]
    else:
        base_name = f"image_result_{int(os.path.basename(args.image).split('?')[0].split('.')[0]) if '.' in args.image else 'output'}"

    save_path = os.path.join(args.out_dir, f"{base_name}_verified.jpg")
    cv2.imwrite(save_path, result["annotated_image"])

    print("\n" + "="*70)
    print("        IMAGE DEEPFAKE & AI-GENERATED VERIFICATION REPORT        ")
    print("="*70)
    print(f" Input Image         : {args.image}")
    print(f" Verification Result : {result['label']}")
    print(f" Confidence Score    : {result['confidence']:.2f}%")
    print(f" Raw Probability     : {result['probability']:.4f}")
    print("-" * 70)
    print(" Dynamic Modality Attention Weights Allocation:")
    print(f"   Spatial Branch (EfficientNet-B0) : {result['weights']['spatial']:.4f}")
    print(f"   Frequency / Artifacts            : {result['weights']['temporal']:.4f}")
    print(f"   Biological / Skin Tone           : {result['weights']['biological']:.4f}")
    print("-" * 70)
    print(f" Annotated Result Saved To: {os.path.abspath(save_path)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
