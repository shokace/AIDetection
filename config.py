# config.py
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# DATA PATHS
TRAIN_DIR = fr"{PROJECT_ROOT}\dataset\train"
VAL_DIR   = fr"{PROJECT_ROOT}\dataset\val"
TEST_DIR  = fr"{PROJECT_ROOT}\dataset\test"

# MODEL SAVE / LOAD
MODEL_PATH = fr"{PROJECT_ROOT}\model\resnet50_fakeness.pt"

# INFERENCE (single image or batch folder)
INFER_IMAGE_PATH = fr"{PROJECT_ROOT}\test_images\petar2.png"
INFER_FOLDER_PATH = fr"{PROJECT_ROOT}\test_images"

# CLASSES AND DATA STRUCTURES
CLASS_NAMES = ["fake", "real"]