"""
config/settings.py
==================
Single place for every path, number and rule used by the project.

Rule: this file has NO machine-learning imports (no TensorFlow, no OpenCV),
so it loads instantly and can be tested anywhere. Other modules import
their numbers from here instead of hardcoding them.

Colors/fonts are NOT here. They live in gui/theme.py (added in Stage 7).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Application info
# ----------------------------------------------------------------------
APP_NAME = "FaceID Workstation"
APP_SUBTITLE = "Face Recognition System Using Artificial Neural Networks"
APP_VERSION = "1.0.0"

WINDOW_DEFAULT_SIZE = (1366, 768)   # target resolution 1
WINDOW_MIN_SIZE = (1200, 680)       # never smaller than this (no clipped content)

# ----------------------------------------------------------------------
# 2. Folder and file paths (all relative to the project root)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"       # dataset/person_001/, person_002/ ...
MODELS_DIR = BASE_DIR / "models"         # one sub-folder per trained model
RESULTS_DIR = BASE_DIR / "results"       # graphs, confusion matrix, reports
ASSETS_DIR = BASE_DIR / "assets"
ICONS_DIR = ASSETS_DIR / "icons"
IMAGES_DIR = ASSETS_DIR / "images"
DB_PATH = BASE_DIR / "database" / "face_system.db"

PERSON_FOLDER_PREFIX = "person_"         # folder name = person_<db id, 3 digits>

# ----------------------------------------------------------------------
# 3. Model bundle files
# ----------------------------------------------------------------------
# Every training run creates ONE folder that holds everything belonging to
# that model, so the model and its label map can never get out of sync:
#
#   models/model_20260101_120000/
#       model.keras       <- the trained ANN
#       label_map.json    <- class index -> user id / name
#       meta.json         <- config, image size, checksum, metrics
#   models/active_model.json   <- points to the bundle currently in use
#
# The pointer file is written LAST, only after the whole bundle is saved.
MODEL_BUNDLE_PREFIX = "model_"
MODEL_FILE_NAME = "model.keras"
LABEL_MAP_FILE_NAME = "label_map.json"
MODEL_META_FILE_NAME = "meta.json"
ACTIVE_MODEL_POINTER = MODELS_DIR / "active_model.json"

# ----------------------------------------------------------------------
# 4. Face detection (OpenCV Haar Cascade)
# ----------------------------------------------------------------------
HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"  # found via cv2.data
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5       # higher = fewer false detections
HAAR_MIN_FACE_SIZE = (60, 60)

# ----------------------------------------------------------------------
# 5. Image preprocessing (SAME values for training and prediction)
# ----------------------------------------------------------------------
IMAGE_SIZE = 64                          # ANN input is 64 x 64 grayscale
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE      # 4096 numbers after flattening
STORED_FACE_SIZE = 128                   # cropped faces are saved at 128x128;
                                         # downscaled to IMAGE_SIZE only when
                                         # the ANN needs them
ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MAX_IMAGE_FILE_MB = 15

# ----------------------------------------------------------------------
# 6. Dataset rules
# ----------------------------------------------------------------------
MIN_IMAGES_PER_USER = 10         # registration is refused below this
RECOMMENDED_IMAGES_PER_USER = 30
MIN_USERS_FOR_TRAINING = 2       # a classifier needs at least 2 classes

# Stratified split. Every person must appear in train, validation AND test.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42                 # same split every run (repeatable results)
MIN_SAMPLES_PER_SPLIT = 1        # each person needs >= 1 image in each split

# ----------------------------------------------------------------------
# 7. Registration input rules
# ----------------------------------------------------------------------
NAME_MAX_LENGTH = 60
REGISTRATION_ID_MAX_LENGTH = 30
# letters, digits, dash, underscore, dot; must start with a letter or digit
REGISTRATION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$"

# ----------------------------------------------------------------------
# 8. Recognition
# ----------------------------------------------------------------------
# If the highest softmax probability is below this value the face is
# reported as "Unknown". This is a basic rejection rule, NOT full
# open-set recognition. The user can change it at runtime (saved in the
# app_settings table).
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
CONFIDENCE_THRESHOLD_RANGE = (0.30, 0.99)

DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MAX_CAMERA_INDEX_TO_SCAN = 4     # camera selector checks indexes 0..3


# ----------------------------------------------------------------------
# 9. ANN configuration (adjustable if the model overfits)
# ----------------------------------------------------------------------
@dataclass
class ANNConfig:
    """
    Everything that defines the MLP and how it is trained.
    Change the numbers here (or from the Training page later) without
    touching any model code. The whole object is saved in meta.json next
    to the model, so you always know what a saved model was trained with.
    """

    hidden_layers: tuple = (512, 256, 128)    # neurons in each hidden layer
    dropout_rates: tuple = (0.4, 0.3, 0.2)    # one dropout rate per hidden layer
    activation: str = "relu"
    l2_regularization: float = 0.0            # raise to e.g. 0.001 if overfitting
    optimizer: str = "adam"
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 16
    early_stopping: bool = True
    early_stopping_patience: int = 8          # epochs without improvement

    _OPTIMIZERS = ("adam", "sgd", "rmsprop")
    _ACTIVATIONS = ("relu", "elu", "tanh")

    def validate(self) -> None:
        """Raise ValueError with a clear message if a value makes no sense."""
        if len(self.hidden_layers) == 0:
            raise ValueError("At least one hidden layer is required.")
        if len(self.hidden_layers) != len(self.dropout_rates):
            raise ValueError("Give exactly one dropout rate for each hidden layer.")
        if any((not isinstance(n, int)) or n < 1 for n in self.hidden_layers):
            raise ValueError("Hidden layer sizes must be whole numbers >= 1.")
        if any(not (0.0 <= r < 1.0) for r in self.dropout_rates):
            raise ValueError("Dropout rates must be between 0.0 and 0.99.")
        if self.activation not in self._ACTIVATIONS:
            raise ValueError(f"Activation must be one of {self._ACTIVATIONS}.")
        if self.optimizer not in self._OPTIMIZERS:
            raise ValueError(f"Optimizer must be one of {self._OPTIMIZERS}.")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0.")
        if self.l2_regularization < 0:
            raise ValueError("L2 regularization cannot be negative.")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("Epochs and batch size must be at least 1.")
        if self.early_stopping_patience < 1:
            raise ValueError("Early stopping patience must be at least 1.")

    def to_dict(self) -> dict:
        """Plain dict (lists instead of tuples) so it can be saved as JSON."""
        data = asdict(self)
        data["hidden_layers"] = list(self.hidden_layers)
        data["dropout_rates"] = list(self.dropout_rates)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ANNConfig":
        """Rebuild a config from saved JSON (unknown keys are ignored)."""
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__
                 and not k.startswith("_")}
        if "hidden_layers" in known:
            known["hidden_layers"] = tuple(known["hidden_layers"])
        if "dropout_rates" in known:
            known["dropout_rates"] = tuple(known["dropout_rates"])
        config = cls(**known)
        config.validate()
        return config


# ----------------------------------------------------------------------
# 10. Data augmentation (applied to the TRAINING split only, after splitting)
# ----------------------------------------------------------------------
@dataclass
class AugmentationConfig:
    copies_per_image: int = 3            # extra augmented copies per training image
    horizontal_flip: bool = True
    max_rotation_degrees: float = 10.0
    brightness_range: tuple = (0.8, 1.2)
    max_shift_fraction: float = 0.05     # shift up to 5% of the image size


# ----------------------------------------------------------------------
# 11. Helpers
# ----------------------------------------------------------------------
DEFAULT_ANN_CONFIG = ANNConfig()
AUGMENTATION = AugmentationConfig()


def validate_settings() -> None:
    """Check that the fixed settings are consistent. Raises ValueError."""
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must add up to 1.0 (got {total}).")
    if min(TRAIN_RATIO, VAL_RATIO, TEST_RATIO) <= 0:
        raise ValueError("Every split ratio must be greater than 0.")
    # Each split needs at least MIN_SAMPLES_PER_SPLIT images of every person.
    if MIN_IMAGES_PER_USER < 3 * MIN_SAMPLES_PER_SPLIT:
        raise ValueError("MIN_IMAGES_PER_USER is too small for a 3-way split.")
    lo, hi = CONFIDENCE_THRESHOLD_RANGE
    if not (lo <= DEFAULT_CONFIDENCE_THRESHOLD <= hi):
        raise ValueError("Default confidence threshold is outside its range.")
    DEFAULT_ANN_CONFIG.validate()


def ensure_directories() -> None:
    """Create every folder the application needs (safe to call repeatedly)."""
    for folder in (DATASET_DIR, MODELS_DIR, RESULTS_DIR, ICONS_DIR,
                   IMAGES_DIR, DB_PATH.parent):
        folder.mkdir(parents=True, exist_ok=True)


def person_folder_name(user_id: int) -> str:
    """Dataset folder name for a database user id, e.g. 7 -> 'person_007'."""
    return f"{PERSON_FOLDER_PREFIX}{user_id:03d}"
