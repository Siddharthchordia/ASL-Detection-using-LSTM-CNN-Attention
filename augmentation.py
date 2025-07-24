import os
import numpy as np
import math
import shutil

DATA_PATH = "data2"
AUGMENTED_DATA_PATH = "data2_augmented"
SEQ_LENGTH = 60
ACTIONS = [
    "goodbye", "hello", "yes", "no", "thank you", "please", "sorry", "stop", "help",
    "what", "how", "where", "when", "eat", "drink", "sleep", "IDLE_STATE"
]

def horizontal_flip(sequence):
    flipped = sequence.copy()
    for i in range(0, 33 * 4, 4):
        flipped[i] = 1 - flipped[i]
    for i in range(33 * 4, 33 * 4 + 21 * 3, 3):
        flipped[i] = 1 - flipped[i]
    for i in range(33 * 4 + 21 * 3, len(sequence), 3):
        flipped[i] = 1 - flipped[i]
    return flipped

def rotate_keypoints(sequence, angle_degrees):
    angle_rad = math.radians(angle_degrees)
    cos_val, sin_val = math.cos(angle_rad), math.sin(angle_rad)
    rotated = sequence.copy()

    def rotate_xy(x, y):
        x_new = cos_val * (x - 0.5) - sin_val * (y - 0.5) + 0.5
        y_new = sin_val * (x - 0.5) + cos_val * (y - 0.5) + 0.5
        return x_new, y_new

    for i in range(0, 33 * 4, 4):
        rotated[i], rotated[i + 1] = rotate_xy(sequence[i], sequence[i + 1])
    for i in range(33 * 4, 33 * 4 + 21 * 3, 3):
        rotated[i], rotated[i + 1] = rotate_xy(sequence[i], sequence[i + 1])
    for i in range(33 * 4 + 21 * 3, len(sequence), 3):
        rotated[i], rotated[i + 1] = rotate_xy(sequence[i], sequence[i + 1])
    return rotated

def augmentation():
    # Reset the augmented directory
    if os.path.exists(AUGMENTED_DATA_PATH):
        shutil.rmtree(AUGMENTED_DATA_PATH)
    os.makedirs(AUGMENTED_DATA_PATH, exist_ok=True)

    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        new_action_path = os.path.join(AUGMENTED_DATA_PATH, action)
        os.makedirs(new_action_path, exist_ok=True)

        sequence_folders = sorted(
            [f for f in os.listdir(action_path) if f.isdigit()],
            key=lambda x: int(x)
        )
        new_seq_index = 0

        for seq in sequence_folders:
            seq_path = os.path.join(action_path, seq)
            frames = [np.load(os.path.join(seq_path, f"{i}.npy")) for i in range(SEQ_LENGTH)]
            original_seq = np.array(frames)

            # Save Original Sequence
            orig_path = os.path.join(new_action_path, str(new_seq_index))
            os.makedirs(orig_path, exist_ok=True)
            for i, frame in enumerate(original_seq):
                np.save(os.path.join(orig_path, f"{i}.npy"), frame)
            new_seq_index += 1

            # 1. Horizontal Flip
            flipped_seq = np.array([horizontal_flip(f) for f in original_seq])
            flip_path = os.path.join(new_action_path, str(new_seq_index))
            os.makedirs(flip_path, exist_ok=True)
            for i, frame in enumerate(flipped_seq):
                np.save(os.path.join(flip_path, f"{i}.npy"), frame)
            new_seq_index += 1

            # 2. Z-axis Rotations
            for angle in [20, -20, 45, -45, 60, -60]:
                rotated_seq = np.array([rotate_keypoints(f, angle) for f in original_seq])
                rot_path = os.path.join(new_action_path, str(new_seq_index))
                os.makedirs(rot_path, exist_ok=True)
                for i, frame in enumerate(rotated_seq):
                    np.save(os.path.join(rot_path, f"{i}.npy"), frame)
                new_seq_index += 1

    print("✅ All augmentations completed and saved to 'data2_augmented/'")

# Run augmentation
augmentation()