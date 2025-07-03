# Gestura 🤟🎥

**Gestura** is a real-time American Sign Language (ASL) gesture recognition app that overlays live subtitles on a virtual camera, making video calls more accessible and inclusive.

The system uses a custom LSTM-CNN Attention-based model trained on a self-collected dataset, powered by MediaPipe landmark extraction.

---

## 🗂 Project Structure

```bash
├── ASL.py              # Core logic and model architecture (used as a library)
├── mainGUI.py          # Main entry point with GUI (run this to launch app)
├── model_training.py   # Script for training and saving gesture recognition model
├── models/             # Saved models
├── data2/              # Training data, not pushed to GitHub (Ignored)