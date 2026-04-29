"""
setup_project.py
Run once to create the full project folder structure.

Usage:
    python setup_project.py
"""

import os

# ─────────────────────────────────────────────
# FOLDERS TO CREATE
# ─────────────────────────────────────────────
FOLDERS = [
    # Source code
    "src/detection",
    "src/state",
    "src/reasoning",
    "src/evaluation",
    "src/utils",

    # Dashboard
    "dashboard",

    # Scripts
    "scripts",

    # Config
    "configs",

    # Data
    "data/visdrone/sequences",
    "data/visdrone/annotations",

    # Tests
    "tests",

    # Logs
    "logs",

    # Docs
    "docs/images",
]

# ─────────────────────────────────────────────
# STUB FILES TO CREATE
# ─────────────────────────────────────────────
STUBS = {
    # Package inits
    "src/__init__.py":            "",
    "src/detection/__init__.py":  "",
    "src/state/__init__.py":      "",
    "src/reasoning/__init__.py":  "",
    "src/evaluation/__init__.py": "",
    "src/utils/__init__.py":      "",

    # Phase 1 — Detection
    "src/detection/detector.py":      "# YOLO26s object detector",
    "src/detection/tracker.py":       "# ByteTrack multi-object tracker",
    "src/detection/zone_manager.py":  "# Zone loading and intersection logic",

    # Phase 2 — State
    "src/state/schema.py":            "# LangGraph state schema",
    "src/state/updater.py":           "# State updater node",
    "src/state/behaviors.py":         "# Loitering, probing, crowd detection",

    # Phase 3 — Reasoning
    "src/reasoning/prompts.py":       "# UAE security analyst prompts",
    "src/reasoning/reasoner.py":      "# GPT-4o-mini reasoning chain",
    "src/reasoning/alert_logger.py":  "# Structured alert log writer",

    # Phase 4 — Dashboard
    "dashboard/app.py":               "# Streamlit surveillance dashboard",

    # Phase 5 — Evaluation
    "src/evaluation/visdrone_parser.py": "# VisDrone annotation parser",
    "src/evaluation/evaluator.py":       "# MOTA and IDF1 computation",
    "src/evaluation/false_alert.py":     "# FAR and MTTA measurement",

    # Utils
    "src/utils/logger.py":            "# Shared logger",
    "src/utils/video_utils.py":       "# Frame loading and resizing helpers",

    # Scripts
    "scripts/run_demo.py":            "# Run full pipeline on test_video",
    "scripts/evaluate.py":            "# Run VisDrone benchmark evaluation",

    # Tests
    "tests/__init__.py":              "",
    "tests/test_detection.py":        "# Tests for detector and tracker",
    "tests/test_state.py":            "# Tests for LangGraph state",
    "tests/test_reasoning.py":        "# Tests for LLM reasoner",
    "tests/test_evaluation.py":       "# Tests for evaluation metrics",

    # Gitkeep for empty data folders
    "data/visdrone/sequences/.gitkeep":   "",
    "data/visdrone/annotations/.gitkeep": "",
    "logs/.gitkeep":                       "",
}


# ─────────────────────────────────────────────
# CREATE STRUCTURE
# ─────────────────────────────────────────────

def create_structure():
    print("Creating project structure...\n")

    # Folders
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"  📁 {folder}/")

    print()

    # Stub files
    for path, content in STUBS.items():
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content + "\n" if content else "")
            print(f"  📄 {path}")
        else:
            print(f"  ⏭  {path} (exists)")

    print("\n✅ Project structure created.")
    print("\nNext steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. cp .env.example .env  (then add your API key)")
    print("  3. Place test_video.mp4 in data/")
    print("  4. python tests/test_detection.py")


if __name__ == "__main__":
    create_structure()