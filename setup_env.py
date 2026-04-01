"""
AgriVisionAI — First-time .env setup helper.
Called by start_agrivision.bat during initial configuration.
"""
import sys
import os

def main():
    print("=" * 51)
    print("       [INITIAL SETUP DETECTED]")
    print("=" * 51)
    print("A Groq API Key enables the AI Recovery Engine.")
    print("Get a free key at: https://console.groq.com/keys")
    print()

    groq_key = input("Paste your GROQ_API_KEY (or press Enter to skip): ").strip()
    blur_thresh = input("Set Blur Threshold (press Enter for default 80): ").strip()

    if not blur_thresh:
        blur_thresh = "80"

    env_content = f"""# AgriVisionAI Configuration - Auto-generated
GROQ_API_KEY={groq_key}
GROQ_MODEL=llama-3.3-70b-versatile
MODEL_PATH=models/best_model.pth
CLASS_NAMES_PATH=models/class_names.json
BLUR_THRESHOLD={blur_thresh}
CONFIDENCE_THRESHOLD=0.60
"""

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    with open(env_path, "w") as f:
        f.write(env_content)

    print()
    print("[OK] .env file successfully configured!")
    print("=" * 51)
    print()

if __name__ == "__main__":
    main()
