from image_loader import load_image
from cnn_model import predict

# ── config: set your image source here ───────────────────────────────────────
URL        = ""
LOCAL_PATH = r"C:\SANIN\Internship\materials\DATA\No_Uturn\015_1_0006.png"
# ─────────────────────────────────────────────────────────────────────────────

status = 'run'

while status == 'run':
    print("\n============================")
    print("  Traffic Sign Classifier")
    print("============================")
    print("1. Predict from local path")
    print("2. Predict from URL")
    print("3. Exit")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == '1':
        path = input("Enter local image path (or press Enter for default): ").strip()
        if not path:
            path = LOCAL_PATH
        try:
            img_array = load_image(local_path=path)
            class_name, confidence = predict(img_array)
            print(f"\n Predicted Class : {class_name}")
            print(f" Confidence      : {confidence}%")
        except Exception as e:
            print(f"[ERROR] {e}")

    elif choice == '2':
        url = input("Enter image URL (or press Enter for default): ").strip()
        if not url:
            url = URL
        if not url:
            print("[ERROR] No URL set. Please enter a URL.")
        else:
            try:
                img_array = load_image(url=url)
                class_name, confidence = predict(img_array)
                print(f"\n Predicted Class : {class_name}")
                print(f" Confidence      : {confidence}%")
            except Exception as e:
                print(f"[ERROR] {e}")

    elif choice == '3':
        print("Exiting...")
        status = 'stop'

    else:
        print("[ERROR] Invalid choice. Enter 1, 2, or 3.")
