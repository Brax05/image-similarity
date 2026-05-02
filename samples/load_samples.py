"""
Descarga 10 imágenes de ejemplo desde CIFAR-10 (HuggingFace) para probar
el modelo de similitud. Se guardan como .jpg en esta carpeta.

Uso:
    python samples/load_samples.py

Requiere: pip install datasets Pillow
"""

from pathlib import Path

SAMPLES_DIR = Path(__file__).parent

CATEGORIES = {
    0: "airplane",
    1: "automobile",
    3: "cat",
    5: "dog",
    2: "bird",
}

SAMPLES_PER_CLASS = 2


def load_samples():
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        print("Faltan dependencias. Ejecuta: pip install datasets Pillow")
        raise

    print("Cargando CIFAR-10 desde HuggingFace (solo split de test, ~30 MB)...")
    ds = load_dataset("cifar10", split="test")

    saved = []
    counters = {label: 0 for label in CATEGORIES}

    print(f"\nGuardando {SAMPLES_PER_CLASS} imágenes por categoría en: {SAMPLES_DIR}\n")

    for example in ds:
        label = example["label"]
        if label not in counters:
            continue
        if counters[label] >= SAMPLES_PER_CLASS:
            continue

        counters[label] += 1
        idx = counters[label]
        name = CATEGORIES[label]
        dest = SAMPLES_DIR / f"{name}_{idx}.jpg"

        img: Image.Image = example["img"]
        img = img.resize((128, 128), Image.LANCZOS)
        img.save(dest, "JPEG", quality=90)
        print(f"  [OK] {dest.name}  ({img.size[0]}×{img.size[1]}px)")
        saved.append(dest.name)

        if all(v >= SAMPLES_PER_CLASS for v in counters.values()):
            break

    print(f"\nGuardadas {len(saved)} imágenes:")
    for f in saved:
        print(f"  samples/{f}")

    print("\n--- Cómo usarlas en el notebook ---")
    print('  score = compare_images("samples/cat_1.jpg", "samples/cat_2.jpg")')
    print('  score = compare_images("samples/cat_1.jpg", "samples/dog_1.jpg")')
    print('  score = compare_images("samples/airplane_1.jpg", "samples/automobile_1.jpg")')


if __name__ == "__main__":
    load_samples()
