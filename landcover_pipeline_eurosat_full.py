import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


# -------------------------------
# 1️⃣ Prepare Data Split
# -------------------------------
def prepare_data(data_dir, out_dir, val_split=0.2):
    os.makedirs(out_dir, exist_ok=True)
    dataset = datasets.ImageFolder(data_dir)
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    torch.save(train_set.indices, os.path.join(out_dir, "train_idx.pt"))
    torch.save(val_set.indices, os.path.join(out_dir, "val_idx.pt"))
    print(f"[INFO] Split dataset into {n_train} train and {n_val} val samples.")


# -------------------------------
# 2️⃣ Compute Normalization Stats
# -------------------------------
def compute_stats(data_dir, out_path):
    print("[INFO] Computing normalization stats...")
    dataset = datasets.ImageFolder(
        data_dir,
        transform=transforms.ToTensor()
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.
    std = 0.
    total = 0
    for imgs, _ in loader:
        batch_samples = imgs.size(0)
        imgs = imgs.view(batch_samples, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        total += batch_samples

    mean /= total
    std /= total
    np.savez(out_path, mean=mean.numpy(), std=std.numpy())
    print(f"[DONE] mean={mean}, std={std}")
    print(f"[SAVED] {out_path}")


# -------------------------------
# 3️⃣ Train Model + Visualization
# -------------------------------
def train_model(data_root, stats_path, save_dir, epochs=10, lr=1e-4, batch_size=32):
    os.makedirs(save_dir, exist_ok=True)
    stats = np.load(stats_path)
    mean, std = stats["mean"], stats["std"]

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_data = datasets.ImageFolder(os.path.join(data_root, "train"), transform=transform_train)
    val_data = datasets.ImageFolder(os.path.join(data_root, "val"), transform=transform_val)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0.0
    train_losses, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        train_losses.append(total_loss)
        val_accs.append(acc)

        print(f"[EPOCH {epoch+1}/{epochs}] Loss: {total_loss:.3f} | Val Acc: {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"[SAVED] Best model (acc={best_acc:.3f})")

    # Plot training curves
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label="Validation Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.show()
    print(f"[TRAIN COMPLETE] Best accuracy: {best_acc:.3f}")


# -------------------------------
# 4️⃣ Evaluate Model + Confusion Matrix
# -------------------------------
def evaluate(data_dir, model_path, stats_path):
    stats = np.load(stats_path)
    mean, std = stats["mean"], stats["std"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"[EVAL] Accuracy: {acc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


# -------------------------------
# 5️⃣ Inference + Visualization
# -------------------------------
def infer_and_save(image_path, model_path, stats_path, data_root, out_path="pred.png"):
    stats = np.load(stats_path)
    mean, std = stats["mean"], stats["std"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = datasets.ImageFolder(data_root)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, 1)
        label = idx_to_class[pred.item()]
        print(f"[INFER] Predicted: {label}")

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"Predicted: {label}"

    try:
        text_w, text_h = draw.textsize(text, font=font)
    except AttributeError:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    draw.rectangle([5, 5, 5 + text_w + 10, 5 + text_h + 10], fill=(0, 0, 0))
    draw.text((10, 10), text, fill=(255, 255, 0), font=font)
    img.save(out_path)
    print(f"[SAVED] {out_path}")
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# -------------------------------
# Main CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    prep = sub.add_parser("prepare_data")
    prep.add_argument("--data-dir", required=True)
    prep.add_argument("--out-dir", required=True)

    stats = sub.add_parser("compute_stats")
    stats.add_argument("--data-dir", required=True)
    stats.add_argument("--out", required=True)

    train = sub.add_parser("train")
    train.add_argument("--data-root", required=True)
    train.add_argument("--stats", required=True)
    train.add_argument("--save-dir", required=True)
    train.add_argument("--epochs", type=int, default=10)

    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--data-dir", required=True)
    eval_p.add_argument("--model", required=True)
    eval_p.add_argument("--stats", required=True)

    infer = sub.add_parser("infer")
    infer.add_argument("--image", required=True)
    infer.add_argument("--model", required=True)
    infer.add_argument("--stats", required=True)
    infer.add_argument("--data-root", required=True)
    infer.add_argument("--out", default="pred.png")

    args = parser.parse_args()

    if args.command == "prepare_data":
        prepare_data(args.data_dir, args.out_dir)
    elif args.command == "compute_stats":
        compute_stats(args.data_dir, args.out)
    elif args.command == "train":
        train_model(args.data_root, args.stats, args.save_dir, epochs=args.epochs)
    elif args.command == "evaluate":
        evaluate(args.data_dir, args.model, args.stats)
    elif args.command == "infer":
        infer_and_save(args.image, args.model, args.stats, args.data_root)
    else:
        print("Invalid command")


if __name__ == "__main__":
    main()
