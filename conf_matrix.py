import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import os
from param import TEST_DATA_DIR, CKPT_NET, NUM_CLASSES, DATA_DIR
from train_net import NeuralNet, CustomDataset


def conf_matrix():
    """モデルの評価と混同行列の作成"""
    # デバイスの設定
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ネットワークのロード
    net = NeuralNet(NUM_CLASSES).to(device)
    checkpoint = torch.load(CKPT_NET)
    net.load_state_dict(checkpoint)
    net.eval()  # 評価モード

    # データの準備
    val_transforms = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    test_dataset = datasets.ImageFolder(
        root=DATA_DIR, transform=val_transforms
    )  # 本当はTEST_DATA_DIR !!
    val_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=10, shuffle=False
    )

    all_labels = []
    all_preds = []

    # 予測の実行
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 混同行列の計算
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    # 混同行列の正規化
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # 混同行列の可視化
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=test_dataset.classes,
        yticklabels=test_dataset.classes,
    )
    plt.xlabel("Prediction")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.savefig("ConfMatrix.png")

    # 各クラスの識別率（efficiency）の計算
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, accuracy in enumerate(class_accuracies):
        print(f"Class '{test_dataset.classes[i]}': Efficiency = {accuracy * 100:.2f}%")

    print(f"Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    conf_matrix()
