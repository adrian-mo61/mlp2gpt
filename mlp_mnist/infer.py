# infer.py
import torch
from PIL import Image
from model import MnistModel
from data import transform  # 使用与训练相同的预处理

def predict(image_path, model_path, device='cpu'):
    """
    对单张手写数字图片进行预测

    Args:
        image_path (str): 图片路径（支持 JPG/PNG 等格式，灰度或彩色）
        model_path (str): 训练好的模型权重路径（.pth 文件）
        device (str): 运行设备，'cpu' 或 'cuda'

    Returns:
        int: 预测的数字类别 (0-9)
        float: 各类别的置信度（可选）
    """
    # 1. 加载模型（结构与训练时一致）
    input_size = 28 * 28
    hidden_size = [128, 64]
    output_size = 10
    model = MnistModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 2. 加载并预处理图片
    # 转为灰度图（MNIST 是单通道）
    image = Image.open(image_path).convert('L')
    # 应用训练时的变换：ToTensor + Normalize
    image = transform(image)
    # 增加 batch 维度（模型要求输入形状为 [batch, channels, height, width]）
    image = image.unsqueeze(0)
    image = image.to(device)

    # 3. 推理
    with torch.no_grad():
        output = model(image)                # 形状：[1, 10]
        probabilities = torch.softmax(output, dim=1)  # 转为概率分布
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_class].item()

    return pred_class, confidence


if __name__ == "__main__":
    # ---------- 使用示例 ----------
    # 请将 'test.png' 替换为你的图片路径，'best_mnist_model.pth' 替换为训练好的权重文件
    img_path = "test.png"
    model_path = "best_mnist_model.pth"

    pred, conf = predict(img_path, model_path, device='cpu')
    print(f"预测结果: {pred}，置信度: {conf:.4f}")