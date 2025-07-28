import os
from PIL import Image
import torch
from torchvision import transforms  
from model import Mynet

# CIFAR-10类别名称
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图片路径
img_path = 'test_images/test9.png'
img = Image.open(img_path)

# 图片处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)  # 从 [3, 32, 32] 变成 [1, 3, 32, 32]
img_tensor = img_tensor.to(device)

# 加载模型
model_path = 'model/mynet_best_model.pth'
model = Mynet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# 预测
with torch.no_grad():
    output = model(img_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()

# 显示结果
print(f"\nPrediction Results:")
print(f"Predicted class index: {predicted_class}")
print(f"Predicted class name: {CIFAR10_CLASSES[predicted_class]}")
print(f"Confidence: {probabilities[0][predicted_class].item()*100:.2f}%")

# 显示Top-5预测
print(f"\nTop-5 Predictions:")
top_probs, top_indices = torch.topk(probabilities[0], 5)
for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
    symbol = "→" if i == 0 else " "
    print(f"  {symbol} {CIFAR10_CLASSES[idx]}: {prob.item()*100:.2f}%")
