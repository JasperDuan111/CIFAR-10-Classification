import torch
from torch import nn
from model import Mynet, InceptionNet, ResNet18, AlexNet, VGGNet, InceptionNetGeLU, MobileNet
from Data import get_data
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mynet = InceptionNet().to(device)
# mynet.load_state_dict(torch.load('./model/mynet_best_model.pth', map_location=device))
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(
    mynet.parameters(),
    lr=0.01,               
    weight_decay=0.01,      
)

# ReduceLROnPlateau学习率调度（当每个epoch的loss连续三次没有下降时，学习率减半）
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=5)

train_step = 0
epochs = 30
best_accuracy = 0.0

if __name__ == "__main__":
    train_data_loader, test_data_loader = get_data()
    # # OneCycleLR 学习率调度器
    # steps_per_epoch = len(train_data_loader)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=0.1,            # 训练中最大的学习率
    #     epochs=epochs,
    #     steps_per_epoch=steps_per_epoch,
    #     pct_start=0.3,          # 上升阶段占比（前30%时间用于上升）
    #     anneal_strategy='cos',  # 余弦退火策略
    #     div_factor=25.0,        # 初始 lr = max_lr / div_factor
    #     final_div_factor=1e4,   # 最终 lr = max_lr / final_div_factor
    #     cycle_momentum=True,       # 开启动量反向调度
    #     base_momentum=0.85,        
    #     max_momentum=0.95        
    # )
    
    for epoch in range(epochs):
        mynet.train()
        for i, (image, label) in enumerate(train_data_loader):
            image = image.to(device)
            label = label.to(device)

            output = mynet(image)
            loss = loss_function(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # OneCycleLR更新学习率
            # scheduler.step()  

            train_step += 1
            if train_step % 100 == 0:
                print(f"Step {train_step}, Loss: {loss.item():.4f}")

        mynet.eval()
        total_test_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for i, (image, label) in enumerate(test_data_loader):
                image = image.to(device)
                label = label.to(device)

                output = mynet(image)
                loss = loss_function(output, label)
                
                total_test_loss += loss.item()
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        avg_test_loss = total_test_loss / len(test_data_loader)
        accuracy = (all_predictions == all_labels).mean()
        
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        print(f"-------Epoch {epoch + 1}--------------")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("----------------------------")
        
        # 取消以下注释以使用ReduceLROnPlateau学习率调度
        # scheduler.step(avg_test_loss)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(mynet.state_dict(), f'./model/{mynet.__class__.__name__}_best_model.pth')
            print(f"New best model saved with accuracy: {accuracy:.4f}")
            print("----------------------------")
