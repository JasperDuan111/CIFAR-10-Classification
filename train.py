import torch
from torch import nn
from model import Mynet
from Data import get_data
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mynet = Mynet().to(device)
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(mynet.parameters())

# 学习率调度（当loss连续三次没有下降时，学习率减半）
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=3)

train_step = 0
epochs = 30
best_accuracy = 0.0

if __name__ == "__main__":
    train_data_loader, test_data_loader = get_data()
    for epoch in range(epochs):
        mynet.train()
        for j, (image, label) in enumerate(train_data_loader):
            image = image.to(device)
            label = label.to(device)

            output = mynet(image)
            loss = loss_function(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_step += 1
            if train_step % 100 == 0:
                print(f"Step {train_step}, Loss: {loss.item():.4f}")

        mynet.eval()
        total_test_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for j, (image, label) in enumerate(test_data_loader):
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
        
        # 取消以下注释以使用学习率调度
        # scheduler.step(avg_test_loss)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(mynet.state_dict(), f'./model/mynet_best_model.pth')
            print(f"New best model saved with accuracy: {accuracy:.4f}")