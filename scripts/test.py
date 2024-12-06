import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.nn import functional as F

def test_model(model, testloader):
    
    criterion = nn.CrossEntropyLoss()
    total_loss, total_acc, label_list, pred_list, output_list = test(model, testloader, criterion)
    
    return total_loss, total_acc, label_list, pred_list, output_list 


# Get the accuracy and predicted labels
def test(model, testloader, criterion):
    model.to("cpu")
  
    model.eval() 
    # model.train()
    running_loss = 0.0
    running_corrects = 0
    label_list = []
    pred_list = []
    output_list = []

    for inputs, labels in testloader:
        input_size = len(testloader.dataset)
        with torch.no_grad():
           
            outputs = model(inputs)
        
    

            probabilities = F.softmax(outputs, dim=1)
            labels_indices = torch.argmax(labels, dim=1)
            loss = criterion(outputs, labels_indices)
            
            _, preds = torch.max(probabilities, 1)
            # consider the batch size
            for i in range (len(preds)):
              pred_list.append(preds[i].item())
            for j in range (len(labels_indices)):
              label_list.append(labels_indices[i].item())
            for k in range (len(probabilities)):
              output_list.append(probabilities[i].to("cpu").numpy())
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels_indices)
    total_loss = running_loss / input_size
    total_acc = running_corrects.double() / input_size
    total_acc = total_acc.to("cpu").numpy()
    
    print('Loss: {:.4f} Acc: {:.4f}'.format(total_loss, total_acc))
    return total_loss, total_acc, label_list, pred_list, output_list


