import torch
from torch import nn
from torch import optim

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, Resize
from torchvision.models.segmentation import deeplabv3_resnet50 as deeplab, DeepLabV3_ResNet50_Weights as weights

import numpy as np
import matplotlib.pyplot as plt

from  utils import create_seg_model, apply_color_map_seg

# preprocess the images and label 
def pre_process():

    img_transform = Compose([
        Resize((256, 512)),
        ToTensor()
    ])

    target_transform = Compose([
        Resize((256, 512)),
        Lambda(lambd= lambda x: np.array(x, dtype=np.int32)),
        ToTensor()
    ])

    return img_transform, target_transform

# load the data and preprocess while loading
def load_data(batch_size=10):
    img_transform, target_transform = pre_process()
    train_data = datasets.Cityscapes(root="data", split="train", mode="fine", target_type="semantic", transform=img_transform, target_transform=target_transform)
    test_data  = datasets.Cityscapes(root="data", split="test", mode="fine", target_type="semantic",  transform=img_transform, target_transform=target_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


# used to visualize model's preformance during training
def visualize_progress(image, label, logit):
   
    image = image.squeeze(dim=0)
    label = label.squeeze(dim=0)
    
    # predict pexil class
    pred  = logit.softmax(dim=1)
    pred   = pred.argmax(1)
   
    # display
    pred = pred.squeeze(dim=0)
    plt.subplot(1,3,1)
    plt.axis("off")
    plt.imshow(apply_color_map_seg(pred))
    plt.subplot(1,3,2)
    plt.axis("off")
    plt.imshow(image.squeeze(dim=0).cpu().permute(1,2,0))
    plt.subplot(1,3,3)
    plt.axis("off")
    plt.imshow(apply_color_map_seg(label.cpu().squeeze(dim=0)))
    plt.show()


# defining train loop
def train(model, train_loader, loss_fn, optimizer, epochs, batch_size, device):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n------------------------------------------------------------------------------------")
        for batch, (img, mask) in enumerate(train_loader):
            img, mask = img.to(device), mask.to(device)
            logit = model(img)["out"] 
            loss  = loss_fn(logit, mask.squeeze(dim=1).long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # progress report
            if batch % 100 == 0:
                loss, current  = loss.item(), batch * batch_size + len(img)
                print(f"loss: {loss:>7f} [{current:>5d} / {len(train_loader.dataset):>5d}]")
            # if batch % 100 == 0:
            #     visualize_progress(img[0].unsqueeze(0), mask[0].unsqueeze(0), logit[0].unsqueeze(0)) 
    print("Done!")
   
# defining test loop
def test(model, test_loader, loss_fn, device):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():  # load the pretrained model trained for 20 catagories
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logit = model(x)["out"]
            pred  = logit.softmax(dim=1)
            test_loss = test_loss + loss_fn(logit,y.squeeze(dim=1).long()).item()
            correct = correct + (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(test_loader)
    correct /= (len(test_loader.dataset) * x[0].shape[1] * x[0].shape[2]) # pixel accuracy
    print(f"Test Error: \n Pexil Accuracy: {(100*correct):>0.1f}, Avg loss: {test_loss:.8f} \n")

def main():
    # define parameters
    learning_rate = 1e-3
    num_class     = 34
    epochs        = 20
    batch_size    = 3
    
    # check if cude is available and set the device
    if(torch.cuda.is_available()):
        device = "cuda"
    else: 
        device = "cpu"
    
    # load the model 
    model = create_seg_model(numclass=num_class)
    model.to(device=device)
    model.eval()
   
    # load the train and test data
    train_loader, test_loader = load_data(batch_size=batch_size)

    # define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # run training loop
    train(model=model,train_loader=train_loader, 
          test_loader=test_loader, 
          loss_fn=loss_fn, 
          optimizer=optimizer, 
          epochs=epochs, 
          batch_size=batch_size,
          device=device
          )

    # post processing (save the model)
    torch.save(model, "experiment/Deeplabv3.pth")

if __name__ == "__main__":
    main()
