import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.models import alexnet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Load pretrained AlexNet
model = alexnet(pretrained=True).cuda()

# Optimizer & loss (assume dataset is set up)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Helper functions
def prune_model(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)

def remove_masks(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
            except:
                pass

if __name__ == '__main__':
    # Iterative pruning loop
    pruning_schedule = [0.2, 0.2, 0.2]  # Prune 20% of weights per stage
    ##

    # Replace this with your actual path
    imagenet_train_path = r'D:\Dataset\tinyimagenet\tiny-imagenet-200\train'

    # ImageNet normalization values
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Define transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Load dataset
    train_dataset = datasets.ImageFolder(root=imagenet_train_path, transform=train_transforms)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    NUM_EPOCHS =5
    for i, amount in enumerate(pruning_schedule):
        print(f"\n--- Iteration {i+1}: Pruning {amount*100}% of weights ---")
        prune_model(model, amount)

        # Fine-tune
        for epoch in range(NUM_EPOCHS):
            print(f"\n--- epoch {epoch + 1}")
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:  # Assume train_loader is defined
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {running_loss:.4f}")

    # Finalize pruning
    remove_masks(model)
    torch.save(model.state_dict(), "alexnet_pruned.pth")
    # def compute_sparsity(tensor):
    #     return 100. * float(torch.sum(tensor == 0)) / tensor.numel()
    #
    # print(f"Sparsity: {compute_sparsity(model.fc.weight):.2f}%")
