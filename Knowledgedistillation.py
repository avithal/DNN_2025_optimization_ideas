import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
# 1. data preperation
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    """
    Combine distillation loss with standard cross-entropy
    """
    ce_loss = F.cross_entropy(student_logits, labels)
    print(f"ce_loss_student , Loss: {ce_loss:.4f}")

    soft_teacher = F.log_softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    eps = 1e-10
    soft_teacher = torch.clamp(soft_teacher, min=eps)
    soft_student = torch.clamp(soft_student, min=eps)
    ce_loss_teacher = F.cross_entropy(teacher_logits, labels)
    print(f"ce_loss_teacher , Loss: {ce_loss_teacher:.4f}")
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    return alpha * kd_loss + (1 - alpha) * ce_loss

if __name__ == '__main__':
    #  Load Teacher and Student Models
    teacher = models.resnet50(pretrained=True)
    student = models.resnet18(pretrained=False)

    # Modify output layer for CIFAR10 (10 classes)
    teacher.fc = nn.Linear(teacher.fc.in_features, 10)
    student.fc = nn.Linear(student.fc.in_features, 10)

    teacher.eval()  # Freeze teacher
    teacher.cuda()
    student.cuda()

    #Training Loop
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    for epoch in range(5):
        student.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)
            loss = distillation_loss(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}")



