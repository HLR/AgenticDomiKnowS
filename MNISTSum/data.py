
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np

def get_data(size=1000, train=True):
    mnist_dataset = MNIST(root=".", train=train, download=True, transform=ToTensor())
    
    data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=len(mnist_dataset))
    images, labels = next(iter(data_loader))
    
    dataset = []
    for _ in range(size):
        idx_a, idx_b = np.random.randint(0, len(images), 2)
        
        img_a, digit_a = images[idx_a], labels[idx_a].item()
        img_b, digit_b = images[idx_b], labels[idx_b].item()
        
        dataset.append({
            'img_a': img_a,
            'img_b': img_b,
            'digit_a': digit_a,
            'digit_b': digit_b,
            'sum': digit_a + digit_b
        })
        
    return dataset

if __name__ == '__main__':
    data = get_data()
    print(f"Generated {len(data)} data points.")
    print("Example data point:")
    print(data[0])
