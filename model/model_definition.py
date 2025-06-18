import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class CatBreedClassifierResNet50(nn.Module):
    def __init__(self, num_classes=12):
        super(CatBreedClassifierResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

def load_pytorch_model(model_path, num_classes, device='cpu'):
    torchvision_version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    if torchvision_version >= (0, 13):
        from torchvision.models import ResNet50_Weights
        try:
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except AttributeError:
             model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
        nn.LogSoftmax(dim=1)
    )

    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)

    model.eval()
    model = model.to(device)
    return model

if __name__ == '__main__':
    num_classes_test = 12
    print(f"Testing model definition with torchvision version {torchvision.__version__}")
    try:
        if torchvision_version >= (0, 13):
            from torchvision.models import ResNet50_Weights
            test_model_instance = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            test_model_instance = models.resnet50(pretrained=True)

        num_ftrs_test = test_model_instance.fc.in_features
        test_model_instance.fc = nn.Sequential(
            nn.Linear(num_ftrs_test, num_classes_test),
            nn.LogSoftmax(dim=1)
        )
        dummy_model_path = "dummy_resnet50_model.pth"
        torch.save(test_model_instance.state_dict(), dummy_model_path)
        print(f"Dummy model state_dict saved to {dummy_model_path}")

        loaded_model = load_pytorch_model(dummy_model_path, num_classes_test)
        print("Model definition and loading function seem OK.")

        dummy_input = torch.randn(1, 3, 224, 224)
        output = loaded_model(dummy_input.to(torch.device('cpu')))
        print(f"Output shape: {output.shape}")

        import os
        os.remove(dummy_model_path)
        print(f"Dummy model file {dummy_model_path} removed.")

    except Exception as e:
        print(f"Error in model definition/loading test: {e}")
        import traceback
        traceback.print_exc()