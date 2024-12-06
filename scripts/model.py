import timm
from torch import nn

# Function to create a model adapted for a new task
def create_model(model_name='vgg16'):
    # Instantiate the model with specified configuration and pretrained weights
    model = timm.create_model(model_name, pretrained=True)

    # Update model heads based on model_name
    if model_name in ["vgg16", "vgg11", "vgg19"]:
        n_inputs = model.head.fc.in_features
        model.head.fc = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
      )

    elif model_name in ["resnet50", "resnet34"]:
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 2)
        )

    elif model_name in ["vit_base_patch16_224", "vit_tiny_patch16_224", "swin_tiny_patch4_window7_224"]:
        n_inputs = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 2)
        )
    
    return model
