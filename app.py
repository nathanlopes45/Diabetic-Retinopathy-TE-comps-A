from flask import Flask, render_template, request
import torch
from PIL import Image
import io
import base64
import numpy as np
from torchvision import datasets,transforms,models
import torchvision.models as models 
from torch import nn


#FILE = 'model.pt'
#model = torch.load(FILE)
#model = model.eval()

# model = torch.load('C:\\Users\\mathe\\Desktop\\frontend-retinopathy\\model.pt',torch.device('cpu'))
# model.eval()

# class MLP(nn.Module):

#   def __init__(self):
#     super().__init__()
#     self.resnet = models.resnet152(pretrained=True)  # Example: Load pretrained ResNet18 model
#         # Modify the last layer of ResNet to match the number of input features for your custom layers
#     num_ftrs = self.resnet.fc.in_features
#     self.resnet.fc = nn.Linear(num_ftrs, 512)
#     self.layers = nn.Sequential(
#       nn.Linear(num_ftrs,512),
#       nn.ReLU(),
#       nn.Linear(512,5),
#       nn.LogSoftmax(dim=1)
#     )


#   def forward(self, x):
#     '''Forward pass'''
#     x = self.resnet(x)
#     x = torch.flatten(x, 1)
#     x = self.layers(x)
#     return x


app = Flask(__name__, template_folder='templates')

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])





def predict():
    if request.method == "POST":
        image = request.files.get('imagefield', '')
        image = Image.open(image)
        #img_array = np.array(image)

        test_transforms=transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        x = test_transforms(image)
        torch.save(x, "transformed_image.pt")

        model=models.resnet152(pretrained=True)
        model.fc=nn.Sequential(nn.Linear(model.fc.in_features,512),nn.ReLU(),nn.Linear(512,5),nn.LogSoftmax(dim=1))
        # model.to('cpu')

        checkpoint = torch.load('./model.pt',torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        device = torch.device("cpu")
        model.to(device)

        with torch.no_grad():
            x = x.unsqueeze(0)
            input_image = x  # Load image
            
            input_image = input_image.to(device)
            output = model(input_image)
            probabilities = torch.exp(output)
            top_p, top_class = probabilities.topk(1, dim=1)
            predicted_class = top_class.item()

        # Loading the saved model
        # save_path = './model.pt'

        # mlp = MLP()
        # mlp.load_state_dict(torch.load(save_path, torch.device('cpu'))['model_state_dict'])
        # mlp.eval()

        # Generate prediction
        # prediction = mlp(image)
        
        # # Predicted class value using argmax
        # predicted_class = np.argmax(prediction)

        print(predicted_class,"************************************************************************")
       
        # print(x)
        # print("*******************************")
        # print("*******************************")
        # print(x.shape)

        # result = model.forward(x)
        # _, y_hat = result.max(1)
        # print (str(result.item()))

        # with torch.no_grad():
        #     output = model(x.unsqueeze(0))
        #     prediction = torch.argmax(output, dim=1).item()

        # print(prediction)
       
        #print(img_array)
        return render_template('home.html', pred=predicted_class)
    else:
        return render_template('home.html', pred="Error")


if __name__ == '__main__':
    app.run(debug=True, port=8000)