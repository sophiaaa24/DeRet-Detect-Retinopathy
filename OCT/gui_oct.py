import torch
import requests
import gradio as gr
from PIL import Image
from torchvision import transforms

model = torch.load('OCT model.pt')
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL' ]
def predict(inp):
 inp = Image.fromarray(inp.astype('uint8'), 'RGB')
 inp = transforms.ToTensor()(inp).unsqueeze(0)
 with torch.no_grad():
     prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
 return {class_names[i]: float(prediction[i]) for i in range(4)}
inputs = gr.inputs.Image(shape=(224,224))
outputs = gr.outputs.Label(num_top_classes=4)
gr.Interface(fn=predict, inputs=inputs, outputs=outputs).launch()
