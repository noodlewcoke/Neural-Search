from flask import Flask, jsonify, request, render_template
import torch
import torch.nn as nn
from models import biEncoder
from torchvision.datasets import MNIST
import torchvision
from search_trial import Seeker
from PIL import Image
import io
from flask_cors import CORS
import base64


app = Flask(__name__)
CORS(app)

mnist_data_test = MNIST('../data/', train=False, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))]))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
        'bias' : True,
        'lr' : 0
    }
model = biEncoder(config)
model.load('saved_models/biencoder2')
model.to(device)
model.eval()
img_db = torch.load('embeddings/embeddings2.pt')
search_engine = Seeker(model, img_db, device)

def transform_image(image_bytes):
    my_transforms = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
    
    # print(image_bytes[:5])
    if image_bytes.startswith('data:image/png;base64,'):
        image_bytes = image_bytes.replace('data:image/png;base64,', '')
    image_bytes = base64.b64decode(image_bytes)

    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((28, 28), Image.ANTIALIAS)
    image = image.convert("L")
    return my_transforms(image)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    results = search_engine(tensor, top_k=25)
    return results

def prepare_predictions(predictions):
    results = []

    for prediction in predictions:
        sim, index = prediction
        image = mnist_data_test.data[int(index)]
        image = Image.fromarray(image.numpy(), mode='L')
        image = image.resize((64, 64), Image.ANTIALIAS)

        b = io.BytesIO()
        image.save(b, 'png')
        im_bytes = b.getvalue()
        image = base64.b64encode(im_bytes)
        results.append([sim, image.decode()])
    return results

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        # print(request.get_json())
        # file = request.files['file']
        # image_bytes = file.read()
        image_bytes = request.get_json()['file']
        results = get_prediction(image_bytes=image_bytes)
        results = prepare_predictions(results)
        return jsonify({'top_25' : results})


if __name__ == '__main__':
    app.run()