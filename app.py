import gradio as gr
from fastai.vision.all import *
import webbrowser as wb
import requests

path = Path('model.pkl')
learn_inf = load_learner(path)
url = 'https://www.youtube.com/watch?v=_HCqOqGwDfE'
response = requests.get(url)
if response.status_code == 200:
    response = 'Accessed successfully.'
else:
    response = 'Failed to reach the URL.'

def img(input_img):
    input_img = PILImage.create(input_img)
    rc, _, _ = learn_inf.predict(input_img)
    if rc == 'black':
        wb.open(url)
    return rc, response

app = gr.Interface(fn=img, inputs=gr.Image(type='filepath'), outputs='text')
app.launch()
