from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

MODEL_PATH = os.path.join('model', 'best_model_classification.keras')

model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['Kaca', 'Kardus', 'Kertas', 'Logam', 'Plastik', 'Residu']
recommendations = {
    'Kaca': [
        {'title': 'Empty Glass Bottle Reuse Ideas | DIY / Decorative Lantern from recycled glass jar | Upcycling | Lamp', 'link': 'https://youtu.be/_OJ6Hz5vFNY?si=CeqxLMYN57hGi4xC'},
        {'title': 'How to make transparent hanging pot | DIY hanging pot | Wall hanging planters | DIY gardening ideas', 'link': 'https://youtu.be/hvEn6JRJyK4?si=e6_4CIIJvdd39bSh'},
        {'title': 'Three amazing handicraft ideas from glass and white cement. tiga ide kerajinan dari pecahan kaca.', 'link': 'https://youtu.be/VDCM6WqCG6k?si=HeLp0BE031-cRTjx'}
    ],
    'Kardus': [
        {'title': 'REALISTIC MINI BASKET FROM CARDBOARD | DIY Handmade Cardboard Craft | Best Display Ideas', 'link': 'https://youtu.be/-WPRaZ_SO9E?si=ix5FR_0f-cWGv37U'},
        {'title': 'How to create beautiful photo frame only using cardboard / easy homemade DIY', 'link': 'https://youtu.be/2op4UHuUiQw?si=ahjThRf9D8scmy8e'},
        {'title': '4 Cardboard box wall shelf decorating ideas | DIY wall shelf decor | Easy Crafts', 'link': 'https://youtu.be/VRRiPU2umCo?si=-dQz72BBPvxTNBky'}
    ],
    'Kertas': [
        {'title': 'DIY Paper Christmas Gift Box: Easy Craft Idea for Festive Giving!', 'link': 'https://youtu.be/lZETbW1s2e0?si=fAP0nDr64XlP41t6'},
        {'title': 'Cara membuat kreasi kerajinan keranjang dari kertas bekas, koran bekas, majalah bekas', 'link': 'https://youtu.be/7h7Xh-c4KJ0?si=04pWowS7VooG9yAA'},
        {'title': 'Easy giant wall decor with old newspaper or old book | newspaper wall hanging flower decor', 'link': 'https://youtu.be/Esrj055DtA4?si=bf-DtAKUCO0PD_UK'}
    ],
    'Logam': [
        {'title': 'Crafting Miniature Masterpieces: Upcycling Aluminum Cans into Tiny Pots', 'link': 'https://youtube.com/shorts/00L3nudLdGo?si=8yCXVJlenq1XN05H'},
        {'title': 'Ide kreatif tempat pensil dari kaleng bekas | How to make a pencil case from used cans', 'link': 'https://youtu.be/uBlFoMjQYKQ?si=M53BqF3qSaMElaCL'},
        {'title': 'Crafts with soda cans/coca cola bottle craft ideas', 'link': 'https://youtu.be/is5vyLbpBpI?si=hCgMjKF6EPoKrLjC'}
    ],
    'Plastik': [
        {'title': 'Plastic Bottle Flower Vase DIY Ideas | Home Decor |', 'link': 'https://youtu.be/hYDkLNW4deU?si=g95-g0zOCxPGPqTw'},
        {'title': 'DIY Plastic Bottle Enchanted Rose | Best Out of Waste |', 'link': 'https://youtu.be/cGJdhgCA9JE?si=rywUOjwfXdXE0s66'},
        {'title': 'Upcycled coffee sachet / Plastic wrapper purse #handmadecraft', 'link': 'https://youtu.be/uIFvr71I8Go?si=5bYp01SGgn_9NZOP'}
    ],
    'Residu': [
        {'title': 'Cara Membuat Bunga dari Styrofoam | Kerajinan dari Gabus Styrofoam | DIY Rose Flower', 'link': 'https://youtu.be/HgqwGoY0wNU?si=E59koIHQeVIqbmu4'},
        {'title': 'Kerajinan dari Kertas Bungkus Nasi || Ide Kreatif Kertas Nasi || Paper Craft', 'link': 'https://youtu.be/cZ-VxhP7kXM?si=O4RCcYCVniG40MHk'},
        {'title': 'Daur ulang sampah plastik bungkus snack menjadi tempat pensil || Recycle from plastic snack', 'link': 'https://youtu.be/MJd3bo_XRaU?si=20WoBsQkrtdGcYlJ'}
    ]
}


IMG_SIZE = (224, 224)


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file:
        try:
            img = preprocess_image(file.read())

            predictions = model.predict(img)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            reusable_goods = recommendations.get(predicted_class, [])

            return jsonify({
                'class': predicted_class,
                'confidence': float(confidence),
                'recommendations': reusable_goods
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file'}), 400


if __name__ == '__main__':
    app.run()
