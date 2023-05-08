import os
import flask
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request 
from tensorflow.keras.preprocessing.image import load_img , img_to_array


from src.utils import check_image_extension




app = Flask(__name__)
model_dir = "./models/"

model = load_model(os.path.join(model_dir , 'scratch_model.h5'))


ALLOWED_EXT = set(['jpeg', 'JPEG', 'png', 'PNG', 'JPG', 'jpg'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = ["EO", "IO", "IPTE", "LO" ,"PTE"]


def predict(image_path):
    
    img = load_img(image_path , target_size = (256 , 256))
    img = img_to_array(img)
    img = img.reshape(1 , 256 , 256 , 3)
    img = img.astype('float32')
    img = img/255.0
    pred = model.predict(img)
    pred = pred.argmax()
    pred = classes[pred]
    return pred


@app.route('/')
def home():
        return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                # check if the image has jpeg extension if not change it to jpeg
                file.filename = check_image_extension(file.filename)
                
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                class_result  = predict(img_path)


            else:
                error = "Please upload images of jpeg extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = class_result)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True, port=6000)


