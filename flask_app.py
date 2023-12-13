import os
from flask import Flask, render_template, request, send_from_directory, url_for
from model_processing import labels, process_img, predict, new_model, pretrained_model
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['IMAGES_DIRECTORY'] = os.getcwd() + '/static/images'
EXTENSIONS = ['jpg', 'jpeg', 'png']


def allowed_file(filename):
    check_tab = [filename.endswith(x) for x in EXTENSIONS]
    return any(check_tab)


@app.route('/', methods=['GET', 'POST'])
def upload_function():
    if request.method == 'POST':
        image = request.files['img']
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['IMAGES_DIRECTORY'], filename))
        img = os.path.join('/static/images', filename)
        if allowed_file(filename):
            label1 = predict(process_img(img), new_model)
            label2 = predict(process_img(img), pretrained_model)
            return render_template('index.html', img=img, lab1=label1, lab2=label2)
        else:
            return render_template('index.html', ext='Extension not supported. Please try again')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
