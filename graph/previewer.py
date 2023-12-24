import os
import sys
from flask import Flask, render_template

sys.path.append(os.path.realpath('..'))
from config import root_path

app = Flask(__name__)
images_path = os.path.join(root_path, 'graph/static/images')


@app.route('/')
def index():
    image_filenames = os.listdir(images_path)
    image_fullpaths = [os.path.realpath(os.path.join(images_path, fn)) for fn in image_filenames]
    return render_template(
        'index.html',
        image_filenames=image_filenames,
        image_fullpaths=image_fullpaths
    )


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
