import os
import sys
import draw_graph
from flask import Flask, render_template, request
from draw_graph import get_performance, draw_full_scatter, draw_magnify_scatter, draw_subplot_scatter

sys.path.append(os.path.realpath('..'))
from config import root_path

app = Flask(__name__)
images_path = os.path.join(root_path, 'graph/static/images')
model_name = 'xjy_20231218'
performance = get_performance(model_name)
category_mapping = {
    'full': draw_full_scatter,
    'magnify': draw_magnify_scatter,
    'subplot': draw_subplot_scatter
}


@app.route('/')
def index():
    category = request.args.get('category', 'full')
    if category not in category_mapping:
        category = 'full'

    if str(request.args.get('repaint', '0')) == '1':
        category_mapping[category](performance)

    image_filenames = list(filter(lambda fn: fn.startswith(category), os.listdir(images_path)))
    image_fullpaths = [os.path.realpath(os.path.join(images_path, fn)) for fn in image_filenames]
    return render_template(
        'index.html',
        category=category,
        image_filenames=image_filenames,
        image_fullpaths=image_fullpaths
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
