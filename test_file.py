

import json
from flask import Flask

app = Flask(__name__)

@app.route('/derp')
def derp():
    return json.dumps({'response': 'hello nerd'})

if __name__ == '__main__':
    app.run()