import os

from app import app

if __name__ == '__main__':
    if 'localhost' in os.environ:
        app.run(host='0.0.0.0', port=3001, debug=True)
    else:
        app.run(debug=False)
