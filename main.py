from website import create_app
import os

app = create_app()

# run flask application (running server)
if __name__ == '__main__':
    app.run(debug=True)
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024