from website import create_app

app = create_app()

# run flask application (running server)
if __name__ == '__main__':
    app.run(debug=True, port=5000)
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
