from app import create_app

app = create_app()

if __name__ == "__main__":
    # Make this very explicit - host and port
    app.run(host='0.0.0.0', port=5000, debug=True)
    print("Server running on all interfaces at port 5000")