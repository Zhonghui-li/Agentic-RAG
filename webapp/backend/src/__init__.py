import os
from flask import Flask
from flask_cors import CORS
from flask_pymongo import PyMongo

app = Flask(__name__)

# CORS: restrict to known frontend origins in production.
# Set ALLOWED_ORIGINS env var to a comma-separated list of domains once hosting URL is confirmed.
# Example: ALLOWED_ORIGINS=https://your-domain.com,http://localhost:3000
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
_allowed_origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else "*"
CORS(app, resources={r"/*": {"origins": _allowed_origins}})

app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://localhost:27017/flask_database")
mongo = PyMongo(app)

if __name__ == "__main__":
    app.run(debug=True)
