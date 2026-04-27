import os
from dotenv import load_dotenv

load_dotenv()

import src.control.conversation_control
import src.control.get_response
import src.control.user_control
from src import app

if __name__ == "__main__":
    # Use port 5001 to avoid conflict with macOS AirPlay on port 5000
    app.run(port=5001, debug=True)
