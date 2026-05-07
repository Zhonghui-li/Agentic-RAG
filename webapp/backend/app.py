import os
from dotenv import load_dotenv

load_dotenv()

import src.control.conversation_control
import src.control.get_response
import src.control.user_control
from src import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False)
