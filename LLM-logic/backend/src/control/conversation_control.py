from bson.errors import InvalidId
from flask import jsonify, request
from src import app, mongo
import datetime

from src.model.conversation_model import (
    delete_conversation_by_id,
    get_all_conversations,
    get_all_user_conversations,
    get_conversation_by_id,
    is_valid_objectid,
    append_message_by_id,
)


@app.route("/new_conversation", methods=["POST"])
def add_conversation_route():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid input"}), 400

        data["last_modified"] = datetime.datetime.now(tz=datetime.timezone.utc)

        conversation_id = mongo.db.conversations.insert_one(data).inserted_id
        return (
            jsonify({"message": "Conversation added", "id": str(conversation_id)}),
            201,
        )
    except InvalidId:
        return jsonify({"error": "Invalid ID format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/conversation/all", methods=["GET"])
def get_conversations_route():
    conversations = get_all_conversations()
    return jsonify(conversations), 200


@app.route("/conversation/user/<string:username>", methods=["GET"])
def get_user_conversation_route(username):
    conversations = get_all_user_conversations(username)
    return jsonify(conversations), 200


@app.route("/conversation/<string:conv_id>", methods=["GET"])
def get_conversation_route(conv_id):
    conversation = get_conversation_by_id(conv_id)
    return jsonify(conversation)


@app.route("/conversation/new_message/<string:conv_id>", methods=["PUT"])
def new_message_route(conv_id):
    try:
        data = request.get_json()
        success = append_message_by_id(conv_id, data)
        if success:
            return jsonify({"message": "Conversation updated"})
        return jsonify({"error": "Conversation not found"}), 404
    except InvalidId:
        return jsonify({"error": "Invalid Conversation ID"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/conversation/<string:conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    if not is_valid_objectid(conv_id):
        return jsonify({"error": "Invalid conversation ID"}), 400

    success = delete_conversation_by_id(conv_id)
    if success:
        return jsonify({"message": "Conversation deleted"})
    return jsonify({"error": "Conversation not found"}), 404
