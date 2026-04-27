from bson.errors import InvalidId
from flask import jsonify, request, session
from src import app
from src.model.user_model import (
    create_user,
    delete_user_by_id,
    get_all_users,
    is_valid_objectid,
    update_user_by_id,
    get_user_by_user_id,
)
from bcrypt import gensalt, hashpw, checkpw


@app.route("/user", methods=["POST"])
def create_user_route():
    data = request.json

    if "name" not in data or "user_id" not in data or "password" not in data:
        return jsonify({"error": "Name, User ID, and Password are required"}), 400

    user_id = create_user(data["name"], data["user_id"], data["password"])

    return jsonify({"message": "User created", "id": str(user_id)}), 201


@app.route("/user/all", methods=["GET"])
def get_users_route():
    users = get_all_users()
    return jsonify(users)


@app.route("/user/<string:user_id>", methods=["GET"])
def get_user_route(user_id):
    try:
        user = get_user_by_user_id(user_id)
        if user:
            return jsonify(user)
        return jsonify({"error": "User not found"}), 404
    except InvalidId:
        return jsonify({"error": "Invalid user ID"}), 400


@app.route("/user/<string:user_id>", methods=["PUT"])
def update_user_route(user_id):
    try:
        data = request.json

        if "name" not in data:
            return jsonify({"error": "Name is required"}), 400

        name = data["name"]
        user_id = data["user_id"]
        previous_password = data["previous_password"]
        new_password = data["new_password"]

        success = update_user_by_id(user_id, name, previous_password, new_password)

        if success:
            return jsonify({"message": "User updated"})

        return jsonify({"error": "User not found"}), 404

    except InvalidId:
        return jsonify({"error": "Invalid user ID"}), 400


@app.route("/user/<string:user_id>", methods=["DELETE"])
def delete_user_route(user_id):
    success = delete_user_by_id(user_id)

    if success:
        return jsonify({"message": "User deleted"})

    return jsonify({"error": "User not found"}), 404


@app.route("/login", methods=["POST"])
def login_user_route():
    data = request.json
    if "user_id" not in data or "password" not in data:
        return jsonify({"error": "User ID and Password are required"}), 400

    user = get_user_by_user_id(data["user_id"])

    if user and (
        checkpw(data["password"].encode("utf-8"), user["password"].encode("utf-8"))
    ):
        return jsonify(user), 200

    return jsonify({"error": "Invalid User ID or Password"}), 401
