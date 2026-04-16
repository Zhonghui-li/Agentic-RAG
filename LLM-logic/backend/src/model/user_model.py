from bson.errors import InvalidId
from bson.objectid import ObjectId
from src import mongo
from bcrypt import gensalt, hashpw, checkpw

USER_COLLECTION = mongo.db.users


def is_valid_objectid(user_id) -> bool:
    try:
        USER_COLLECTION.find_one({"_id": ObjectId(user_id)})
        return True
    except InvalidId:
        return False


def create_user(name, user_id, password):

    hashed_password = hashpw(password.encode("utf-8"), gensalt())

    user = USER_COLLECTION.insert_one(
        {"name": name, "user_id": user_id, "password": hashed_password.decode("utf-8")}
    )

    return user.inserted_id


def get_all_users():
    users = list(USER_COLLECTION.find())
    return [{**user, "_id": str(user["_id"])} for user in users]


def get_user_by_user_id(user_id):
    user = USER_COLLECTION.find_one({"user_id": user_id})
    if user:
        user["_id"] = str(user["_id"])
    return user


def update_user_by_id(user_id, name, previous_password, new_password):

    user = get_user_by_user_id(user_id)
    if checkpw(previous_password.encode("utf-8"), user["password"].encode("utf-8")):
        hashed_password = hashpw(new_password.encode("utf-8"), gensalt())

        result = USER_COLLECTION.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "name": name,
                    "user_id": user_id,
                    "password": hashed_password.decode("utf-8"),
                }
            },
        )

        return result.modified_count > 0
    else:
        return -1


def delete_user_by_id(user_id):
    try:
        result = USER_COLLECTION.delete_one({"user_id": user_id})
        return result.deleted_count > 0
    except InvalidId:
        return False
