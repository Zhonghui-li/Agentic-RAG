from bson.errors import InvalidId
from bson.json_util import dumps
from bson.objectid import ObjectId
from src import mongo
from flask import jsonify
import datetime
from typing import Optional, Dict, Any

CONV_COLLECTION = mongo.db.conversations


def is_valid_objectid(object_id) -> bool:
    """Check if a string is a valid MongoDB ObjectId."""
    try:
        ObjectId(object_id)
        return True
    except InvalidId:
        return False


def get_conversation(conv_id):
    """Get a conversation by its ID."""
    try:
        print(conv_id)
        conversation = CONV_COLLECTION.find_one(ObjectId(conv_id))
        print(conversation["_id"])
        print(conversation)
        try:
            return conversation_to_json(conversation)
        except:
            print("HERE")
            return False
    except:
        return False


def add_conversation(user_id, text, time, provider="openai", model=None):
    """
    Add a new conversation to the database.
    
    Args:
        user_id: The user ID
        text: The conversation text
        time: The conversation timestamp
        provider: The LLM provider (openai, claude, gemini)
        model: The specific model used
        
    Returns:
        The ID of the created conversation
    """
    conversation_id = CONV_COLLECTION.insert_one(
        {
            "user_id": ObjectId(user_id), 
            "text": text, 
            "time": time,
            "provider": provider,
            "model": model,
            "last_modified": datetime.datetime.now(tz=datetime.timezone.utc)
        }
    ).inserted_id
    return conversation_id


def get_all_conversations():
    """Get all conversations."""
    conversations = CONV_COLLECTION.find()
    return [conversation_to_json(conversation) for conversation in conversations]


def get_all_user_conversations(username):
    """Get all conversations for a specific user."""
    conversations = CONV_COLLECTION.find(
        {"name": {"$exists": True}, "user": username},
        {"_id": 1, "name": 1, "user": 1, "last_modified": 1, "provider": 1, "model": 1}
    ).sort("last_modified", -1)
    return [conversation_to_json(conversation) for conversation in conversations]


def conversation_to_json(conversation):
    """Convert a conversation document to JSON format."""
    conversation["_id"] = str(conversation["_id"])
    return conversation


def get_conversation_by_id(conv_id):
    """Get a conversation by its ID."""
    try:
        conversation = CONV_COLLECTION.find_one({"_id": ObjectId(conv_id)})
        return conversation_to_json(conversation)
    except InvalidId:
        return False


def update_conversation_by_id(conv_id, updates):
    """Update a conversation by its ID."""
    try:
        result = CONV_COLLECTION.update_one(
            {"_id": ObjectId(conv_id)}, {"$set": updates}
        )
        return result.matched_count > 0
    except InvalidId:
        return False


def delete_conversation_by_id(conv_id):
    """Delete a conversation by its ID."""
    try:
        result = CONV_COLLECTION.delete_one({"_id": ObjectId(conv_id)})
        return result.deleted_count > 0
    except InvalidId:
        return False


def append_message_by_id(conv_id, data):
    """
    Append a message to an existing conversation.
    
    Args:
        conv_id: The conversation ID
        data: The message data
        
    Returns:
        True if successful, False otherwise
    """
    prev_message = get_conversation(conv_id)

    if prev_message is False:
        return False
    if not is_valid_objectid(conv_id):
        return False

    msg = prev_message.get("messages", [])
    new_messages = data["messages"]

    # If provider and model are included in the data, update those fields too
    updates = {
        "messages": msg + new_messages, 
        "last_modified": datetime.datetime.now(tz=datetime.timezone.utc)
    }
    
    # Update provider and model if provided in the data
    if "provider" in data:
        updates["provider"] = data["provider"]
    if "model" in data:
        updates["model"] = data["model"]

    success = update_conversation_by_id(conv_id, updates)
    return success


def create_new_conversation(user_id, name, first_message, provider="openai", model=None):
    """
    Create a new conversation with initial message.
    
    Args:
        user_id: The user ID
        name: The conversation name
        first_message: The initial message
        provider: The LLM provider (openai, claude, gemini)
        model: The specific model used
        
    Returns:
        The ID of the created conversation
    """
    conversation = {
        "user": user_id,
        "name": name,
        "messages": [first_message],
        "provider": provider,
        "model": model,
        "created_at": datetime.datetime.now(tz=datetime.timezone.utc),
        "last_modified": datetime.datetime.now(tz=datetime.timezone.utc)
    }
    
    conversation_id = CONV_COLLECTION.insert_one(conversation).inserted_id
    return str(conversation_id)


def update_conversation_provider(conv_id, provider, model=None):
    """
    Update the provider and model for a conversation.
    
    Args:
        conv_id: The conversation ID
        provider: The new provider
        model: The new model (optional)
        
    Returns:
        True if successful, False otherwise
    """
    updates = {"provider": provider}
    if model:
        updates["model"] = model
    
    updates["last_modified"] = datetime.datetime.now(tz=datetime.timezone.utc)
    
    return update_conversation_by_id(conv_id, updates)


def get_conversation_stats():
    """
    Get statistics about conversations by provider and model.
    
    Returns:
        Dictionary with provider and model statistics
    """
    # Count by provider
    provider_stats = list(CONV_COLLECTION.aggregate([
        {"$group": {"_id": "$provider", "count": {"$sum": 1}}}
    ]))
    
    # Count by model
    model_stats = list(CONV_COLLECTION.aggregate([
        {"$group": {"_id": "$model", "count": {"$sum": 1}}}
    ]))
    
    # Format results
    providers = {stat["_id"] or "unknown": stat["count"] for stat in provider_stats}
    models = {stat["_id"] or "unknown": stat["count"] for stat in model_stats}
    
    return {
        "providers": providers,
        "models": models,
        "total": sum(providers.values())
    }
