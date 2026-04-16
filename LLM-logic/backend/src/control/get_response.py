from flask import jsonify, request, Response
import json
import time
import traceback
from openai import OpenAIError
import anthropic
import google.api_core.exceptions as GoogleAPIError

from src import app
from src.control.query import query, proslm_query, query_rag_service, stream_rag_service
from src.model.conversation_model import append_message_by_id


@app.route("/get_response", methods=["POST"])
def get_response_control():
    start_time = time.time()
    request_id = f"req_{int(start_time)}"

    print(f"\n[DEBUG] [{request_id}] ====== NEW REQUEST ======")

    try:
        data = request.json
        conv_id = data["conv_id"]
        method = data["method"]
        provider = data.get("provider", "openai")
        model = data.get("model")

        # Debug information
        print(f"[DEBUG] [{request_id}] Received request:")
        print(f"[DEBUG] [{request_id}] - Conversation ID: {conv_id}")
        print(f"[DEBUG] [{request_id}] - Method: {method}")
        print(f"[DEBUG] [{request_id}] - Provider: {provider}")
        print(f"[DEBUG] [{request_id}] - Model: {model}")

        # Save user message
        print(f"[DEBUG] [{request_id}] Saving user message to conversation")
        append_message_by_id(conv_id, data)

        # Extract user input
        input_text = data["messages"][-1]["content"]
        print(f"[DEBUG] [{request_id}] User input: {input_text[:50]}...")

        # Process method name
        processed_method = method
        if method == "chain-of-thought":
            processed_method = "cot"
        elif method == "rag-agent":
            # Handle RAG Agent method - uses the agentic RAG pipeline
            try:
                print(f"[DEBUG] [{request_id}] Using RAG Agent pipeline")
                # Build conversation history (all turns except the current one)
                raw_messages = data.get("messages", [])
                history = [
                    {"role": "assistant" if m["role"] == "bot" else m["role"], "content": m["content"]}
                    for m in raw_messages[:-1]
                    if m.get("role") in ("user", "bot", "assistant") and m.get("content")
                ]
                result = query_rag_service(input_text, use_router=True, history=history)
                response_json = {"choices": [{"message": {"content": result}}]}

                save_res = {
                    "messages": [{"role": "bot", "content": result}],
                    "provider": "rag",
                    "model": "rag-agent",
                }

                append_message_by_id(conv_id, save_res)

                total_time = time.time() - start_time
                print(f"[DEBUG] [{request_id}] RAG Agent completed in {total_time:.2f} seconds")
                return jsonify(response_json)
            except Exception as e:
                print(f"[ERROR] [{request_id}] RAG Agent error: {str(e)}")
                return jsonify({"error": f"RAG Agent Error: {str(e)}"}), 500
        elif method == "pro-slm":
            result = "The 'Pro-SLM' method is not available. Please use 'std' (standard LLM) or select RAG Agent."
            response_json = {"choices": [{"message": {"content": result}}]}
            return jsonify(response_json)

        print(f"[DEBUG] [{request_id}] Processed method: {processed_method}")

        try:
            # Get response from LLM
            print(f"[DEBUG] [{request_id}] Calling query function")
            response_message, actual_provider, actual_model = query(
                input_text, processed_method, provider, model
            )

            query_time = time.time() - start_time
            print(f"[DEBUG] [{request_id}] Query completed in {query_time:.2f} seconds")
            print(
                f"[DEBUG] [{request_id}] Response from {actual_provider} ({actual_model or 'default model'})"
            )
            print(
                f"[DEBUG] [{request_id}] Response length: {len(response_message)} characters"
            )

            # Add fallback notice if applicable
            response_content = response_message
            if "fallback" in actual_provider:
                response_content = f"[NOTE: Gemini quota exceeded, falling back to OpenAI]\n\n{response_message}"

            # Format response for frontend
            response_json = {"choices": [{"message": {"content": response_content}}]}

            # Save bot response to conversation
            print(f"[DEBUG] [{request_id}] Saving bot response to conversation")
            save_res = {
                "messages": [{"role": "bot", "content": response_content}],
                "provider": actual_provider,
                "model": actual_model,
            }
            append_message_by_id(conv_id, save_res)

            total_time = time.time() - start_time
            print(
                f"[DEBUG] [{request_id}] Total request processing time: {total_time:.2f} seconds"
            )
            print(f"[DEBUG] [{request_id}] ====== REQUEST COMPLETE ======\n")

            return jsonify(response_json)

        except OpenAIError as e:
            print(f"[ERROR] [{request_id}] OpenAI Error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": f"OpenAI Error: {str(e)}"}), 500
        except anthropic.APIError as e:
            # Using the correct Anthropic exception class
            print(f"[ERROR] [{request_id}] Claude Error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": f"Claude Error: {str(e)}"}), 500
        except GoogleAPIError.GoogleAPIError as e:
            print(f"[ERROR] [{request_id}] Gemini Error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": f"Gemini Error: {str(e)}"}), 500
        except Exception as e:
            print(f"[ERROR] [{request_id}] Unexpected error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        print(f"[ERROR] [{request_id}] Request processing error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/get_response_stream", methods=["POST"])
def get_response_stream_control():
    """
    Stream responses from the RAG Agent using Server-Sent Events.
    """
    start_time = time.time()
    request_id = f"stream_req_{int(start_time)}"

    print(f"\n[DEBUG] [{request_id}] ====== NEW STREAMING REQUEST ======")

    try:
        data = request.json
        conv_id = data["conv_id"]
        method = data.get("method", "rag-agent")

        # Save user message
        append_message_by_id(conv_id, data)

        # Extract user input
        input_text = data["messages"][-1]["content"]
        print(f"[DEBUG] [{request_id}] User input: {input_text[:50]}...")

        # Only RAG Agent supports streaming currently
        if method != "rag-agent":
            return jsonify({"error": "Streaming only supported for RAG Agent"}), 400

        # Build conversation history
        raw_messages = data.get("messages", [])
        history = [
            {"role": "assistant" if m["role"] == "bot" else m["role"], "content": m["content"]}
            for m in raw_messages[:-1]
            if m.get("role") in ("user", "bot", "assistant") and m.get("content")
        ]

        # Store complete response for saving to conversation
        complete_response = []

        def generate():
            try:
                for chunk in stream_rag_service(input_text, use_router=True, history=history):
                    # Parse the chunk to accumulate the answer
                    if chunk.startswith('data: '):
                        try:
                            import json
                            json_str = chunk[6:].strip()
                            if json_str:
                                event_data = json.loads(json_str)
                                if event_data.get('type') == 'token':
                                    complete_response.append(event_data.get('content', ''))
                        except:
                            pass
                    yield chunk

                # After streaming completes, save the full response
                full_response = ''.join(complete_response)
                if full_response:
                    save_res = {
                        "messages": [{"role": "bot", "content": full_response}],
                        "provider": "rag",
                        "model": "rag-agent",
                    }
                    append_message_by_id(conv_id, save_res)
                    print(f"[DEBUG] [{request_id}] Saved streaming response to conversation")

            except Exception as e:
                print(f"[ERROR] [{request_id}] Streaming error: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*'
            }
        )

    except Exception as e:
        print(f"[ERROR] [{request_id}] Stream setup error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
