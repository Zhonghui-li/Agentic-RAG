import { useState, useRef, useEffect, useContext } from 'react';

import { AuthContext } from '@/auth/AuthProvider';
import getOneConversation from '@/data/getOneConversation';
import type { ChatGPTResponse } from '@/hooks/type';
import { AppContext } from '@/pages/AppContext';

import StartNewConversation from './ConversationFlask';
import type { Conv_Confirm, Result, Message, Conversation } from './types';

export const useChatHandler = () => {
  const [textInput, setTextInput] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const { currConversation, setCurrConversation, setNewConvCount } = useContext(AppContext);
  const authContext = useContext(AuthContext);
  // Track if we just created a new conversation to skip the fetch
  const isNewConversation = useRef<boolean>(false);

  useEffect(() => {
    if (currConversation !== '') {
      // Skip fetching if we just created this conversation (messages are already in state)
      if (isNewConversation.current) {
        isNewConversation.current = false;
        return;
      }

      const fetchConversation = async () => {
        const data: Conversation = await getOneConversation(currConversation);
        const conversation = data['messages'];
        if (conversation) {
          const msgs: Message[] = JSON.parse(JSON.stringify(conversation)) as Message[];
          setMessages(msgs);
        } else {
          setMessages([]);
        }
      };

      void fetchConversation();
    } else {
      setMessages([]);
    }
  }, [currConversation]);

  const handleClick = async (method: string, provider = 'openai'): Promise<void> => {
    if (textInput.trim() === '') return;

    let conversationId = currConversation;

    if (currConversation === '') {
      const data = (await StartNewConversation(
        authContext?.user || '',
        textInput.substring(0, 20),
      )) as Conv_Confirm;

      conversationId = data.id;
      // Mark as new conversation so useEffect won't overwrite our messages
      isNewConversation.current = true;
      setCurrConversation(data.id);
      setNewConvCount((prevCount) => prevCount + 1);
    }
    console.log(`Method: ${method}, Provider: ${provider}`);

    try {
      const userMessage: Message = { role: 'user', content: textInput };
      setMessages((prevMessages) => [...prevMessages, userMessage]);

      // Handle RAG Agent with streaming
      if (method === 'rag-agent') {
        const body = {
          model: 'rag-agent',
          messages: [userMessage],
          conv_id: conversationId,
          method: 'rag-agent',
          provider: 'rag',
        };

        setTextInput('');

        // Add placeholder bot message for streaming
        const botMessageIndex = messages.length + 1; // +1 for user message just added
        setMessages((prevMessages) => [...prevMessages, { role: 'bot', content: '' }]);

        try {
          const res = await fetch('http://127.0.0.1:5001/get_response_stream', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
          });

          if (!res.ok) {
            throw new Error('Streaming request failed');
          }

          const reader = res.body?.getReader();
          const decoder = new TextDecoder();
          let accumulatedContent = '';
          let statusMessage = '';

          if (reader) {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              const chunk = decoder.decode(value, { stream: true });
              const lines = chunk.split('\n');

              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  try {
                    const jsonStr = line.slice(6).trim();
                    if (jsonStr) {
                      const eventData = JSON.parse(jsonStr) as {
                        type: string;
                        content: string;
                      };

                      if (eventData.type === 'status') {
                        // Show status as italic text
                        statusMessage = `_${eventData.content}_\n\n`;
                        setMessages((prevMessages) => {
                          const newMessages = [...prevMessages];
                          newMessages[botMessageIndex] = {
                            role: 'bot',
                            content: statusMessage + accumulatedContent,
                          };
                          return newMessages;
                        });
                      } else if (eventData.type === 'token') {
                        // Accumulate the actual answer
                        accumulatedContent += eventData.content;
                        setMessages((prevMessages) => {
                          const newMessages = [...prevMessages];
                          newMessages[botMessageIndex] = {
                            role: 'bot',
                            content: accumulatedContent,
                          };
                          return newMessages;
                        });
                      } else if (eventData.type === 'done') {
                        // Streaming complete - final message is already set
                        console.log('Streaming complete');
                      } else if (eventData.type === 'error') {
                        setMessages((prevMessages) => {
                          const newMessages = [...prevMessages];
                          newMessages[botMessageIndex] = {
                            role: 'bot',
                            content: `Error: ${eventData.content}`,
                          };
                          return newMessages;
                        });
                      }
                    }
                  } catch (parseError) {
                    console.error('Failed to parse SSE data:', parseError);
                  }
                }
              }
            }
          }
        } catch (streamError) {
          console.error('Streaming error:', streamError);
          // Fallback to non-streaming if streaming fails
          const res = await fetch('http://127.0.0.1:5001/get_response', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
          });

          if (res.ok) {
            const response = (await res.json()) as ChatGPTResponse;
            const responseMessage =
              response.choices[0]?.message?.content || 'No response from RAG Agent';
            setMessages((prevMessages) => {
              const newMessages = [...prevMessages];
              newMessages[botMessageIndex] = { role: 'bot', content: responseMessage };
              return newMessages;
            });
          }
        }
        return;
      }

      // Select default model based on provider
      let MODEL_NAME = 'gpt-3.5-turbo';
      if (provider === 'claude') {
        MODEL_NAME = 'claude-3-opus-20240229';
      } else if (provider === 'gemini') {
        MODEL_NAME = 'gemini-pro';
      }

      const body = {
        model: MODEL_NAME,
        messages: [userMessage],
        conv_id: conversationId,
        method: method === 'chain-of-thought' ? 'cot' : method,
        provider: provider, // Add provider to the request
      };

      setTextInput('');

      const res = await fetch('http://127.0.0.1:5001/get_response', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        throw new Error('Network response was not ok');
      }

      const response = (await res.json()) as ChatGPTResponse;
      if (
        !response.choices ||
        !response.choices[0] ||
        !response.choices[0].message ||
        !response.choices[0].message.content
      ) {
        throw new Error('Invalid response format');
      }

      const responseMessage = response.choices[0].message.content;

      const result: Result = {
        type: 'success',
        data: responseMessage,
      };

      const botMessage: Message =
        result.type === 'success'
          ? { role: 'bot', content: result.data }
          : { role: 'bot', content: 'An error occurred while fetching the response.' };

      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.log(error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { role: 'bot', content: textInput },
        { role: 'bot', content: 'An error occurred while fetching the response.' },
      ]);
    } finally {
      setTextInput('');
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return {
    textInput,
    setTextInput,
    messages,
    handleClick,
    messagesEndRef,
  };
};
