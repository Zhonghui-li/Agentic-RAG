import type { Conv_Confirm } from './types';

async function StartNewConversation(userName: string, name: string) {
  try {
    const response = await fetch('http://127.0.0.1:5001/new_conversation', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ user: userName, name: name }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return (await response.json()) as Conv_Confirm;
  } catch (error) {
    console.error('There was a problem with the fetch operation:', error);
    return { error: 'An error occurred while sending the conversation.' };
  }
}

export default StartNewConversation;
