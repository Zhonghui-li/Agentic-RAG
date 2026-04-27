import type { Conversation } from '@/components/UserInput/types';

async function getOneConversation(conv_id: string) {
  try {
    const response = await fetch(`http://127.0.0.1:5001/conversation/${conv_id}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const conversation = (await response.json()) as Conversation;
    return conversation;
  } catch (error) {
    console.error('There was a problem with the fetch operation:', error);
    return { _id: '', last_modified: '', name: '', messages: [], user: '' };
  }
}

export default getOneConversation;
