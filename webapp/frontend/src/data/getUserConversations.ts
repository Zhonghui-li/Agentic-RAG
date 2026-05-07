import { BACKEND_URL } from '@/lib/apiConfig';
import type { UserConversation } from '@/components/types';

async function getUserConversations(user: string) {
  // Skip if user is empty or undefined
  if (!user || user.trim() === '') {
    console.log('getUserConversations: user is empty, skipping fetch');
    return [];
  }

  try {
    const response = await fetch(`${BACKEND_URL}/conversation/user/${user}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const conversations = (await response.json()) as UserConversation[];
    return conversations;
  } catch (error) {
    console.error('There was a problem with the fetch operation:', error);
    return [];
  }
}

export default getUserConversations;
