import { BACKEND_URL } from '@/lib/apiConfig';
async function deleteOneConversation(conv_id: string) {
  try {
    const response = await fetch(`${BACKEND_URL}/conversation/${conv_id}`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return true;
  } catch (error) {
    console.error('There was a problem with the fetch operation:', error);
    return false;
  }
}

export default deleteOneConversation;
