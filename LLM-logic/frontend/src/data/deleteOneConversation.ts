async function deleteOneConversation(conv_id: string) {
  try {
    const response = await fetch(`http://127.0.0.1:5001/conversation/${conv_id}`, {
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
