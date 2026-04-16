export type User = {
  _id: string;
  name: string;
  password: string;
  user_id: string;
};

export type Error = {
  error: string;
};

export type UserConversation = {
  _id: string;
  name: string;
  user: string;
  last_modified: string;
};
