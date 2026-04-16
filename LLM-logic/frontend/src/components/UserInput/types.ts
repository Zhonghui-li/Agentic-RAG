export type Result = Ok | Err;

export type Ok = {
  data: string;
  type: 'success';
};

export type Conv_Confirm = {
  id: string;
};

export type Err = {
  data: string;
  type: 'err';
};

export type Message = {
  role: 'text' | 'user' | 'bot';
  content: string;
};

export type ChatMessageBoxProps = {
  messages: Message[];
};

export type ModalInput = {
  open: boolean;
  handleClose: () => void;
};

export type Conversation = {
  _id: string;
  last_modified: string;
  messages: Message[];
  name: string;
  user: string;
};
