import { Box, MenuItem } from '@mui/material';
import type { SelectChangeEvent } from '@mui/material/Select';
import Select from '@mui/material/Select';
import { Send } from 'lucide-react';
import { useCallback, useState, useContext } from 'react';

import { AuthContext } from '@/auth/AuthProvider';
import { Button } from '@/components/SidebarNav/button';
import { Input } from '@/components/SidebarNav/input';
import { AppContext } from '@/pages/AppContext';

import ChatMessageBox from './ChatMessageBox';
import { useChatHandler } from './useChatHandler';

function UserInput() {
  const { textInput, setTextInput, messages, handleClick, messagesEndRef } = useChatHandler();
  const [selectedOption1, setSelectedOption1] = useState('openai');
  const { isDark } = useContext(AppContext);
  const [selectedOption2, setSelectedOption2] = useState('pro-slm');
  const authContext = useContext(AuthContext);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        void handleClick(selectedOption2, selectedOption1);
      }
    },
    [handleClick, selectedOption2, selectedOption1],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setTextInput(e.target.value);
    },
    [setTextInput],
  );

  const handleSelectChange1 = (event: SelectChangeEvent) => {
    setSelectedOption1(event.target.value);
  };

  const handleSelectChange2 = (event: SelectChangeEvent) => {
    setSelectedOption2(event.target.value);
  };

  return (
    <Box
      className={
        isDark
          ? 'flex-grow p-4 bg-[#3f3f51] flex flex-col'
          : 'flex-grow p-4 bg-[#ECEBEB] flex flex-col'
      }
    >
      <Box className="flex-grow overflow-auto">
        <Box
          className="chat-container"
          sx={{
            height: '100vh',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <Box
            className="message-box"
            sx={{
              flexGrow: 1,
              overflow: 'auto',
            }}
          >
            <ChatMessageBox messages={messages} />
            <div ref={messagesEndRef} />
          </Box>
          <Box
            className="input-container"
            sx={{
              flexShrink: 0,
              backgroundColor: 'transparent',
              padding: '8px',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            {/* First dropdown - LLM Provider */}
            <Select
              displayEmpty
              onChange={handleSelectChange1}
              sx={{
                width: '150px',
                height: '40px',
                marginRight: '8px',
                '.MuiSelect-select': {
                  backgroundColor: isDark ? '#202123' : '#6482AC',
                  color: '#FFFFFF',
                  padding: '10px 14px',
                },
                '.MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(0, 0, 0, 0)',
                },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(0, 0, 0, 0)',
                },
                '&:hover .MuiOutlinedInput-notchedOutline': {
                  borderColor: isDark ? 'rgba(255, 255, 255, 1)' : 'rgba(0, 0, 0, 0.15)',
                  borderWidth: isDark ? '1px' : '2px',
                },
                '.MuiSelect-icon': {
                  color: '#FFFFFF',
                },
              }}
              value={selectedOption1}
            >
              <MenuItem value="openai">OPEN AI</MenuItem>
              <MenuItem value="claude">CLAUDE</MenuItem>
              <MenuItem value="gemini">GEMINI</MenuItem>
            </Select>

            {/* Second dropdown - Method */}
            <Select
              displayEmpty
              onChange={handleSelectChange2}
              sx={{
                width: '150px',
                height: '40px',
                marginRight: '8px',
                '.MuiSelect-select': {
                  backgroundColor: 'white',
                  padding: '10px 14px',
                },
                '.MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(0, 0, 0, 0)',
                },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(0, 0, 0, 0.23)',
                },
                '&:hover .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(0, 0, 0, 0.23)',
                },
              }}
              value={selectedOption2}
            >
              <MenuItem value="rag-agent">RAG Agent</MenuItem>
              <MenuItem value="pro-slm">Pro-SLM</MenuItem>
              <MenuItem value="rag">RAG</MenuItem>
              <MenuItem value="chain-of-thought">Chain of Thought</MenuItem>
              <MenuItem value="standard">Standard</MenuItem>
            </Select>

            <Input
              aria-label="Input for chat messages"
              className="flex-grow"
              disabled={authContext?.isAuthenticated === false}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              placeholder="Say Something"
              style={{ height: '40px' }}
              value={textInput}
            />
            <Button
              aria-label="Send message"
              className="ml-2 custom-white-button"
              disabled={authContext?.isAuthenticated === false}
              onClick={() => void handleClick(selectedOption2, selectedOption1)}
              size="icon"
              style={{ height: '40px', minWidth: '40px', color: isDark ? '#202123' : '#6482AC' }}
              variant="outline"
            >
              <Send />
            </Button>
          </Box>
        </Box>
      </Box>
    </Box>
  );
}

export default UserInput;
