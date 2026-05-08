import { Box, Divider, Typography } from '@mui/material';
import { CirclePlus } from 'lucide-react';
import * as React from 'react';

import { AppContext } from '@/pages/AppContext';

import { Popup } from './Popup';
import type { Message } from './types';

export function MessageView(props: Message) {
  const [open, setOpen] = React.useState(false);
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);
  const { isDark } = React.useContext(AppContext);

  if (props.role === 'user') {
    // If Message is from User
    return (
      <Typography
        sx={{
          alignItems: 'flex-start',
          backgroundColor: isDark ? '#2b2d42' : '#c0d6df',
          borderRadius: '15px',
          boxShadow: '0 2px 5px rgba(0, 0, 0, 0.1)',
          color: isDark ? '#ffffff' : '#000000',
          display: 'flex',
          flexDirection: 'column',
          padding: 1.2,
        }}
      >
        {props.content}
      </Typography>
    );
  } else {
    // If message is from LLM
    return (
      <Box
        sx={{
          alignItems: 'flex-start',
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-between',
        }}
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, paddingRight: 1 }}>
          <Typography
            sx={{
              color: props.content === '' ? (isDark ? '#a1a1b7' : '#8d99ae') : (isDark ? '#ffffff' : '#000000'),
              fontStyle: props.content === '' ? 'italic' : 'normal',
            }}
            variant="body1"
          >
            {props.content === '' ? 'Thinking...' : props.content}
          </Typography>
          {props.sources && props.sources.length > 0 && (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
              {props.sources.map((src) => (
                <Box
                  key={src}
                  sx={{
                    fontSize: '0.7rem',
                    color: isDark ? '#a1a1b7' : '#5a6a7e',
                    backgroundColor: isDark ? '#3a3a52' : '#e8edf2',
                    borderRadius: '4px',
                    padding: '2px 6px',
                  }}
                >
                  📄 {src}
                </Box>
              ))}
            </Box>
          )}
        </Box>
        <Divider
          flexItem
          orientation="vertical"
          sx={{
            backgroundColor: isDark ? '#585871' : '#dddcdc',
          }}
          variant="middle"
        />
        <Box
          sx={{
            borderRadius: '10px',
            padding: 0.5,
            '&:hover': {
              backgroundColor: isDark ? '#585871' : '#dddcdc',
            },
          }}
        >
          <CirclePlus
            onClick={handleOpen}
            style={{
              color: isDark ? '#a1a1b7' : '#8d99ae',
              height: 18,
              width: 18,
            }}
          />
          <Popup handleClose={handleClose} open={open} />
        </Box>
      </Box>
    );
  }
}
