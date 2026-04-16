import {
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
  List,
  ListItem,
  ListItemButton,
  Typography,
} from '@mui/material';
import { Sidebar } from 'lucide-react';
import { useState, useEffect, useContext, useCallback } from 'react';
import { FaSun, FaMoon, FaUser, FaQuestionCircle } from 'react-icons/fa';
import { FaDeleteLeft } from 'react-icons/fa6';

import { AuthContext } from '@/auth/AuthProvider';
import CustomSignInButton from '@/components/CustomSignInButton';
import CustomSignOutButton from '@/components/CustomSignOutButton';
import FAQPopup from '@/components/FAQPopup';
import MyAccountPopUp from '@/components/MyAccountPopUp';
import StartNewConversation from '@/components/UserInput/ConversationFlask';
import type { UserConversation } from '@/components/types';
import deleteOneConversation from '@/data/deleteOneConversation';
import getOneConversation from '@/data/getOneConversation';
import getUserConversations from '@/data/getUserConversations';
import { AppContext } from '@/pages/AppContext';

const SidebarNav = () => {
  const [isOpen, setIsOpen] = useState(true);
  const [conversations, setConversations] = useState<JSX.Element[]>([]);
  const { currConversation, setCurrConversation, isDark, setIsDark } = useContext(AppContext);
  const authContext = useContext(AuthContext);
  const [faqOpen, setFaqOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleteId, setDeleteId] = useState('');

  const toggleSidebar = () => {
    setIsOpen((prevState) => !prevState);
  };

  const toggleBrightness = () => {
    setIsDark((prevMode) => !prevMode);
  };

  const handleAccountOpen = () => {
    setAccountOpen(true);
  };

  const handleAccountClose = () => {
    setAccountOpen(false);
  };

  const handleFAQOpen = () => {
    setFaqOpen(true);
  };

  const handleFAQClose = () => {
    setFaqOpen(false);
  };

  const changeConversation = useCallback(
    (id: string) => {
      setCurrConversation(id);
    },
    [setCurrConversation],
  );

  const handleDelete = useCallback(
    (id: string) => {
      setDeleteId(id);
      setConfirmDelete(true);
    },
    [setDeleteId, setConfirmDelete],
  );

  const handleDeleteClose = () => {
    setDeleteId('');
    setConfirmDelete(false);
  };

  const deleteConversation = async () => {
    const res: boolean = await deleteOneConversation(deleteId);
    if (res) {
      await fetchData();
    }
    handleDeleteClose();
  };

  const fetchData = useCallback(async () => {
    const data: UserConversation[] = await getUserConversations(authContext?.user || '');
    const arr = data.map((conversation: UserConversation) => (
      <ListItem
        disablePadding
        key={conversation._id}
        sx={{
          marginBottom: '0.5rem',
          color: 'gray.300',
          fontSize: '0.875rem',
          textOverflow: 'ellipsis',
          overflow: 'hidden',
          whiteSpace: 'nowrap',
        }}
      >
        <ListItemButton onClick={() => changeConversation(conversation._id)}>
          {conversation.name}
        </ListItemButton>
        <Button
          onClick={() => handleDelete(conversation._id)}
          sx={{
            color: 'white',
          }}
        >
          <FaDeleteLeft />
        </Button>
      </ListItem>
    ));
    setConversations(arr);
  }, [authContext?.user, changeConversation, handleDelete]);

  useEffect(() => {
    if (authContext?.isAuthenticated && authContext?.user) {
      void (async () => {
        await fetchData();
      })();
    } else {
      setConversations([]);
      setCurrConversation('');
    }
  }, [authContext, fetchData, setConversations, setCurrConversation]);

  const handleNewChat = async () => {
    if (!currConversation) {
      console.log('No active conversation selected.');
      return;
    }
    try {
      const conversationData = await getOneConversation(currConversation);
      if (conversationData.messages && conversationData.messages.length > 0) {
        const data = await StartNewConversation(authContext?.user || '', 'New Chat');
        console.log('Started a new conversation');
        if ('id' in data) {
          setCurrConversation(data.id);
          console.log('ID: ', data.id);
        } else {
          console.error('Failed to start a new conversation:', data.error);
        }
      } else {
        console.log('Current chat has no messages. A new chat will not be created.');
        return; // Exit the function if the current chat has messages
      }
    } catch (error) {
      console.error('Error starting a new conversation:', error);
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        width: isOpen ? '16rem' : '4rem',
        height: '100%',
        backgroundColor: isDark ? '#202123' : '#6482AC',
        color: 'white',
        transition: 'width 0.3s',
      }}
    >
      <Box
        component="aside"
        sx={{
          display: isOpen ? 'flex' : 'none',
          width: '16rem',
          height: '100%',
          backgroundColor: isDark ? '#202123' : '#6482AC',
          color: 'white',
          padding: '1rem',
          flexDirection: 'column',
          position: 'relative', // Add position relative to the container
        }}
      >
        {/* Top section with New Chat button */}
        <Box>
          <Box display="flex" flexDirection="row" gap={2} mb={2}>
            <Button
              onClick={() => {
                void handleNewChat();
              }}
              sx={{
                width: '100%',
                backgroundColor: isDark ? '#202123' : '#6482AC',
                border: '1px solid',
                borderColor: 'gray.700',
                color: 'white',
                paddingY: '0.5rem',
                paddingX: '1rem',
                borderRadius: '9999px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                '&:hover': {
                  backgroundColor: isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
                },
              }}
            >
              <Typography sx={{ display: 'flex', alignItems: 'center' }} variant="body2">
                <Box component="span" mr={2}>
                  +
                </Box>{' '}
                New chat
              </Typography>
            </Button>
            <Button
              onClick={toggleSidebar}
              sx={{
                width: '2rem',
                height: '2rem',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                color: 'white',
              }}
            >
              <Sidebar size={24} />
            </Button>
          </Box>
          <Typography sx={{ color: 'gray.400', marginBottom: '0.5rem' }} variant="body2">
            Previous 7 days
          </Typography>
        </Box>

        {/* Scrollable conversation list */}
        <Box
          sx={{
            overflowY: 'auto',
            flex: '1 1 auto',
            mb: 2,
            // Add some bottom padding to ensure last items are fully visible
            pb: 1,
            // Hide scrollbar in various browsers while maintaining functionality
            '&::-webkit-scrollbar': {
              width: '0.4rem',
            },
            '&::-webkit-scrollbar-thumb': {
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '0.2rem',
            },
            '&::-webkit-scrollbar-track': {
              backgroundColor: 'transparent',
            },
            // Firefox scrollbar
            scrollbarWidth: 'thin',
            scrollbarColor: 'rgba(255, 255, 255, 0.1) transparent',
          }}
        >
          <List>{conversations}</List>
        </Box>

        {/* Fixed footer area */}
        <Box
          sx={{
            width: '100%',
            mt: 'auto',
          }}
        >
          <Divider sx={{ borderColor: 'gray.700', marginBottom: '1rem' }} />
          <List>
            <ListItem disablePadding>
              <ListItemButton
                onClick={toggleBrightness}
                sx={{
                  color: '#fff',
                  fontSize: '14px',
                  fontWeight: 400,
                  display: 'flex',
                  alignItems: 'center',
                  padding: '8px 12px',
                  borderRadius: '0.375rem',
                  '&:hover': {
                    backgroundColor: isDark ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.2)',
                  },
                }}
              >
                {isDark ? (
                  <>
                    <FaSun style={{ marginRight: '0.5rem' }} /> Light Mode
                  </>
                ) : (
                  <>
                    <FaMoon style={{ marginRight: '0.5rem' }} /> Dark Mode
                  </>
                )}
              </ListItemButton>
            </ListItem>
            {authContext?.isAuthenticated && (
              <ListItem disablePadding>
                <ListItemButton
                  onClick={handleAccountOpen}
                  sx={{
                    color: '#fff',
                    fontSize: '14px',
                    fontWeight: 400,
                    display: 'flex',
                    alignItems: 'center',
                    padding: '8px 12px',
                    borderRadius: '0.375rem',
                    '&:hover': {
                      backgroundColor: isDark ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.2)',
                    },
                  }}
                >
                  <FaUser style={{ marginRight: '0.5rem' }} /> My Account
                </ListItemButton>
              </ListItem>
            )}
            <ListItem disablePadding>
              <ListItemButton
                onClick={handleFAQOpen}
                sx={{
                  color: '#fff',
                  fontSize: '14px',
                  fontWeight: 400,
                  display: 'flex',
                  alignItems: 'center',
                  padding: '8px 12px',
                  borderRadius: '0.375rem',
                  '&:hover': {
                    backgroundColor: isDark ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.2)',
                  },
                }}
              >
                <FaQuestionCircle style={{ marginRight: '0.5rem' }} /> Updates & FAQ
              </ListItemButton>
            </ListItem>
          </List>
          <Divider sx={{ borderColor: 'gray.700', marginBottom: '1rem' }} />
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem',
            }}
          >
            {authContext?.isAuthenticated ? (
              <>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    textOverflow: 'ellipsis',
                    overflow: 'hidden',
                    whiteSpace: 'nowrap',
                    paddingBottom: '0.5rem',
                  }}
                >
                  Signed in as {authContext.user}
                </div>
                <CustomSignOutButton />
              </>
            ) : (
              <CustomSignInButton />
            )}
          </div>
        </Box>
      </Box>

      <Box
        onClick={toggleSidebar}
        sx={{
          display: isOpen ? 'none' : 'flex',
          alignItems: 'start',
          paddingTop: '1rem',
          justifyContent: 'center',
          width: '4rem',
          height: '100%',
          cursor: 'pointer',
        }}
      >
        <Sidebar size={24} />
      </Box>

      <MyAccountPopUp handleClose={handleAccountClose} open={accountOpen} />

      <FAQPopup handleClose={handleFAQClose} open={faqOpen} />

      <Dialog onClose={handleDeleteClose} open={confirmDelete}>
        <DialogTitle>Confirm Deletion</DialogTitle>
        <DialogContent>
          Are you sure you want to delete this conversation? This action cannot be undone.
        </DialogContent>
        <DialogActions>
          <Button color="primary" onClick={handleDeleteClose}>
            Cancel
          </Button>
          <Button
            color="error"
            onClick={() => {
              deleteConversation().catch((error) =>
                console.error('Error deleting conversation:', error),
              );
            }}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SidebarNav;
