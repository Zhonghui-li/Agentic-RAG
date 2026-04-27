import { Box, Typography, Modal, TextField, Button, Divider } from '@mui/material';
import { X } from 'lucide-react';
import { useState, useContext } from 'react';

import { AuthContext } from '@/auth/AuthProvider';

const modalStyle = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 500,
  bgcolor: 'background.paper',
  border: '2px solid #000',
  boxShadow: 24,
  p: 4,
};

type MyAccountPopupProps = {
  open: boolean;
  handleClose: () => void;
};

const MyAccountPopup: React.FC<MyAccountPopupProps> = ({ open, handleClose }) => {
  const authContext = useContext(AuthContext);
  const [newUsername, setNewUsername] = useState(authContext?.user || '');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [usernameError, setUsernameError] = useState('');
  const [passwordError, setPasswordError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const handleUsernameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNewUsername(e.target.value);
    setUsernameError('');
  };

  const handleCurrentPasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentPassword(e.target.value);
    setPasswordError('');
  };

  const handleNewPasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNewPassword(e.target.value);
    setPasswordError('');
  };

  const handleConfirmPasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setConfirmPassword(e.target.value);
    setPasswordError('');
  };

  const handleSaveUsername = async () => {
    if (!newUsername.trim()) {
      setUsernameError('Username cannot be empty');
      return;
    }

    try {
      const response = await fetch('http://127.0.0.1:5001/api/user/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user: authContext?.user, newUsername }), // Removed userId
      });

      if (response.ok) {
        authContext?.setUser(newUsername);
        setSuccessMessage('Username updated successfully');
        setTimeout(() => setSuccessMessage(''), 3000);
      } else {
        setUsernameError('Failed to update username');
      }
    } catch (error: unknown) {
      setUsernameError('An error occurred while updating username');
    }
  };

  const handleSavePassword = async () => {
    setPasswordError('');

    if (!currentPassword) {
      setPasswordError('Current password is required');
      return;
    }
    if (!newPassword) {
      setPasswordError('New password is required');
      return;
    }
    if (newPassword !== confirmPassword) {
      setPasswordError('New passwords do not match');
      return;
    }
    if (newPassword.length < 8) {
      setPasswordError('Password must be at least 8 characters long');
      return;
    }

    try {
      const response = await fetch('http://127.0.0.1:5001/api/user/change-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user: authContext?.user,
          currentPassword,
          newPassword,
        }),
      });

      if (response.ok) {
        setCurrentPassword('');
        setNewPassword('');
        setConfirmPassword('');
        setSuccessMessage('Password updated successfully');
        setTimeout(() => setSuccessMessage(''), 3000);
      } else {
        const data: unknown = await response.json();
        if (typeof data === 'object' && data !== null && 'message' in data) {
          setPasswordError((data as { message: string }).message);
        } else {
          setPasswordError('Failed to update password');
        }
      }
    } catch (error: unknown) {
      setPasswordError('An error occurred while updating password');
    }
  };

  return (
    <Modal aria-labelledby="my-account-title" onClose={handleClose} open={open}>
      <Box sx={modalStyle}>
        <Box alignItems="center" display="flex" justifyContent="space-between">
          <Typography id="my-account-title" variant="h6">
            My Account
          </Typography>
          <button onClick={handleClose}>
            <X />
          </button>
        </Box>

        {successMessage && (
          <Typography color="success.main" sx={{ mt: 1, mb: 1 }}>
            {successMessage}
          </Typography>
        )}

        <Typography sx={{ mt: 2 }}>Change your username:</Typography>
        <TextField
          fullWidth
          label="New Username"
          onChange={handleUsernameChange}
          sx={{ mt: 2 }}
          value={newUsername}
        />
        {usernameError && (
          <Typography color="error" sx={{ mt: 1 }}>
            {usernameError}
          </Typography>
        )}
        <Box display="flex" justifyContent="flex-end" sx={{ mt: 2 }}>
          <Button onClick={() => void handleSaveUsername()} variant="contained">
            Update Username
          </Button>
        </Box>

        <Divider sx={{ my: 3 }} />

        <Typography sx={{ mt: 2 }}>Change your password:</Typography>
        <TextField
          fullWidth
          label="Current Password"
          onChange={handleCurrentPasswordChange}
          sx={{ mt: 2 }}
          type="password"
          value={currentPassword}
        />
        <TextField
          fullWidth
          label="New Password"
          onChange={handleNewPasswordChange}
          sx={{ mt: 2 }}
          type="password"
          value={newPassword}
        />
        <TextField
          fullWidth
          label="Confirm New Password"
          onChange={handleConfirmPasswordChange}
          sx={{ mt: 2 }}
          type="password"
          value={confirmPassword}
        />
        {passwordError && (
          <Typography color="error" sx={{ mt: 1 }}>
            {passwordError}
          </Typography>
        )}
        <Box display="flex" justifyContent="flex-end" sx={{ mt: 2 }}>
          <Button onClick={() => void handleSavePassword()} variant="contained">
            Update Password
          </Button>
        </Box>

        <Divider sx={{ my: 3 }} />

        <Box display="flex" justifyContent="flex-end">
          <Button onClick={handleClose} variant="outlined">
            Close
          </Button>
        </Box>
      </Box>
    </Modal>
  );
};

export default MyAccountPopup;
