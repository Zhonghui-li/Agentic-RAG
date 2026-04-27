import { Button, TextField, Typography, Box, Link } from '@mui/material';
import type React from 'react';
import { useState, useContext } from 'react';

import { AuthContext } from '@/auth/AuthProvider';
import type { Error } from '@/components/types';

type CustomSignUpFormProps = {
  onClose: () => void;
  switchToSignIn: () => void;
};

const CustomSignUpForm: React.FC<CustomSignUpFormProps> = ({ onClose, switchToSignIn }) => {
  const [name, setName] = useState('');
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const authContext = useContext(AuthContext);

  const handleSignUp = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5001/user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, user_id: userId, password }),
      });
      if (response.ok) {
        // const data = await response.json();
        if (authContext) {
          authContext.setIsAuthenticated(true);
          authContext.setUser(name);
          authContext.setUserId(userId);
        }
        onClose();
      } else {
        const errorData = (await response.json()) as Error;
        setError(errorData.error);
      }
    } catch (err) {
      setError('An error occurred while trying to sign up.');
    }
  };

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setName(e.target.value);
  };

  const handleUserIdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUserId(e.target.value);
  };

  const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPassword(e.target.value);
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#fff',
        borderRadius: '8px',
        padding: '32px',
        maxWidth: '400px',
        width: '100%',
      }}
    >
      <Typography variant="h4">Sign Up</Typography>
      <TextField
        fullWidth
        label="Name"
        onChange={handleNameChange}
        sx={{
          margin: '8px 0',
        }}
        value={name}
        variant="outlined"
      />
      <TextField
        fullWidth
        label="User ID"
        onChange={handleUserIdChange}
        sx={{
          margin: '8px 0',
        }}
        value={userId}
        variant="outlined"
      />
      <TextField
        fullWidth
        label="Password"
        onChange={handlePasswordChange}
        sx={{
          margin: '8px 0',
        }}
        type="password"
        value={password}
        variant="outlined"
      />
      {error && <Typography color="error">{error}</Typography>}
      <Button
        fullWidth
        onClick={() => void handleSignUp()}
        sx={{
          backgroundColor: '#007bff',
          color: '#fff',
          marginTop: '16px',
          '&:hover': {
            backgroundColor: '#0056b3',
          },
        }}
        variant="contained"
      >
        Sign Up
      </Button>
      <Button onClick={onClose} style={{ marginTop: '16px' }} variant="outlined">
        Close
      </Button>
      <Typography style={{ marginTop: '16px' }} variant="body2">
        Already have an account?{' '}
        <Link href="#" onClick={switchToSignIn} underline="hover">
          Sign in here!
        </Link>
      </Typography>
    </Box>
  );
};

export default CustomSignUpForm;
