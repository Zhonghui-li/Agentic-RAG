import { Button, TextField, Typography, Box, Link } from '@mui/material';
import type React from 'react';
import { useState, useContext } from 'react';

import { AuthContext } from '@/auth/AuthProvider';
import type { User, Error } from '@/components/types';

type CustomSignInFormProps = {
  onClose: () => void;
  switchToSignUp: () => void;
};

const CustomSignInForm: React.FC<CustomSignInFormProps> = ({ onClose, switchToSignUp }) => {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const authContext = useContext(AuthContext);

  const handleSignIn = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5001/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, password }),
      });
      if (response.ok) {
        const data = (await response.json()) as User;
        if (authContext) {
          authContext.setIsAuthenticated(true);
          authContext.setUser(data['name']);
          authContext.setUserId(userId);
        }
        onClose();
      } else {
        const errorData = (await response.json()) as Error;
        setError(errorData.error);
      }
    } catch (err) {
      setError('An error occurred while trying to sign in.');
    }
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
      <Typography variant="h4">Sign In</Typography>
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
        onClick={() => void handleSignIn()}
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
        Sign In
      </Button>
      <Button onClick={onClose} style={{ marginTop: '16px' }} variant="outlined">
        Close
      </Button>
      <Typography style={{ marginTop: '16px' }} variant="body2">
        Don&apos;t have an account?{' '}
        <Link href="#" onClick={switchToSignUp} underline="hover">
          Sign up here!
        </Link>
      </Typography>
    </Box>
  );
};

export default CustomSignInForm;
