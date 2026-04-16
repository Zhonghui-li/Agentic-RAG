import { Button } from '@mui/material';
import { useContext } from 'react';

import { AuthContext } from '@/auth/AuthProvider';

const CustomSignOutButton = () => {
  const authContext = useContext(AuthContext);

  const handleSignOut = () => {
    if (authContext) {
      authContext.setIsAuthenticated(false);
      authContext.setUser(null);
      authContext.setUserId(null);
    }
  };

  return (
    <Button
      onClick={handleSignOut}
      style={{ backgroundColor: '#007bff', color: '#fff' }}
      variant="contained"
    >
      Sign Out
    </Button>
  );
};

export default CustomSignOutButton;
