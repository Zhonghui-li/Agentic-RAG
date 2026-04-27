import { createContext, useState } from 'react';
import type { ReactNode, Dispatch, SetStateAction } from 'react';

type AppState = {
  currConversation: string;
  setCurrConversation: Dispatch<SetStateAction<string>>;
  newConvCount: number;
  setNewConvCount: Dispatch<SetStateAction<number>>;
  isDark: boolean;
  setIsDark: Dispatch<SetStateAction<boolean>>;
};

const defaultState = {
  currConversation: '',
  setCurrConversation: () => undefined,
  newConvCount: 0,
  setNewConvCount: () => undefined,
  isDark: true,
  setIsDark: () => undefined,
} as AppState;

export const AppContext = createContext(defaultState);

type AppProviderProps = {
  children: ReactNode;
};

export default function AppProvider({ children }: AppProviderProps) {
  const [currConversation, setCurrConversation] = useState<string>('');
  const [newConvCount, setNewConvCount] = useState<number>(0);
  const [isDark, setIsDark] = useState<boolean>(true);

  return (
    <AppContext.Provider
      value={{
        currConversation,
        setCurrConversation,
        newConvCount,
        setNewConvCount,
        isDark,
        setIsDark,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}
