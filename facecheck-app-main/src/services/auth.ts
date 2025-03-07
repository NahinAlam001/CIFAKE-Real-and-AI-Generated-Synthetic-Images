import { Platform } from 'react-native';
import * as SecureStore from 'expo-secure-store';

// Types
type AuthTokens = {
  accessToken: string;
  refreshToken?: string;
  idToken?: string;
  expiresAt: number;
};

type UserInfo = {
  id: string;
  email: string;
  name?: string;
  username?: string;
  photoUrl?: string;
};

// Token Storage
const TOKEN_STORAGE_KEY = 'auth_tokens';
const USER_STORAGE_KEY = 'user_info';

// Store tokens securely
export const storeTokens = async (tokens: AuthTokens): Promise<void> => {
  try {
    await SecureStore.setItemAsync(TOKEN_STORAGE_KEY, JSON.stringify(tokens));
    return Promise.resolve();
  } catch (error) {
    console.error('Error storing tokens:', error);
    return Promise.reject(error);
  }
};

// Retrieve tokens
export const getTokens = async (): Promise<AuthTokens | null> => {
  try {
    const tokens = await SecureStore.getItemAsync(TOKEN_STORAGE_KEY);
    return tokens ? JSON.parse(tokens) : null;
  } catch (error) {
    console.error('Error retrieving tokens:', error);
    return null;
  }
};

// Store user info
export const storeUserInfo = async (userInfo: UserInfo): Promise<void> => {
  try {
    await SecureStore.setItemAsync(USER_STORAGE_KEY, JSON.stringify(userInfo));
    return Promise.resolve();
  } catch (error) {
    console.error('Error storing user info:', error);
    return Promise.reject(error);
  }
};

// Retrieve user info
export const getUserInfo = async (): Promise<UserInfo | null> => {
  try {
    const userInfo = await SecureStore.getItemAsync(USER_STORAGE_KEY);
    return userInfo ? JSON.parse(userInfo) : null;
  } catch (error) {
    console.error('Error retrieving user info:', error);
    return null;
  }
};

// Clear auth data (logout)
export const clearAuthData = async (): Promise<void> => {
  try {
    await SecureStore.deleteItemAsync(TOKEN_STORAGE_KEY);
    await SecureStore.deleteItemAsync(USER_STORAGE_KEY);
    return Promise.resolve();
  } catch (error) {
    console.error('Error clearing auth data:', error);
    return Promise.reject(error);
  }
};

// Check if tokens are valid
export const isAuthenticated = async (): Promise<boolean> => {
  try {
    const tokens = await getTokens();
    if (!tokens) return false;
    
    // Check if token is expired
    const now = Date.now();
    return tokens.expiresAt > now;
  } catch (error) {
    console.error('Error checking authentication:', error);
    return false;
  }
};

// Placeholder for Google Auth
export const loginWithGoogle = async (): Promise<{ tokens: AuthTokens; user: UserInfo }> => {
  // This will be implemented with actual Google Auth SDK
  console.log('Google login initiated');
  
  // Mock implementation for UI testing
  const mockTokens: AuthTokens = {
    accessToken: 'mock-google-access-token',
    idToken: 'mock-google-id-token',
    expiresAt: Date.now() + 3600000, // 1 hour from now
  };
  
  const mockUser: UserInfo = {
    id: 'google-123456',
    email: 'user@example.com',
    name: 'Test User',
    photoUrl: 'https://example.com/photo.jpg',
  };
  
  // Store the tokens and user info
  await storeTokens(mockTokens);
  await storeUserInfo(mockUser);
  
  return { tokens: mockTokens, user: mockUser };
};

// Placeholder for Apple Auth
export const loginWithApple = async (): Promise<{ tokens: AuthTokens; user: UserInfo }> => {
  // This will be implemented with actual Apple Auth SDK
  console.log('Apple login initiated');
  
  // Mock implementation for UI testing
  const mockTokens: AuthTokens = {
    accessToken: 'mock-apple-access-token',
    idToken: 'mock-apple-id-token',
    expiresAt: Date.now() + 3600000, // 1 hour from now
  };
  
  const mockUser: UserInfo = {
    id: 'apple-123456',
    email: 'user@example.com',
    name: 'Test User',
  };
  
  // Store the tokens and user info
  await storeTokens(mockTokens);
  await storeUserInfo(mockUser);
  
  return { tokens: mockTokens, user: mockUser };
};

// Logout function
export const logout = async (): Promise<void> => {
  return clearAuthData();
}; 