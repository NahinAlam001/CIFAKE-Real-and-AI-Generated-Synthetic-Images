import { useState, useEffect, useCallback } from 'react';
import * as authService from '../services/auth';

export type AuthUser = {
  id: string;
  email: string;
  name?: string;
  photoUrl?: string;
};

export function useAuth() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Check authentication status on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        const isAuthenticated = await authService.isAuthenticated();
        
        if (isAuthenticated) {
          const userInfo = await authService.getUserInfo();
          if (userInfo) {
            setUser({
              id: userInfo.id,
              email: userInfo.email,
              name: userInfo.name,
              photoUrl: userInfo.photoUrl,
            });
          }
        }
      } catch (err) {
        console.error('Auth check error:', err);
        setError('Failed to check authentication status');
      } finally {
        setIsLoading(false);
      }
    };
    
    checkAuth();
  }, []);

  // Login with Google
  const loginWithGoogle = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const { user: googleUser } = await authService.loginWithGoogle();
      
      setUser({
        id: googleUser.id,
        email: googleUser.email,
        name: googleUser.name,
        photoUrl: googleUser.photoUrl,
      });
      
      return true;
    } catch (err) {
      console.error('Google login error:', err);
      setError('Failed to login with Google');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Login with Apple
  const loginWithApple = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const { user: appleUser } = await authService.loginWithApple();
      
      setUser({
        id: appleUser.id,
        email: appleUser.email,
        name: appleUser.name,
        photoUrl: appleUser.photoUrl,
      });
      
      return true;
    } catch (err) {
      console.error('Apple login error:', err);
      setError('Failed to login with Apple');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Logout
  const logout = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      await authService.logout();
      setUser(null);
      
      return true;
    } catch (err) {
      console.error('Logout error:', err);
      setError('Failed to logout');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    user,
    isAuthenticated: !!user,
    isLoading,
    error,
    loginWithGoogle,
    loginWithApple,
    logout,
  };
} 