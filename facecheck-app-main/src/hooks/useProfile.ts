import { useState, useEffect, useCallback } from 'react';
import * as profileService from '../services/profile';
import { UserProfile } from '../services/profile';

export function useProfile() {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch user profile on mount
  useEffect(() => {
    fetchProfile();
  }, []);

  // Fetch profile
  const fetchProfile = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const userProfile = await profileService.getUserProfile();
      setProfile(userProfile);
    } catch (err) {
      console.error('Error fetching profile:', err);
      setError('Failed to load user profile');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Update profile
  const updateProfile = useCallback(async (updates: Partial<UserProfile>) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const updatedProfile = await profileService.updateUserProfile(updates);
      setProfile(updatedProfile);
      
      return true;
    } catch (err) {
      console.error('Error updating profile:', err);
      setError('Failed to update profile');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Upload profile picture
  const uploadProfilePicture = useCallback(async (imageUri: string) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const { photoUrl } = await profileService.uploadProfilePicture(imageUri);
      
      // Update profile with new photo URL
      if (profile) {
        const updatedProfile = await profileService.updateUserProfile({
          ...profile,
          photoUrl,
        });
        
        setProfile(updatedProfile);
      }
      
      return photoUrl;
    } catch (err) {
      console.error('Error uploading profile picture:', err);
      setError('Failed to upload profile picture');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [profile]);

  // Delete account
  const deleteAccount = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const success = await profileService.deleteUserAccount();
      
      if (success) {
        setProfile(null);
      }
      
      return success;
    } catch (err) {
      console.error('Error deleting account:', err);
      setError('Failed to delete account');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    profile,
    isLoading,
    error,
    fetchProfile,
    updateProfile,
    uploadProfilePicture,
    deleteAccount,
  };
} 