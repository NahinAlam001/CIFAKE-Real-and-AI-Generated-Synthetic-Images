// Types
export type UserProfile = {
  id: string;
  name: string;
  username: string;
  email: string;
  credits: number;
  photoUrl?: string;
  createdAt?: string;
  lastLogin?: string;
};

// Mock profile for UI development
const MOCK_PROFILE: UserProfile = {
  id: 'user-123',
  name: 'Dean Gomez',
  username: 'deangomez',
  email: 'deangomez@gmail.com',
  credits: 69,
  photoUrl: 'https://randomuser.me/api/portraits/men/32.jpg',
  createdAt: '2023-01-15T10:30:00Z',
  lastLogin: '2023-03-01T08:45:00Z'
};

// Get user profile
export const getUserProfile = async (): Promise<UserProfile> => {
  // This will be replaced with an actual API call
  return new Promise((resolve) => {
    // Simulate network delay
    setTimeout(() => {
      resolve(MOCK_PROFILE);
    }, 1000);
  });
};

// Update user profile
export const updateUserProfile = async (
  updates: Partial<UserProfile>
): Promise<UserProfile> => {
  // This will be replaced with an actual API call
  console.log('Updating profile with:', updates);
  
  return new Promise((resolve) => {
    // Simulate network delay
    setTimeout(() => {
      // Merge updates with existing profile
      const updatedProfile = { ...MOCK_PROFILE, ...updates };
      resolve(updatedProfile);
    }, 1000);
  });
};

// Upload profile picture
export const uploadProfilePicture = async (
  imageUri: string
): Promise<{ photoUrl: string }> => {
  // This will be replaced with actual image upload
  console.log('Uploading profile picture:', imageUri);
  
  return new Promise((resolve) => {
    // Simulate network delay and upload
    setTimeout(() => {
      // Just return the same URL for now (in real implementation, this would be a server URL)
      resolve({ photoUrl: imageUri });
    }, 2000);
  });
};

// Delete user account
export const deleteUserAccount = async (): Promise<boolean> => {
  // This will be replaced with an actual API call
  console.log('Deleting user account');
  
  return new Promise((resolve) => {
    // Simulate network delay
    setTimeout(() => {
      resolve(true);
    }, 1000);
  });
}; 