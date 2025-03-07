// Types
export type ImageAnalysisResult = {
  id: string;
  originalImageUrl: string;
  processedImageUrl?: string;
  detectedFaces: number;
  confidence: number;
  isAI: boolean;
  metadata: {
    width: number;
    height: number;
    format: string;
    createdAt: string;
  };
  tags?: string[];
};

export type UploadImageParams = {
  uri: string;
  type?: string;
  filename?: string;
};

// Mock analysis result for UI development
const MOCK_ANALYSIS_RESULT: ImageAnalysisResult = {
  id: 'analysis-123',
  originalImageUrl: 'https://example.com/original.jpg',
  processedImageUrl: 'https://example.com/processed.jpg',
  detectedFaces: 1,
  confidence: 0.92,
  isAI: true,
  metadata: {
    width: 1200,
    height: 800,
    format: 'jpeg',
    createdAt: new Date().toISOString(),
  },
  tags: ['generated', 'portrait', 'female'],
};

// Upload image
export const uploadImage = async (
  image: UploadImageParams
): Promise<{ imageId: string; uploadUrl: string }> => {
  // This will be replaced with actual image upload
  console.log('Uploading image:', image.uri);
  
  return new Promise((resolve) => {
    // Simulate network delay
    setTimeout(() => {
      resolve({
        imageId: `image-${Date.now()}`,
        uploadUrl: image.uri, // In real implementation, this would be a server URL
      });
    }, 1500);
  });
};

// Analyze image
export const analyzeImage = async (
  imageId: string
): Promise<ImageAnalysisResult> => {
  // This will be replaced with actual image analysis
  console.log('Analyzing image:', imageId);
  
  return new Promise((resolve) => {
    // Simulate processing delay
    setTimeout(() => {
      resolve({
        ...MOCK_ANALYSIS_RESULT,
        id: imageId,
      });
    }, 3000);
  });
};

// Get user's image history
export const getImageHistory = async (): Promise<ImageAnalysisResult[]> => {
  // This will be replaced with an actual API call
  return new Promise((resolve) => {
    // Simulate network delay
    setTimeout(() => {
      // Generate a few mock results
      const mockResults = Array(5).fill(null).map((_, index) => ({
        ...MOCK_ANALYSIS_RESULT,
        id: `analysis-${index + 1}`,
        confidence: Math.random() * 0.5 + 0.5, // Random confidence between 0.5 and 1.0
        detectedFaces: Math.floor(Math.random() * 3) + 1, // 1-3 faces
        isAI: Math.random() > 0.5, // Randomly true or false
        metadata: {
          ...MOCK_ANALYSIS_RESULT.metadata,
          createdAt: new Date(Date.now() - index * 86400000).toISOString(), // Different dates
        },
      }));
      
      resolve(mockResults);
    }, 1000);
  });
};

// Delete an image analysis
export const deleteImageAnalysis = async (analysisId: string): Promise<boolean> => {
  // This will be replaced with an actual API call
  console.log('Deleting image analysis:', analysisId);
  
  return new Promise((resolve) => {
    // Simulate network delay
    setTimeout(() => {
      resolve(true);
    }, 500);
  });
}; 