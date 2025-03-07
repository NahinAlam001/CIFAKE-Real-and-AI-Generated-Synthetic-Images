import { useState, useCallback } from 'react';
import * as imagesService from '../services/images';
import { ImageAnalysisResult, UploadImageParams } from '../services/images';
import { useCredits } from './useCredits';

// Cost in credits for image analysis
const IMAGE_ANALYSIS_COST = 1;

export function useImageAnalysis() {
  const [results, setResults] = useState<ImageAnalysisResult[]>([]);
  const [currentResult, setCurrentResult] = useState<ImageAnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Use the credits hook to manage credits
  const { userCredits, useCreditsForService } = useCredits();

  // Upload and analyze an image
  const analyzeImage = useCallback(async (image: UploadImageParams) => {
    try {
      setIsLoading(true);
      setError(null);
      setCurrentResult(null);
      
      // Check if user has enough credits
      if (userCredits < IMAGE_ANALYSIS_COST) {
        setError('Not enough credits. Please purchase more credits to continue.');
        return null;
      }
      
      // Upload the image
      const { imageId } = await imagesService.uploadImage(image);
      
      // Use credits for the analysis
      const creditUsed = await useCreditsForService(IMAGE_ANALYSIS_COST);
      
      if (!creditUsed) {
        setError('Failed to use credits for analysis');
        return null;
      }
      
      // Analyze the image
      const result = await imagesService.analyzeImage(imageId);
      
      // Update state
      setCurrentResult(result);
      setResults(prev => [result, ...prev]);
      
      return result;
    } catch (err) {
      console.error('Error analyzing image:', err);
      setError('Failed to analyze image');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [userCredits, useCreditsForService]);

  // Get image analysis history
  const getImageHistory = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const history = await imagesService.getImageHistory();
      setResults(history);
      
      return history;
    } catch (err) {
      console.error('Error fetching image history:', err);
      setError('Failed to load image history');
      return [];
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Delete an image analysis
  const deleteAnalysis = useCallback(async (analysisId: string) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const success = await imagesService.deleteImageAnalysis(analysisId);
      
      if (success) {
        // Remove from state
        setResults(prev => prev.filter(result => result.id !== analysisId));
        
        // Clear current result if it's the one being deleted
        if (currentResult && currentResult.id === analysisId) {
          setCurrentResult(null);
        }
      }
      
      return success;
    } catch (err) {
      console.error('Error deleting analysis:', err);
      setError('Failed to delete analysis');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [currentResult]);

  return {
    results,
    currentResult,
    isLoading,
    error,
    analyzeImage,
    getImageHistory,
    deleteAnalysis,
  };
} 