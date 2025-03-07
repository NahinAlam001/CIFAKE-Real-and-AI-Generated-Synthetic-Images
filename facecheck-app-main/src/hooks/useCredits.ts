import { useState, useEffect, useCallback } from 'react';
import * as creditsService from '../services/credits';
import { CreditPackage, PurchaseResult } from '../services/credits';

export function useCredits() {
  const [creditPackages, setCreditPackages] = useState<CreditPackage[]>([]);
  const [userCredits, setUserCredits] = useState<number>(0);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch credit packages and user credits on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        // Fetch packages and user credits in parallel
        const [packages, credits] = await Promise.all([
          creditsService.getCreditPackages(),
          creditsService.getUserCredits(),
        ]);
        
        setCreditPackages(packages);
        setUserCredits(credits);
      } catch (err) {
        console.error('Error fetching credits data:', err);
        setError('Failed to load credits information');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, []);

  // Purchase credits
  const purchaseCredits = useCallback(async (
    packageId: string,
    paymentMethod: 'apple_pay' | 'google_pay' | 'credit_card'
  ): Promise<PurchaseResult> => {
    try {
      setIsLoading(true);
      setError(null);
      
      const result = await creditsService.purchaseCredits(packageId, paymentMethod);
      
      if (result.success && result.totalCredits) {
        setUserCredits(result.totalCredits);
      }
      
      return result;
    } catch (err) {
      console.error('Error purchasing credits:', err);
      setError('Failed to purchase credits');
      return {
        success: false,
        error: 'Failed to purchase credits',
      };
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Use credits for a service
  const useCreditsForService = useCallback(async (amount: number): Promise<boolean> => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Check if user has enough credits
      if (userCredits < amount) {
        setError('Not enough credits');
        return false;
      }
      
      const success = await creditsService.useCredits(amount);
      
      if (success) {
        // Update local state
        setUserCredits(prev => prev - amount);
      }
      
      return success;
    } catch (err) {
      console.error('Error using credits:', err);
      setError('Failed to use credits');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [userCredits]);

  // Refresh user credits
  const refreshCredits = useCallback(async (): Promise<void> => {
    try {
      setIsLoading(true);
      setError(null);
      
      const credits = await creditsService.getUserCredits();
      setUserCredits(credits);
    } catch (err) {
      console.error('Error refreshing credits:', err);
      setError('Failed to refresh credits');
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    creditPackages,
    userCredits,
    isLoading,
    error,
    purchaseCredits,
    useCreditsForService,
    refreshCredits,
  };
} 