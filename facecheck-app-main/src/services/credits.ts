// Types
export type CreditPackage = {
  id: string;
  credits: number;
  price: number;
  isBestValue?: boolean;
};

export type PurchaseResult = {
  success: boolean;
  transactionId?: string;
  credits?: number;
  totalCredits?: number;
  error?: string;
};

// Mock data for UI development
const MOCK_CREDIT_PACKAGES: CreditPackage[] = [
  {
    id: "1",
    credits: 50,
    price: 9.0,
    isBestValue: true,
  },
  {
    id: "2",
    credits: 10,
    price: 4.99,
  },
  {
    id: "3",
    credits: 20,
    price: 9.0,
  },
];

// Get available credit packages
export const getCreditPackages = async (): Promise<CreditPackage[]> => {
  // This will be replaced with an actual API call
  return new Promise((resolve) => {
    // Simulate network delay
    setTimeout(() => {
      resolve(MOCK_CREDIT_PACKAGES);
    }, 500);
  });
};

// Get user's current credit balance
export const getUserCredits = async (): Promise<number> => {
  // This will be replaced with an actual API call
  return new Promise((resolve) => {
    // Simulate network delay
    setTimeout(() => {
      resolve(19); // Mock credit amount
    }, 500);
  });
};

// Purchase credits
export const purchaseCredits = async (
  packageId: string,
  paymentMethod: "apple_pay" | "google_pay" | "credit_card"
): Promise<PurchaseResult> => {
  // This will be replaced with actual payment processing
  console.log(`Processing ${paymentMethod} payment for package ${packageId}`);

  try {
    // Simulate payment processing
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Find the package
    const selectedPackage = MOCK_CREDIT_PACKAGES.find(
      (pkg) => pkg.id === packageId
    );
    if (!selectedPackage) {
      return {
        success: false,
        error: "Package not found",
      };
    }

    // Mock successful purchase
    const currentCredits = await getUserCredits();
    const newTotalCredits = currentCredits + selectedPackage.credits;

    return {
      success: true,
      transactionId: `mock-transaction-${Date.now()}`,
      credits: selectedPackage.credits,
      totalCredits: newTotalCredits,
    };
  } catch (error) {
    console.error("Payment failed:", error);
    return {
      success: false,
      error: "Payment processing failed",
    };
  }
};

// Use credits for a service
export const useCredits = async (amount: number): Promise<boolean> => {
  // This will be replaced with an actual API call
  console.log(`Using ${amount} credits`);

  try {
    const currentCredits = await getUserCredits();

    if (currentCredits < amount) {
      console.error("Not enough credits");
      return false;
    }

    // Simulate API call to deduct credits
    await new Promise((resolve) => setTimeout(resolve, 500));

    return true;
  } catch (error) {
    console.error("Failed to use credits:", error);
    return false;
  }
};
