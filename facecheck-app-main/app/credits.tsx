import { View, StyleSheet, Text, TouchableOpacity, SafeAreaView } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { useState } from 'react';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';

// This type definition will match our backend schema
type CreditPackage = {
  id: string;
  credits: number;
  price: number;
  isBestValue?: boolean;
};

export default function CreditsScreen() {
  // This will later be fetched from the backend
  const [remainingCredits, setRemainingCredits] = useState(19);
  
  // This will later be fetched from the backend
  const creditPackages: CreditPackage[] = [
    {
      id: '1',
      credits: 50,
      price: 9.00,
      isBestValue: true,
    },
    {
      id: '2',
      credits: 10,
      price: 4.99,
    },
    {
      id: '3',
      credits: 20,
      price: 9.00,
    },
  ];

  const [selectedPackage, setSelectedPackage] = useState<string | null>(null);

  const handleBuyNow = async (packageId: string) => {
    const selectedPkg = creditPackages.find(pkg => pkg.id === packageId);
    if (selectedPkg) {
      router.push({
        pathname: '/checkout',
        params: {
          credits: selectedPkg.credits.toString(),
          price: selectedPkg.price.toString(),
          isBestValue: selectedPkg.isBestValue ? 'true' : 'false'
        }
      });
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Buy Credit</Text>
      </View>

      {/* Credits Info Card */}
      <View style={styles.creditInfoCard}>
        <View style={styles.coinIconContainer}>
          <Ionicons name="logo-usd" size={32} color="#F59E0B" />
        </View>
        <View style={styles.creditInfo}>
          <Text style={styles.remainingCreditsLabel}>Remaining Credits:</Text>
          <Text style={styles.remainingCredits}>{remainingCredits}</Text>
          <Text style={styles.creditNote}>1 Credit = 1 Face Detection</Text>
        </View>
      </View>

      {/* Packages Section */}
      <View style={styles.packagesContainer}>
        <Text style={styles.sectionTitle}>Buy Credits</Text>
        
        {creditPackages.map((pkg) => (
          <TouchableOpacity
            key={pkg.id}
            style={[
              styles.packageCard,
              selectedPackage === pkg.id && styles.selectedPackage,
            ]}
            onPress={() => setSelectedPackage(pkg.id)}
          >
            {pkg.isBestValue && (
              <View style={styles.bestValueBadge}>
                <Text style={styles.bestValueText}>★ Best value</Text>
              </View>
            )}
            
            <View style={styles.packageInfo}>
              <Text style={[
                styles.creditAmount,
                selectedPackage === pkg.id && styles.selectedText
              ]}>{pkg.credits}</Text>
              <Text style={[
                styles.creditLabel,
                selectedPackage === pkg.id && styles.selectedText
              ]}>Credits</Text>
              <Text style={[
                styles.price,
                selectedPackage === pkg.id && styles.selectedText
              ]}>${pkg.price.toFixed(2)}</Text>
            </View>

            <TouchableOpacity
              style={[
                styles.buyButton,
                selectedPackage === pkg.id && styles.selectedBuyButton
              ]}
              onPress={() => handleBuyNow(pkg.id)}
            >
              <Text style={[
                styles.buyButtonText,
                selectedPackage === pkg.id && styles.selectedBuyButtonText
              ]}>Buy Now</Text>
            </TouchableOpacity>
          </TouchableOpacity>
        ))}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8F9FA',
  },
  header: {
    padding: 24,
  },
  title: {
    fontSize: 32,
    fontWeight: '600',
    color: '#000',
  },
  creditInfoCard: {
    flexDirection: 'row',
    backgroundColor: '#EEF2FF',
    marginHorizontal: 24,
    padding: 16,
    borderRadius: 16,
    alignItems: 'center',
  },
  coinIconContainer: {
    backgroundColor: '#FEF3C7',
    padding: 12,
    borderRadius: 12,
    marginRight: 16,
  },
  creditInfo: {
    flex: 1,
  },
  remainingCreditsLabel: {
    fontSize: 14,
    color: '#4F46E5',
    marginBottom: 4,
  },
  remainingCredits: {
    fontSize: 32,
    fontWeight: '700',
    color: '#000',
  },
  creditNote: {
    fontSize: 14,
    color: '#6B7280',
  },
  packagesContainer: {
    flex: 1,
    padding: 24,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '600',
    color: '#000',
    marginBottom: 16,
  },
  packageCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 24,
    marginBottom: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  selectedPackage: {
    backgroundColor: '#4F46E5',
  },
  bestValueBadge: {
    position: 'absolute',
    top: -12,
    left: 24,
    backgroundColor: '#F59E0B',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 100,
  },
  bestValueText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
  },
  packageInfo: {
    flex: 1,
  },
  creditAmount: {
    fontSize: 32,
    fontWeight: '700',
    color: '#000',
  },
  creditLabel: {
    fontSize: 16,
    color: '#6B7280',
  },
  price: {
    fontSize: 32,
    fontWeight: '700',
    color: '#000',
    marginTop: 8,
  },
  buyButton: {
    backgroundColor: '#F3F4F6',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 100,
  },
  selectedBuyButton: {
    backgroundColor: '#FFFFFF',
  },
  buyButtonText: {
    color: '#1E40AF',
    fontSize: 16,
    fontWeight: '600',
  },
  selectedBuyButtonText: {
    color: '#1E40AF',
  },
  selectedText: {
    color: '#FFFFFF',
  },
}); 