import { View, StyleSheet, Text, TouchableOpacity, SafeAreaView } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { useLocalSearchParams, router } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

type CheckoutParams = {
  credits: string;
  price: string;
  isBestValue?: string;
};

export default function CheckoutScreen() {
  const params = useLocalSearchParams<CheckoutParams>();
  const credits = parseInt(params.credits || '0');
  const price = parseFloat(params.price || '0');
  const isBestValue = params.isBestValue === 'true';

  const handleApplePay = async () => {
    try {
      // TODO: Implement actual Apple Pay integration
      console.log('Processing Apple Pay payment...');
      
      // Simulate payment processing
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // After successful payment, navigate to success screen
      // Note: In a real implementation, you would get the total credits from your backend
      const currentCredits = 19; // This should come from your backend
      const newTotalCredits = currentCredits + credits;
      
      router.replace({
        pathname: '/payment-success',
        params: {
          credits: credits.toString(),
          totalCredits: newTotalCredits.toString(),
        }
      });
    } catch (error) {
      console.error('Payment failed:', error);
      // TODO: Handle payment failure
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Checkout</Text>
      </View>

      {/* Package Summary */}
      <View style={styles.packageCard}>
        {isBestValue && (
          <View style={styles.bestValueBadge}>
            <Text style={styles.bestValueText}>★ Best value</Text>
          </View>
        )}

        <View style={styles.packageDetails}>
          <View style={styles.leftContent}>
            <Text style={styles.credits}>{credits}</Text>
            <Text style={styles.creditsLabel}>Credits</Text>
            <Text style={styles.price}>${price.toFixed(2)}</Text>
          </View>

          <View style={styles.rightContent}>
            <Text style={styles.dueText}>Due : ${price.toFixed(2)}</Text>
          </View>
        </View>
      </View>

      {/* Payment Section */}
      <View style={styles.paymentSection}>
        <Text style={styles.paymentTitle}>Pay with</Text>
        <TouchableOpacity 
          style={styles.applePayButton}
          onPress={handleApplePay}
        >
          <View style={styles.applePayContent}>
            <Ionicons name="logo-apple" size={24} color="#FFFFFF" />
            <Text style={styles.applePayText}>Pay</Text>
          </View>
        </TouchableOpacity>
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
  packageCard: {
    backgroundColor: '#1E40AF',
    marginHorizontal: 24,
    padding: 24,
    borderRadius: 16,
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
  packageDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  leftContent: {
    flex: 1,
  },
  credits: {
    fontSize: 40,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  creditsLabel: {
    fontSize: 16,
    color: '#FFFFFF',
    opacity: 0.8,
  },
  price: {
    fontSize: 40,
    fontWeight: '700',
    color: '#FFFFFF',
    marginTop: 8,
  },
  rightContent: {
    alignItems: 'flex-end',
  },
  dueText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  paymentSection: {
    backgroundColor: '#EEF2FF',
    margin: 24,
    padding: 24,
    borderRadius: 16,
  },
  paymentTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: '#1F2937',
    marginBottom: 16,
  },
  applePayButton: {
    backgroundColor: '#1E40AF',
    padding: 16,
    borderRadius: 100,
    alignItems: 'center',
  },
  applePayContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  applePayText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
}); 