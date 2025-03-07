import { View, StyleSheet, Text, TouchableOpacity, SafeAreaView, Image } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { useLocalSearchParams, router } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

type SuccessParams = {
  credits: string;
  totalCredits: string;
};

export default function PaymentSuccessScreen() {
  const params = useLocalSearchParams<SuccessParams>();
  const creditsAdded = parseInt(params.credits || '0');
  const totalCredits = parseInt(params.totalCredits || '0');

  const handleContinue = () => {
    router.replace('/(tabs)/index');
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      
      <View style={styles.header}>
        <Text style={styles.title}>Profile</Text>
      </View>

      <View style={styles.content}>
        <View style={styles.illustrationContainer}>
          <Image 
            source={require('../assets/payment-success.png')} 
            style={styles.illustration}
            resizeMode="contain"
          />
        </View>

        <View style={styles.creditInfo}>
          <Text style={styles.successText}>Credits added successfully!</Text>
          <View style={styles.creditCard}>
            <View style={styles.coinContainer}>
              <Ionicons name="logo-usd" size={32} color="#F59E0B" />
            </View>
            <View style={styles.creditDetails}>
              <Text style={styles.creditLabel}>Total Credits</Text>
              <Text style={styles.creditAmount}>{totalCredits}</Text>
            </View>
          </View>
        </View>

        <TouchableOpacity 
          style={styles.continueButton}
          onPress={handleContinue}
        >
          <Text style={styles.continueButtonText}>Continue</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  header: {
    padding: 24,
  },
  title: {
    fontSize: 32,
    fontWeight: '600',
    color: '#000',
  },
  content: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  illustrationContainer: {
    width: '100%',
    aspectRatio: 1.5,
    marginVertical: 24,
  },
  illustration: {
    width: '100%',
    height: '100%',
  },
  creditInfo: {
    width: '100%',
    alignItems: 'center',
  },
  successText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#000',
    marginBottom: 24,
  },
  creditCard: {
    width: '100%',
    backgroundColor: '#EEF2FF',
    borderRadius: 16,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
  },
  coinContainer: {
    backgroundColor: '#FEF3C7',
    padding: 12,
    borderRadius: 12,
    marginRight: 16,
  },
  creditDetails: {
    flex: 1,
  },
  creditLabel: {
    fontSize: 16,
    color: '#6B7280',
  },
  creditAmount: {
    fontSize: 32,
    fontWeight: '700',
    color: '#000',
  },
  continueButton: {
    width: '100%',
    backgroundColor: '#1E40AF',
    padding: 16,
    borderRadius: 100,
    alignItems: 'center',
    position: 'absolute',
    bottom: 48,
  },
  continueButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
}); 