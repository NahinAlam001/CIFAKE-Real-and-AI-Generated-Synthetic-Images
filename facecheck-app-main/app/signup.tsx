import { View, StyleSheet, Image, Text, TouchableOpacity, SafeAreaView } from 'react-native';
import { Link, router } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';

export default function SignUpScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      
      <View style={styles.content}>
        {/* Title */}
        <Text style={styles.title}>Sign Up</Text>

        {/* Illustration */}
        <View style={styles.illustrationContainer}>
          <Image 
            source={require('../assets/auth-illustration.png')} 
            style={styles.illustration}
            resizeMode="contain"
          />
        </View>

        {/* Auth Buttons */}
        <View style={styles.buttonContainer}>
          <TouchableOpacity 
            style={[styles.button]}
            onPress={() => console.log('Google signup')}
          >
            <Ionicons name="logo-google" size={20} color="#FFFFFF" style={styles.buttonIcon} />
            <Text style={styles.buttonText}>Sign up with Google</Text>
          </TouchableOpacity>

          <TouchableOpacity 
            style={[styles.button]}
            onPress={() => console.log('Apple signup')}
          >
            <Ionicons name="logo-apple" size={20} color="#FFFFFF" style={styles.buttonIcon} />
            <Text style={styles.buttonText}>Sign up with Apple ID</Text>
          </TouchableOpacity>

          {/* Login Link */}
          <View style={styles.loginContainer}>
            <Text style={styles.loginText}>Already have an account? </Text>
            <TouchableOpacity onPress={() => router.back()}>
              <Text style={styles.loginLink}>Sign In</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8F9FA',
  },
  content: {
    flex: 1,
    padding: 24,
  },
  title: {
    fontSize: 32,
    fontWeight: '600',
    color: '#000',
    marginTop: 40,
    marginBottom: 40,
  },
  illustrationContainer: {
    alignItems: 'center',
    marginBottom: 48,
  },
  illustration: {
    width: '100%',
    height: 240,
  },
  buttonContainer: {
    gap: 16,
  },
  button: {
    flexDirection: 'row',
    padding: 16,
    borderRadius: 100,
    alignItems: 'center',
    justifyContent: 'center',
    height: 56,
    backgroundColor: '#000000',
  },
  buttonIcon: {
    marginRight: 8,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '500',
    letterSpacing: 0.2,
  },
  loginContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 24,
  },
  loginText: {
    color: '#000',
    fontSize: 14,
  },
  loginLink: {
    color: '#000',
    fontSize: 14,
    fontWeight: '600',
    textDecorationLine: 'underline',
  },
}); 