import { View, StyleSheet, Image, Text, TouchableOpacity, SafeAreaView } from 'react-native';
import { Link, router } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';

export default function LoginScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      
      <View style={styles.content}>
        {/* Title */}
        <Text style={styles.title}>Log In</Text>

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
            onPress={() => {
              console.log('Google login');
              router.push('/home');
            }}
          >
            <Ionicons name="logo-google" size={20} color="#FFFFFF" style={styles.buttonIcon} />
            <Text style={styles.buttonText}>Log in with Google</Text>
          </TouchableOpacity>

          <TouchableOpacity 
            style={[styles.button]}
            onPress={() => {
              console.log('Apple login');
              router.push('/home');
            }}
          >
            <Ionicons name="logo-apple" size={20} color="#FFFFFF" style={styles.buttonIcon} />
            <Text style={styles.buttonText}>Log in with Apple ID</Text>
          </TouchableOpacity>

          {/* Sign Up Link */}
          <View style={styles.signupContainer}>
            <Text style={styles.signupText}>Don't have an account? </Text>
            <TouchableOpacity onPress={() => router.push('/signup')}>
              <Text style={styles.signupLink}>Sign Up</Text>
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
  signupContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 24,
  },
  signupText: {
    color: '#000',
    fontSize: 14,
  },
  signupLink: {
    color: '#000',
    fontSize: 14,
    fontWeight: '600',
    textDecorationLine: 'underline',
  },
}); 