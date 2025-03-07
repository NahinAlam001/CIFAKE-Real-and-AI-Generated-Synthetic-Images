import {
  View,
  StyleSheet,
  Text,
  SafeAreaView,
  TextInput,
  TouchableOpacity,
} from "react-native";
import { StatusBar } from "expo-status-bar";
import { useState, useEffect } from "react";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router"; // Import the router for navigation

// Types for user profile data
type UserProfile = {
  name: string;
  username: string;
  email: string;
  credits: number;
};

export default function ProfileScreen() {
  const [profile, setProfile] = useState<UserProfile>({
    name: "",
    username: "",
    email: "",
    credits: 0,
  });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchUserProfile();
  }, []);

  const fetchUserProfile = async () => {
    try {
      // TODO: Replace with actual API call
      // Simulating API call
      const response = await new Promise<UserProfile>((resolve) => {
        setTimeout(() => {
          resolve({
            name: "Dean Gomez",
            username: "deangomez",
            email: "deangomez@gmail.com",
            credits: 69,
          });
        }, 1000);
      });

      setProfile(response);
      setIsLoading(false);
    } catch (error) {
      console.error("Failed to fetch profile:", error);
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar style="dark" />
        <View style={styles.loadingContainer}>
          <Text>Loading...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />

      <View style={styles.content}>
        {/* Profile Avatar */}
        <View style={styles.avatarContainer}>
          <View style={styles.avatar}>
            <Ionicons name="person-outline" size={48} color="#4F46E5" />
          </View>
          <Text style={styles.name}>{profile.name}</Text>
        </View>

        {/* Credit Balance */}
        <View style={styles.creditCard}>
          <View style={styles.coinContainer}>
            <Ionicons name="logo-usd" size={32} color="#F59E0B" />
          </View>
          <View style={styles.creditDetails}>
            <Text style={styles.creditLabel}>Available Credits</Text>
            <Text style={styles.creditAmount}>{profile.credits}</Text>
          </View>
        </View>

        {/* Profile Fields */}
        <View style={styles.formSection}>
          <View style={styles.inputGroup}>
            <Text style={styles.label}>Username</Text>
            <TextInput
              style={styles.input}
              value={profile.username}
              editable={false}
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Email</Text>
            <TextInput
              style={styles.input}
              value={profile.email}
              editable={false}
            />
          </View>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#FFFFFF",
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    padding: 24,
  },
  backButton: {
    marginRight: 16, // Space between the back button and the title
  },
  title: {
    fontSize: 32,
    fontWeight: "600",
    color: "#000",
  },
  content: {
    flex: 1,
    paddingHorizontal: 24,
    paddingTop: 20,
  },
  avatarContainer: {
    alignItems: "center",
    marginBottom: 32,
  },
  avatar: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: "#EEF2FF",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 16,
  },
  name: {
    fontSize: 24,
    fontWeight: "600",
    color: "#000",
  },
  creditCard: {
    backgroundColor: "#EEF2FF",
    borderRadius: 16,
    padding: 16,
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 32,
  },
  coinContainer: {
    backgroundColor: "#FEF3C7",
    padding: 12,
    borderRadius: 12,
    marginRight: 16,
  },
  creditDetails: {
    flex: 1,
  },
  creditLabel: {
    fontSize: 16,
    color: "#6B7280",
  },
  creditAmount: {
    fontSize: 32,
    fontWeight: "700",
    color: "#000",
  },
  formSection: {
    gap: 24,
  },
  inputGroup: {
    gap: 8,
  },
  label: {
    fontSize: 16,
    fontWeight: "500",
    color: "#000",
  },
  input: {
    backgroundColor: "#F3F4F6",
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    color: "#6B7280",
  },
});
