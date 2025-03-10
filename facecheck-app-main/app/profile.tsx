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
import * as ImagePicker from "expo-image-picker";
import { useRouter } from "expo-router"; // Use the useRouter hook

// Types for user profile data
type UserProfile = {
  name: string;
  username: string;
  email: string;
  credits: number;
};

export default function ProfileScreen() {
  const router = useRouter(); // Access the router using the hook
  const [profile, setProfile] = useState<UserProfile>({
    name: "",
    username: "",
    email: "",
    credits: 0,
  });
  const [isLoading, setIsLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState(null);

  useEffect(() => {
    fetchUserProfile();
  }, []);

  const fetchUserProfile = async () => {
    try {
      // Simulating API call
      const response = await new Promise<UserProfile>((resolve) => {
        setTimeout(() => {
          resolve({
            name: "Dean Gomez",
            username: "deangomez",
            email: "deangomez@gmail.com",
            credits: 19,
          });
        }, 500);
      });

      setProfile(response);
      setIsLoading(false);
    } catch (error) {
      console.error("Failed to fetch profile:", error);
      setIsLoading(false);
    }
  };

  const openCamera = async () => {
    // Request camera permission
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();

    if (permissionResult.granted === false) {
      alert("You've refused to allow this app to access your camera!");
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
      // Here you would typically upload the image to your backend
      console.log("Camera image:", result.assets[0].uri);
    }
  };

  const handleProfile = () => {
    console.log("Profile pressed");
    router.push("/profile"); // Correct usage of router.push()
  };

  const handleCredit = () => {
    console.log("Buy Credits pressed");
    router.push("/credits"); // Correct usage of router.push()
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

      {/* Bottom Navigation */}
      <View style={styles.bottomNav}>
        <TouchableOpacity style={styles.navItem} onPress={handleCredit}>
          <Ionicons name="card-outline" size={24} color="#6B7280" />
          <Text style={styles.navText}>Buy Credits</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.cameraButton} onPress={openCamera}>
          <Ionicons name="camera" size={32} color="white" />
        </TouchableOpacity>

        <TouchableOpacity style={styles.navItem} onPress={handleProfile}>
          <Ionicons name="person-outline" size={24} color="#6B7280" />
          <Text style={styles.navText}>Profile</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F3F4F6",
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  topNav: {
    flexDirection: "row",
    justifyContent: "space-between",
    padding: 16,
    backgroundColor: "#FFFFFF",
    borderBottomWidth: 1,
    borderBottomColor: "#E5E7EB",
    alignItems: "center",
  },
  topNavText: {
    fontSize: 24,
    fontWeight: "600",
    color: "#000",
  },
  content: {
    flex: 1,
    paddingHorizontal: 24,
    paddingVertical: 16,
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
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    color: "#6B7280",
    shadowColor: "#000000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  bottomNav: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingVertical: 16,
    backgroundColor: "#FFFFFF",
    borderTopWidth: 1,
    borderTopColor: "#E5E7EB",
  },
  navItem: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    padding: 12,
    flex: 1,
  },
  cameraButton: {
    backgroundColor: "#4F46E5",
    borderRadius: 50,
    padding: 16,
    marginHorizontal: 8,
  },
  navText: {
    fontSize: 14,
    color: "#6B7280",
    marginLeft: 8,
  },
});
