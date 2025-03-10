import {
  View,
  StyleSheet,
  Text,
  SafeAreaView,
  TouchableOpacity,
  Image,
} from "react-native";
import { StatusBar } from "expo-status-bar";
import { Ionicons } from "@expo/vector-icons";

// Mock Data for the page (you can replace it with actual data later)
const profile = {
  name: "Justin Baldoni",
  occupation: "XYZ",
  awards: "XYZ",
  bio: "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
  imageUri: "https://your-image-url.com/image.jpg", // Replace with the actual URL or local image path
  usedCredits: 1,
  remainingCredits: 19,
};

export default function Page() {
  const handleTryAgain = () => {
    console.log("Trying again...");
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      <View style={styles.profileContainer}>
        {/* Profile Image */}
        <Image source={{ uri: profile.imageUri }} style={styles.profileImage} />
        <Text style={styles.name}>{profile.name}</Text>
        <Text style={styles.occupation}>Occupation: {profile.occupation}</Text>
        <Text style={styles.awards}>Awards: {profile.awards}</Text>
      </View>

      <View style={styles.bioContainer}>
        <Text style={styles.bio}>{profile.bio}</Text>
      </View>

      <View style={styles.creditSection}>
        <View style={styles.creditDetails}>
          <Ionicons name="ios-cash" size={32} color="#F59E0B" />
          <View style={styles.creditInfo}>
            <Text style={styles.creditLabel}>
              Used credits: {profile.usedCredits}
            </Text>
            <Text style={styles.creditLabel}>
              Remaining: {profile.remainingCredits}
            </Text>
            <Text style={styles.creditDetail}>1 Credit = 1 Face Detection</Text>
          </View>
        </View>
      </View>

      {/* Try Again Section */}
      <View style={styles.actionContainer}>
        <Text style={styles.errorText}>Identified the wrong person?</Text>
        <TouchableOpacity
          style={styles.tryAgainButton}
          onPress={handleTryAgain}
        >
          <Text style={styles.tryAgainText}>Try Again</Text>
        </TouchableOpacity>
      </View>

      {/* Bottom Navigation */}
      <View style={styles.bottomNav}>
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="card-outline" size={24} color="#6B7280" />
          <Text style={styles.navText}>Buy Credits</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="camera" size={24} color="#6B7280" />
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
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
    backgroundColor: "#F9FAFB",
  },
  profileContainer: {
    alignItems: "center",
    padding: 20,
    backgroundColor: "#FFFFFF",
    borderBottomWidth: 1,
    borderBottomColor: "#E5E7EB",
  },
  profileImage: {
    width: 120,
    height: 120,
    borderRadius: 60,
    marginBottom: 16,
  },
  name: {
    fontSize: 24,
    fontWeight: "600",
    color: "#000",
  },
  occupation: {
    fontSize: 16,
    color: "#6B7280",
  },
  awards: {
    fontSize: 16,
    color: "#6B7280",
    marginBottom: 16,
  },
  bioContainer: {
    padding: 16,
    backgroundColor: "#FFFFFF",
  },
  bio: {
    fontSize: 16,
    color: "#4B5563",
  },
  creditSection: {
    padding: 16,
    backgroundColor: "#FFFFFF",
    marginTop: 24,
    borderBottomWidth: 1,
    borderBottomColor: "#E5E7EB",
  },
  creditDetails: {
    flexDirection: "row",
    alignItems: "center",
  },
  creditInfo: {
    marginLeft: 12,
  },
  creditLabel: {
    fontSize: 16,
    color: "#6B7280",
  },
  creditDetail: {
    fontSize: 14,
    color: "#9CA3AF",
  },
  actionContainer: {
    alignItems: "center",
    padding: 20,
  },
  errorText: {
    fontSize: 16,
    color: "#F59E0B",
    marginBottom: 12,
  },
  tryAgainButton: {
    backgroundColor: "#4F46E5",
    paddingVertical: 12,
    paddingHorizontal: 40,
    borderRadius: 8,
  },
  tryAgainText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "500",
  },
  bottomNav: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 24,
    paddingVertical: 16,
    backgroundColor: "#FFFFFF",
    borderTopWidth: 1,
    borderTopColor: "#E5E7EB",
  },
  navItem: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 8,
  },
  navText: {
    fontSize: 14,
    color: "#6B7280",
    marginLeft: 8,
  },
});
