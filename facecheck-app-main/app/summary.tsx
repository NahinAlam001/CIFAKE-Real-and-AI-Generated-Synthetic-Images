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
import { router } from "expo-router";

// Mock Data for the page (you can replace it with actual data later)
const profile = {
  name: "Justin Baldoni",
  occupation: "XYZ",
  awards: "XYZ",
  bio: "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
  imageUri: "", // Leave it empty for testing placeholder
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
      <View style={styles.card}>
        {/* Profile Image or Placeholder */}
        {profile.imageUri ? (
          <Image
            source={{ uri: profile.imageUri }}
            style={styles.profileImage}
          />
        ) : (
          <View style={styles.placeholderImage}>
            <Ionicons name="person-circle-outline" size={100} color="#6B7280" />
          </View>
        )}

        <View style={styles.profileDetails}>
          <Text style={styles.name}>{profile.name}</Text>
          <Text style={styles.occupation}>
            Occupation: {profile.occupation}
          </Text>
          <Text style={styles.awards}>Awards: {profile.awards}</Text>
        </View>

        {/* Bio Section */}
        <Text style={styles.bio}>{profile.bio}</Text>
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
  card: {
    backgroundColor: "#FFFFFF",
    margin: 16,
    borderRadius: 12,
    padding: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 4,
  },
  profileImage: {
    width: 100,
    height: 100,
    borderRadius: 50,
    alignSelf: "center",
    marginBottom: 16,
  },
  placeholderImage: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: "#E5E7EB", // Light gray placeholder background
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 16,
    alignSelf: "center",
  },
  profileDetails: {
    alignItems: "center",
    marginBottom: 16,
  },
  name: {
    fontSize: 22,
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
  },
  bio: {
    fontSize: 14,
    color: "#4B5563",
    textAlign: "center",
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
    marginTop: "auto",
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
