import {
  View,
  StyleSheet,
  Text,
  TouchableOpacity,
  SafeAreaView,
  Modal,
  Image,
} from "react-native";
import { Link, router } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { Ionicons } from "@expo/vector-icons";
import { useState } from "react";
import * as ImagePicker from "expo-image-picker";

export default function HomeScreen() {
  const [showMenu, setShowMenu] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);

  const handleLogout = () => {
    setShowMenu(false);
    router.replace("/");
  };

  const handleProfile = () => {
    setShowMenu(false);
    router.push("/profile");
    // Navigate to profile screen
    console.log("Navigate to profile");
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

  const openGallery = async () => {
    // Request media library permission
    const permissionResult =
      await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (permissionResult.granted === false) {
      alert("You've refused to allow this app to access your photos!");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
      // Here you would typically upload the image to your backend
      console.log("Gallery image:", result.assets[0].uri);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />

      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.headerRight}
          onPress={() => setShowMenu(true)}
        >
          <Ionicons name="person-circle-outline" size={24} color="#4F46E5" />
          <Text style={styles.greeting}>Hello! Dean</Text>
        </TouchableOpacity>
      </View>

      {/* Profile Menu Modal */}
      <Modal
        visible={showMenu}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setShowMenu(false)}
      >
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setShowMenu(false)}
        >
          <View style={[styles.menuContainer, { top: 70, right: 20 }]}>
            <TouchableOpacity style={styles.menuItem} onPress={handleProfile}>
              <Ionicons name="person-outline" size={20} color="#1F2937" />
              <Text style={styles.menuText}>Profile</Text>
            </TouchableOpacity>

            <View style={styles.menuDivider} />

            <TouchableOpacity style={styles.menuItem} onPress={handleLogout}>
              <Ionicons name="log-out-outline" size={20} color="#EF4444" />
              <Text style={[styles.menuText, { color: "#EF4444" }]}>
                Logout
              </Text>
            </TouchableOpacity>
          </View>
        </TouchableOpacity>
      </Modal>

      {/* Main Content */}
      <View style={styles.content}>
        <View style={styles.uploadSection}>
          <Text style={styles.uploadText}>
            Capture a picture or upload from your device to discover about the
            person.
          </Text>
          <TouchableOpacity style={styles.uploadButton} onPress={openGallery}>
            {selectedImage ? (
              <Image
                source={{ uri: selectedImage }}
                style={styles.selectedImage}
                resizeMode="contain"
              />
            ) : (
              <Ionicons name="cloud-upload-outline" size={32} color="#6B7280" />
            )}
          </TouchableOpacity>

          {selectedImage && (
            <TouchableOpacity
              style={styles.findButton}
              onPress={() => router.push("/summary")}
            >
              <Text style={styles.findButtonText}>Find People</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>

      {/* Bottom Navigation */}
      <View style={styles.bottomNav}>
        <TouchableOpacity
          style={styles.navItem}
          onPress={() => router.push("/credits")}
        >
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
    backgroundColor: "#F8F9FA",
  },
  header: {
    padding: 16,
    flexDirection: "row",
    justifyContent: "flex-end",
    alignItems: "center",
  },
  headerRight: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    padding: 8,
  },
  greeting: {
    fontSize: 16,
    fontWeight: "500",
    color: "#4F46E5",
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.3)",
  },
  menuContainer: {
    position: "absolute",
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 8,
    minWidth: 180,
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.15,
    shadowRadius: 4,
    elevation: 5,
  },
  menuItem: {
    flexDirection: "row",
    alignItems: "center",
    padding: 12,
    gap: 12,
    borderRadius: 8,
  },
  menuText: {
    fontSize: 16,
    color: "#1F2937",
  },
  menuDivider: {
    height: 1,
    backgroundColor: "#E5E7EB",
    marginVertical: 4,
  },
  content: {
    flex: 1,
    padding: 16,
  },
  uploadSection: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    borderRadius: 16,
    padding: 16,
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
    width: "100%",
  },
  uploadText: {
    fontSize: 16,
    color: "#1F2937",
    textAlign: "center",
    marginBottom: 16,
  },
  uploadButton: {
    flex: 1,
    width: "100%",
    backgroundColor: "#F3F4F6",
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    borderColor: "#E5E7EB",
    borderStyle: "dashed",
    minHeight: 400,
  },
  bottomNav: {
    flexDirection: "row",
    justifyContent: "space-around",
    alignItems: "center",
    backgroundColor: "#FFFFFF",
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: -2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 4,
  },
  navItem: {
    alignItems: "center",
    gap: 4,
  },
  navText: {
    fontSize: 12,
    color: "#6B7280",
  },
  cameraButton: {
    width: 64,
    height: 64,
    backgroundColor: "#4F46E5",
    borderRadius: 32,
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#4F46E5",
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  selectedImage: {
    width: "100%",
    height: "100%",
    borderRadius: 12,
    resizeMode: "contain",
  },
  findButton: {
    backgroundColor: "#1E40AF",
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 100,
    marginTop: 24,
    width: "100%",
    alignItems: "center",
    shadowColor: "#1E40AF",
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  findButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },
});
