import React, { useState } from "react";
import {
  View,
  StyleSheet,
  Text,
  TouchableOpacity,
  SafeAreaView,
  Modal,
  Image,
  ActivityIndicator,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import axios from "axios";
import * as FileSystem from "expo-file-system";
import { useRouter } from "expo-router";

const API_URL = "https://x7rkc2ymrgyc2c-8000.proxy.runpod.net/detect-faces/";

export default function HomeScreen() {
  const router = useRouter();
  const [showMenu, setShowMenu] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [faceResult, setFaceResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Camera function
  const openCamera = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    if (permissionResult.granted === false) {
      alert("You've refused to allow this app to access your camera!");
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 0.8,
      exif: false,
      base64: false,
    });

    if (!result.canceled) {
      const asset = result.assets[0];
      setSelectedImage({
        uri: asset.uri,
        type: asset.mimeType || "image/jpeg",
      });
    }
  };

  const openGallery = async () => {
    const permissionResult =
      await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (permissionResult.granted === false) {
      alert("You've refused to allow this app to access your photos!");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      quality: 0.8,
      exif: false,
      base64: false,
    });

    if (!result.canceled) {
      const asset = result.assets[0];
      setSelectedImage({
        uri: asset.uri,
        type: asset.mimeType || "image/jpeg",
      });
    }
  };

  const uploadImage = async () => {
    if (!selectedImage) {
      alert("Please select an image first!");
      return;
    }

    try {
      setLoading(true);
      setFaceResult(null);

      // Get the local file URI
      const fileUri = selectedImage.uri;

      // Create FormData with proper file structure
      const formData = new FormData();
      formData.append("file", {
        uri: fileUri,
        name: `photo.${selectedImage.type.split("/")[1] || "jpg"}`,
        type: selectedImage.type || "image/jpeg",
      });

      const response = await axios.post(API_URL, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Accept: "application/json",
        },
        timeout: 60000,
      });

      if (response.status === 200) {
        router.push({
          pathname: "/result",
          params: { result: JSON.stringify(response.data) },
        });
      }
    } catch (error) {
      console.error("Full error:", error);
      alert(
        error.response?.data?.detail ||
          "Server error. Please try a different image."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.headerRight}
          onPress={() => setShowMenu(true)}
        >
          <Ionicons name="person-circle-outline" size={24} color="#4F46E5" />
          <Text style={styles.greeting}>Hello! Dean</Text>
        </TouchableOpacity>
      </View>

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
            <TouchableOpacity style={styles.menuItem} onPress={() => {}}>
              <Ionicons name="person-outline" size={20} color="#1F2937" />
              <Text style={styles.menuText}>Profile</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.menuItem} onPress={() => {}}>
              <Ionicons name="log-out-outline" size={20} color="#EF4444" />
              <Text style={[styles.menuText, { color: "#EF4444" }]}>
                Logout
              </Text>
            </TouchableOpacity>
          </View>
        </TouchableOpacity>
      </Modal>

      <View style={styles.content}>
        <View style={styles.uploadSection}>
          <Text style={styles.uploadText}>
            Capture a picture or upload from your device to discover about the
            person.
          </Text>

          <TouchableOpacity style={styles.uploadButton} onPress={openGallery}>
            {selectedImage ? (
              <Image
                source={{ uri: selectedImage.uri }}
                style={styles.selectedImage}
                resizeMode="contain"
              />
            ) : (
              <Ionicons name="cloud-upload-outline" size={32} color="#6B7280" />
            )}
          </TouchableOpacity>

          {selectedImage && (
            <TouchableOpacity style={styles.findButton} onPress={uploadImage}>
              <Text style={styles.findButtonText}>Find People</Text>
            </TouchableOpacity>
          )}

          {loading && (
            <ActivityIndicator
              size="large"
              color="#4F46E5"
              style={styles.loader}
            />
          )}

          {faceResult && (
            <View style={styles.resultContainer}>
              {faceResult.detail ? (
                <Text style={[styles.resultText, { color: "red" }]}>
                  {faceResult.detail}
                </Text>
              ) : (
                <Text style={styles.resultText}>
                  {(faceResult.face_result || "")
                    .split("\n")
                    .map((line, index) => (
                      <Text key={index}>
                        {line}
                        {"\n"}
                      </Text>
                    ))}
                </Text>
              )}
            </View>
          )}
        </View>
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
  greeting: {
    fontSize: 16,
    fontWeight: "500",
    color: "#4F46E5",
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
  selectedImage: {
    width: "100%",
    height: "100%",
    borderRadius: 12,
  },
  findButton: {
    backgroundColor: "#1E40AF",
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 100,
    marginTop: 24,
    width: "100%",
    alignItems: "center",
  },
  findButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },
  resultContainer: {
    marginTop: 20,
    padding: 16,
    backgroundColor: "#fff",
    borderRadius: 8,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  loader: {
    marginTop: 20,
  },
  resultText: {
    fontSize: 14,
    lineHeight: 20,
  },
});
