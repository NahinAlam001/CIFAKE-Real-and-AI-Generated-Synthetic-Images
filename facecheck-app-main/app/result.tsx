// app/result.tsx
import { useLocalSearchParams, useRouter } from "expo-router";
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
} from "react-native";

export default function ResultScreen() {
  const router = useRouter();
  const params = useLocalSearchParams();
  const result = JSON.parse(params.result as string);

  // Handle streaming response format
  const parseResult = (result: any) => {
    if (typeof result === "string") {
      try {
        const parsed = JSON.parse(result);
        return parsed.face_result || "No results found";
      } catch (e) {
        return result;
      }
    }
    return result.face_result || "No results found";
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
        <Text style={styles.backText}>← Back</Text>
      </TouchableOpacity>

      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.resultText}>
          {parseResult(result) || "No results found"}
        </Text>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    padding: 20,
  },
  backButton: {
    marginTop: 40,
    marginBottom: 20,
  },
  backText: {
    fontSize: 18,
    color: "#4F46E5",
  },
  content: {
    flexGrow: 1,
    paddingBottom: 40,
  },
  resultText: {
    fontSize: 16,
    lineHeight: 24,
    color: "#333",
  },
});
