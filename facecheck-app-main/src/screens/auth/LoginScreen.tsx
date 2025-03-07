import React from "react";
import { View, StyleSheet, SafeAreaView, Image } from "react-native";
import { Text, Button } from "../../components/ui";
import { colors, spacing } from "../../theme";

export const LoginScreen = () => {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <View style={styles.header}>
          <Text variant="h1" style={styles.title}>
            Welcome to FaceCheck
          </Text>
          <Text
            variant="body1"
            color={colors.text.secondary}
            style={styles.subtitle}
          >
            Sign in to continue
          </Text>
        </View>

        <View style={styles.buttonContainer}>
          <Button
            title="Continue with Google"
            variant="outline"
            style={styles.button}
            onPress={() => console.log("Google login pressed")}
          />
          <Button
            title="Continue with Apple"
            variant="primary"
            style={styles.button}
            onPress={() => console.log("Apple login pressed")}
          />
        </View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    flex: 1,
    padding: spacing.lg,
    justifyContent: "space-between",
  },
  header: {
    marginTop: spacing.xxl * 2,
    alignItems: "center",
  },
  title: {
    marginBottom: spacing.sm,
    textAlign: "center",
  },
  subtitle: {
    textAlign: "center",
  },
  buttonContainer: {
    marginBottom: spacing.xxl,
  },
  button: {
    marginBottom: spacing.md,
  },
});
