import React from 'react';
import { Text as RNText, TextProps as RNTextProps } from 'react-native';
import styled from 'styled-components/native';
import { colors, typography } from '../../theme';

interface TextProps extends RNTextProps {
  variant?: keyof typeof typography;
  color?: string;
}

const StyledText = styled(RNText)<TextProps>`
  ${({ variant = 'body1' }) => typography[variant]};
  color: ${({ color }) => color || colors.text.primary};
`;

export const Text: React.FC<TextProps> = ({ children, ...props }) => {
  return <StyledText {...props}>{children}</StyledText>;
}; 