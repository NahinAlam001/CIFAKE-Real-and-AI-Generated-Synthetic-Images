import React from 'react';
import { TouchableOpacity, TouchableOpacityProps, ActivityIndicator } from 'react-native';
import styled from 'styled-components/native';
import { colors, borderRadius, typography } from '../../theme';
import { Text } from './Text';

interface ButtonProps extends TouchableOpacityProps {
  variant?: 'primary' | 'secondary' | 'outline';
  size?: 'small' | 'medium' | 'large';
  loading?: boolean;
  title: string;
}

const StyledButton = styled(TouchableOpacity)<ButtonProps>`
  border-radius: ${borderRadius.md}px;
  align-items: center;
  justify-content: center;
  padding: ${({ size = 'medium' }) =>
    size === 'small' ? '8px 16px' : size === 'medium' ? '12px 24px' : '16px 32px'};
  background-color: ${({ variant = 'primary', disabled }) =>
    disabled
      ? colors.gray[300]
      : variant === 'primary'
      ? colors.primary
      : variant === 'secondary'
      ? colors.secondary
      : 'transparent'};
  border-width: ${({ variant }) => (variant === 'outline' ? 1 : 0)}px;
  border-color: ${colors.primary};
  opacity: ${({ disabled }) => (disabled ? 0.5 : 1)};
`;

export const Button: React.FC<ButtonProps> = ({
  title,
  loading,
  variant = 'primary',
  size = 'medium',
  disabled,
  ...props
}) => {
  return (
    <StyledButton
      variant={variant}
      size={size}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <ActivityIndicator color={variant === 'outline' ? colors.primary : colors.text.white} />
      ) : (
        <Text
          variant="button"
          color={variant === 'outline' ? colors.primary : colors.text.white}
        >
          {title}
        </Text>
      )}
    </StyledButton>
  );
}; 