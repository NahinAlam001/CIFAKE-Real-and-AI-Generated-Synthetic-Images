import { TextStyle } from 'react-native';

export const typography = {
  h1: {
    fontSize: 34,
    lineHeight: 41,
    fontWeight: '700',
  } as TextStyle,
  h2: {
    fontSize: 28,
    lineHeight: 34,
    fontWeight: '700',
  } as TextStyle,
  h3: {
    fontSize: 22,
    lineHeight: 28,
    fontWeight: '600',
  } as TextStyle,
  body1: {
    fontSize: 17,
    lineHeight: 22,
    fontWeight: '400',
  } as TextStyle,
  body2: {
    fontSize: 15,
    lineHeight: 20,
    fontWeight: '400',
  } as TextStyle,
  button: {
    fontSize: 17,
    lineHeight: 22,
    fontWeight: '600',
  } as TextStyle,
  caption: {
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '400',
  } as TextStyle,
};

export type Typography = typeof typography; 