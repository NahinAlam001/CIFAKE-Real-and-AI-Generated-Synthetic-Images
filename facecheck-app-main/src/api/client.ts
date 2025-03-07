import { getTokens } from '../services/auth';

// API configuration
const API_CONFIG = {
  baseUrl: 'https://api.facecheck.example.com', // Replace with your actual API URL
  timeout: 10000, // 10 seconds
  version: 'v1',
};

// HTTP methods
type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

// Request options
type RequestOptions = {
  method: HttpMethod;
  headers?: Record<string, string>;
  body?: any;
  requiresAuth?: boolean;
};

// API response
type ApiResponse<T> = {
  data?: T;
  error?: {
    code: string;
    message: string;
  };
  status: number;
};

/**
 * Make an API request
 */
export const apiRequest = async <T>(
  endpoint: string,
  options: RequestOptions = { method: 'GET', requiresAuth: true }
): Promise<ApiResponse<T>> => {
  try {
    const { method, headers = {}, body, requiresAuth = true } = options;
    
    // Build URL
    const url = `${API_CONFIG.baseUrl}/${API_CONFIG.version}${endpoint}`;
    
    // Set up headers
    const requestHeaders: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...headers,
    };
    
    // Add auth token if required
    if (requiresAuth) {
      const tokens = await getTokens();
      if (!tokens) {
        return {
          error: {
            code: 'auth/not-authenticated',
            message: 'User is not authenticated',
          },
          status: 401,
        };
      }
      
      requestHeaders['Authorization'] = `Bearer ${tokens.accessToken}`;
    }
    
    // Build request options
    const requestOptions: RequestInit = {
      method,
      headers: requestHeaders,
    };
    
    // Add body for non-GET requests
    if (method !== 'GET' && body) {
      requestOptions.body = JSON.stringify(body);
    }
    
    // Set timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.timeout);
    requestOptions.signal = controller.signal;
    
    // Make the request
    const response = await fetch(url, requestOptions);
    clearTimeout(timeoutId);
    
    // Parse response
    const responseData = await response.json();
    
    // Return formatted response
    return {
      data: response.ok ? responseData : undefined,
      error: !response.ok ? responseData.error : undefined,
      status: response.status,
    };
  } catch (error) {
    // Handle network errors, timeouts, etc.
    const isAbortError = error instanceof DOMException && error.name === 'AbortError';
    
    return {
      error: {
        code: isAbortError ? 'request/timeout' : 'request/network-error',
        message: isAbortError ? 'Request timed out' : 'Network error occurred',
      },
      status: 0,
    };
  }
};

// Convenience methods
export const get = <T>(endpoint: string, options?: Omit<RequestOptions, 'method'>) => 
  apiRequest<T>(endpoint, { ...options, method: 'GET' });

export const post = <T>(endpoint: string, body: any, options?: Omit<RequestOptions, 'method' | 'body'>) => 
  apiRequest<T>(endpoint, { ...options, method: 'POST', body });

export const put = <T>(endpoint: string, body: any, options?: Omit<RequestOptions, 'method' | 'body'>) => 
  apiRequest<T>(endpoint, { ...options, method: 'PUT', body });

export const patch = <T>(endpoint: string, body: any, options?: Omit<RequestOptions, 'method' | 'body'>) => 
  apiRequest<T>(endpoint, { ...options, method: 'PATCH', body });

export const del = <T>(endpoint: string, options?: Omit<RequestOptions, 'method'>) => 
  apiRequest<T>(endpoint, { ...options, method: 'DELETE' }); 