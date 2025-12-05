/**
 * Get the API base URL based on the current environment
 * - In development (localhost): uses local backend server
 * - In production: uses relative path or production API
 */
export const getApiBase = () => {
  // Check if we're in a browser environment
  if (typeof window === 'undefined') {
    return '/OCR';
  }

  // In development (localhost), use local backend
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://localhost:5002/OCR';
  }

  // In production, use relative path
  // This will work with the vite proxy or direct backend connection
  return window.location.origin + '/OCR';
};

// Export the API base URL as a constant
export const API_BASE = getApiBase();

