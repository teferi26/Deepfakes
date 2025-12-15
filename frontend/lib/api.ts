const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export interface AuthResponse {
  id: string;
  email: string;
  role: string;
  api_key: string;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface UploadResponse {
  object_key: string;
  hash: string;
  size: number;
  media_type: 'image' | 'video';
  job_id: string;
  status: string;
}

export interface JobResult {
  probability: number;
  media_type: string;
  suspected: string;
  explanation: string;
  model_version: string;
  reference: string;
  ai_decision?: 'ai_generated' | 'not_ai_generated' | 'inconclusive';
  analysis_id?: string;
  decision_thresholds?: {
    threshold_low: number;
    threshold_high: number;
  };
}

export interface JobResponse {
  job_id: string;
  status: string;
  result: JobResult | null;
}

class ApiClient {
  private token: string | null = null;
  private apiKey: string | null = null;

  setToken(token: string) {
    this.token = token;
    if (typeof window !== 'undefined') {
      localStorage.setItem('auth_token', token);
    }
  }

  setApiKey(key: string) {
    this.apiKey = key;
    if (typeof window !== 'undefined') {
      localStorage.setItem('api_key', key);
    }
  }

  loadFromStorage() {
    if (typeof window !== 'undefined') {
      this.token = localStorage.getItem('auth_token');
      this.apiKey = localStorage.getItem('api_key');
    }
  }

  clearAuth() {
    this.token = null;
    this.apiKey = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth_token');
      localStorage.removeItem('api_key');
    }
  }

  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {};
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    } else if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }
    return headers;
  }

  async register(email: string, password: string): Promise<AuthResponse> {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Error en registro');
    }
    const data = await res.json();
    this.setApiKey(data.api_key);
    return data;
  }

  async login(email: string, password: string): Promise<TokenResponse> {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Error en login');
    }
    const data = await res.json();
    this.setToken(data.access_token);
    return data;
  }

  async getMe(): Promise<AuthResponse> {
    const res = await fetch(`${API_BASE}/auth/me`, {
      headers: this.getHeaders(),
    });
    if (!res.ok) throw new Error('No autenticado');
    return res.json();
  }

  async uploadFile(file: File, onProgress?: (pct: number) => void): Promise<UploadResponse> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${API_BASE}/v1/upload`);
      
      // Auth headers
      if (this.token) {
        xhr.setRequestHeader('Authorization', `Bearer ${this.token}`);
      } else if (this.apiKey) {
        xhr.setRequestHeader('X-API-Key', this.apiKey);
      }

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress((e.loaded / e.total) * 100);
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          try {
            const err = JSON.parse(xhr.responseText);
            reject(new Error(err.detail || 'Error al subir'));
          } catch {
            reject(new Error('Error al subir archivo'));
          }
        }
      };

      xhr.onerror = () => reject(new Error('Error de red'));

      const formData = new FormData();
      formData.append('file', file);
      xhr.send(formData);
    });
  }

  async getJob(jobId: string): Promise<JobResponse> {
    const res = await fetch(`${API_BASE}/v1/jobs/${jobId}`, {
      headers: this.getHeaders(),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Error al consultar job');
    }
    return res.json();
  }

  async submitFeedback(
    analysisId: string,
    label: 'ai_generated' | 'not_ai_generated',
    comment?: string
  ): Promise<any> {
    const res = await fetch(`${API_BASE}/v1/history/${analysisId}/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getHeaders(),
      },
      body: JSON.stringify({ label, comment }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Error enviando feedback');
    }

    return res.json();
  }

  isAuthenticated(): boolean {
    return !!(this.token || this.apiKey);
  }
}

export const api = new ApiClient();
