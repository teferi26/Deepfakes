'use client';

import { useState, useEffect } from 'react';
import { Shield, Zap, Lock, Globe, CheckCircle, User, LogOut, ChevronRight } from 'lucide-react';
import FileUploader from '@/components/FileUploader';
import AuthModal from '@/components/AuthModal';
import { api } from '@/lib/api';

export default function Home() {
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [analysisCount, setAnalysisCount] = useState(0);

  useEffect(() => {
    // Load auth from storage on mount
    api.loadFromStorage();
    checkAuth();
  }, []);

  const checkAuth = async () => {
    if (api.isAuthenticated()) {
      try {
        const user = await api.getMe();
        setIsAuthenticated(true);
        setUserEmail(user.email);
      } catch {
        // Token expired or invalid
        api.clearAuth();
        setIsAuthenticated(false);
        setUserEmail(null);
      }
    }
  };

  const handleLogout = () => {
    api.clearAuth();
    setIsAuthenticated(false);
    setUserEmail(null);
  };

  const handleAuthSuccess = () => {
    checkAuth();
  };

  const handleUploadComplete = () => {
    setAnalysisCount((c) => c + 1);
  };

  const features = [
    {
      icon: Shield,
      title: 'Detección Avanzada',
      description: 'Algoritmos de última generación para identificar deepfakes, manipulaciones y contenido sintético.',
    },
    {
      icon: Zap,
      title: 'Análisis Rápido',
      description: 'Resultados en segundos gracias a nuestra infraestructura optimizada con procesamiento asíncrono.',
    },
    {
      icon: Lock,
      title: 'Privacidad Garantizada',
      description: 'Tus archivos se procesan de forma segura y se eliminan automáticamente tras el análisis.',
    },
    {
      icon: Globe,
      title: 'Multilenguaje',
      description: 'Interfaz disponible en español e inglés para usuarios de todo el mundo.',
    },
  ];

  return (
    <div className="min-h-screen hero-gradient">
      {/* Header */}
      <header className="sticky top-0 z-40 glass">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-600 to-primary-500 flex items-center justify-center shadow-lg shadow-primary-500/25">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-lg text-slate-800">Peritaciones.io</h1>
                <p className="text-xs text-slate-500 -mt-0.5">IA Forense</p>
              </div>
            </div>

            {/* Auth section */}
            <div className="flex items-center gap-4">
              {isAuthenticated ? (
                <div className="flex items-center gap-3">
                  <div className="hidden sm:block text-right">
                    <p className="text-sm font-medium text-slate-700">{userEmail}</p>
                    <p className="text-xs text-slate-500">{analysisCount} análisis realizados</p>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
                    title="Cerrar sesión"
                  >
                    <LogOut className="w-5 h-5" />
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setIsAuthModalOpen(true)}
                  className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-xl font-medium hover:from-primary-700 hover:to-primary-600 transition-all shadow-lg shadow-primary-500/25"
                >
                  <User className="w-4 h-4" />
                  <span>Acceder</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="pt-12 pb-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-primary-50 text-primary-700 rounded-full text-sm font-medium mb-6">
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
            MVP Activo
          </div>
          
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-slate-900 leading-tight mb-6">
            Detecta <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-600 to-primary-500">deepfakes</span> y manipulaciones con IA
          </h1>
          
          <p className="text-lg sm:text-xl text-slate-600 max-w-2xl mx-auto mb-8">
            Analiza imágenes y videos para identificar contenido sintético generado por inteligencia artificial. Resultados inmediatos y confidenciales.
          </p>

          {!isAuthenticated && (
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-8">
              <button
                onClick={() => setIsAuthModalOpen(true)}
                className="w-full sm:w-auto flex items-center justify-center gap-2 px-8 py-4 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-xl font-semibold text-lg hover:from-primary-700 hover:to-primary-600 transition-all shadow-xl shadow-primary-500/25"
              >
                Empezar gratis
                <ChevronRight className="w-5 h-5" />
              </button>
              <p className="text-sm text-slate-500">Sin tarjeta de crédito • 10 análisis gratuitos</p>
            </div>
          )}
        </div>
      </section>

      {/* Upload Section */}
      <section className="py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-2xl mx-auto">
          {isAuthenticated ? (
            <FileUploader
              onUploadComplete={handleUploadComplete}
              onAuthRequired={() => setIsAuthModalOpen(true)}
            />
          ) : (
            <div 
              onClick={() => setIsAuthModalOpen(true)}
              className="relative border-2 border-dashed border-slate-300 rounded-2xl p-12 text-center cursor-pointer hover:border-primary-500 hover:bg-primary-50/50 transition-all"
            >
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-200 flex items-center justify-center">
                <Lock className="w-8 h-8 text-slate-400" />
              </div>
              <p className="text-lg font-semibold text-slate-700 mb-2">
                Inicia sesión para analizar
              </p>
              <p className="text-sm text-slate-500">
                Crea una cuenta gratuita para subir y analizar tus archivos
              </p>
            </div>
          )}
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 mb-4">
              Tecnología de vanguardia
            </h2>
            <p className="text-lg text-slate-600 max-w-2xl mx-auto">
              Nuestra plataforma utiliza los modelos más avanzados de detección de contenido sintético
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <div key={index} className="card feature-card p-6">
                <div className="w-12 h-12 rounded-xl bg-primary-50 flex items-center justify-center mb-4">
                  <feature.icon className="w-6 h-6 text-primary-600" />
                </div>
                <h3 className="font-semibold text-lg text-slate-800 mb-2">{feature.title}</h3>
                <p className="text-sm text-slate-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white/50">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 mb-4">
              ¿Cómo funciona?
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              { step: '1', title: 'Sube tu archivo', desc: 'Arrastra o selecciona una imagen o video para analizar.' },
              { step: '2', title: 'Análisis automático', desc: 'Nuestra IA procesa el contenido en segundos.' },
              { step: '3', title: 'Obtén resultados', desc: 'Recibe un informe con la probabilidad de manipulación.' },
            ].map((item, index) => (
              <div key={index} className="text-center">
                <div className="w-14 h-14 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary-600 to-primary-500 flex items-center justify-center text-white text-xl font-bold shadow-lg shadow-primary-500/25">
                  {item.step}
                </div>
                <h3 className="font-semibold text-lg text-slate-800 mb-2">{item.title}</h3>
                <p className="text-sm text-slate-600">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Trust indicators */}
      <section className="py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="flex flex-wrap justify-center gap-8 text-slate-500">
            {[
              'Cifrado end-to-end',
              'GDPR Compliant',
              'Sin almacenamiento permanente',
              'API disponible',
            ].map((item, index) => (
              <div key={index} className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <span className="text-sm font-medium">{item}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 sm:px-6 lg:px-8 border-t border-slate-200">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-primary-600" />
              <span className="font-semibold text-slate-700">Peritaciones.io</span>
            </div>
            <p className="text-sm text-slate-500">
              © {new Date().getFullYear()} Peritaciones.io • MVP Forense
            </p>
            <div className="flex gap-6 text-sm text-slate-500">
              <a href="#" className="hover:text-primary-600 transition-colors">Privacidad</a>
              <a href="#" className="hover:text-primary-600 transition-colors">Términos</a>
              <a href="#" className="hover:text-primary-600 transition-colors">API</a>
            </div>
          </div>
        </div>
      </footer>

      {/* Auth Modal */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        onSuccess={handleAuthSuccess}
      />
    </div>
  );
}
