'use client';

import { useState, useCallback, useRef } from 'react';
import { Upload, X, FileImage, FileVideo, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import { api, UploadResponse, JobResponse } from '@/lib/api';

interface FileUploaderProps {
  onUploadComplete?: (result: JobResponse) => void;
  onAuthRequired?: () => void;
}

const ALLOWED_TYPES = [
  'image/jpeg',
  'image/png',
  'image/webp',
  'video/mp4',
  'video/quicktime',
  'video/x-msvideo',
  'video/webm',
];

const MAX_SIZE_MB = 100;

type UploadState = 'idle' | 'uploading' | 'processing' | 'success' | 'error';

export default function FileUploader({ onUploadComplete, onAuthRequired }: FileUploaderProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [state, setState] = useState<UploadState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [jobResult, setJobResult] = useState<JobResponse | null>(null);
  const [feedbackStatus, setFeedbackStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle');
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  
  const inputRef = useRef<HTMLInputElement>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const pollStartTimeRef = useRef<number | null>(null);

  const isVideo = selectedFile?.type.startsWith('video/');

  const validateFile = (file: File): string | null => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return 'Tipo de archivo no soportado. Usa JPEG, PNG, WebP, MP4, MOV, AVI o WebM.';
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `El archivo excede el límite de ${MAX_SIZE_MB}MB.`;
    }
    return null;
  };

  const handleFile = useCallback((file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      setState('error');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setState('idle');
    setJobResult(null);
    setFeedbackStatus('idle');
    setFeedbackError(null);

    // Generate preview
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(file);
    } else if (file.type.startsWith('video/')) {
      setPreview(URL.createObjectURL(file));
    }
  }, []);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const clearFile = useCallback(() => {
    setSelectedFile(null);
    setPreview(null);
    setState('idle');
    setError(null);
    setJobResult(null);
    setFeedbackStatus('idle');
    setFeedbackError(null);
    setUploadProgress(0);
    if (inputRef.current) inputRef.current.value = '';
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  const getVerdict = (result: JobResponse | null) => {
    const decision = result?.result?.ai_decision;
    if (decision === 'ai_generated') {
      return { label: 'Hecho con IA', tone: 'red' as const };
    }
    if (decision === 'not_ai_generated') {
      return { label: 'NO hecho con IA', tone: 'green' as const };
    }
    if (decision === 'inconclusive') {
      return { label: 'Inconcluso', tone: 'yellow' as const };
    }
    // Fallback si el backend aún no lo manda
    const p = result?.result?.probability;
    if (typeof p === 'number') {
      if (p >= 0.65) return { label: 'Hecho con IA', tone: 'red' as const };
      if (p <= 0.35) return { label: 'NO hecho con IA', tone: 'green' as const };
      return { label: 'Inconcluso', tone: 'yellow' as const };
    }
    return { label: 'Inconcluso', tone: 'yellow' as const };
  };

  const handleFeedback = useCallback(async (label: 'ai_generated' | 'not_ai_generated') => {
    const analysisId = jobResult?.result?.analysis_id;
    if (!analysisId) {
      setFeedbackStatus('error');
      setFeedbackError('No se pudo asociar el análisis para feedback (falta analysis_id).');
      return;
    }
    try {
      setFeedbackStatus('sending');
      setFeedbackError(null);
      await api.submitFeedback(analysisId, label);
      setFeedbackStatus('sent');
    } catch (e: any) {
      setFeedbackStatus('error');
      setFeedbackError(e?.message || 'Error enviando feedback');
    }
  }, [jobResult]);

  const pollJobStatus = useCallback(async (jobId: string) => {
    const MAX_POLL_TIME = 5 * 60 * 1000; // 5 minutos máximo para análisis
    pollStartTimeRef.current = Date.now();
    
    const poll = async () => {
      // Check timeout
      if (pollStartTimeRef.current && Date.now() - pollStartTimeRef.current > MAX_POLL_TIME) {
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
        setState('error');
        setError('El análisis tardó demasiado. Por favor, intenta de nuevo.');
        return;
      }
      
      try {
        const job = await api.getJob(jobId);
        // Celery returns: pending, started, success, failure, retry, revoked
        // Map to our states
        const status = job.status.toLowerCase();
        
        if (status === 'success' || status === 'done') {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
          
          setState('success');
          setJobResult(job);
          onUploadComplete?.(job);
        } else if (status === 'failure' || status === 'failed' || status === 'revoked') {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
          
          setState('error');
          setError('El análisis falló. Intenta de nuevo.');
        }
        // For pending, started, retry - keep polling
      } catch (err) {
        console.error('Error polling job:', err);
      }
    };

    // Poll every 3 seconds (más tiempo para evitar sobrecarga)
    pollIntervalRef.current = setInterval(poll, 3000);
    poll(); // Initial check
  }, [onUploadComplete]);

  const handleUpload = useCallback(async () => {
    if (!selectedFile) return;

    if (!api.isAuthenticated()) {
      onAuthRequired?.();
      return;
    }

    setState('uploading');
    setError(null);
    setUploadProgress(0);

    try {
      const uploadResult: UploadResponse = await api.uploadFile(selectedFile, setUploadProgress);
      setState('processing');
      pollJobStatus(uploadResult.job_id);
    } catch (err) {
      setState('error');
      setError(err instanceof Error ? err.message : 'Error al subir el archivo');
    }
  }, [selectedFile, onAuthRequired, pollJobStatus]);

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Drop Zone */}
      {!selectedFile && (
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={clsx(
            'relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300',
            'hover:border-primary-500 hover:bg-primary-50/50',
            dragActive && 'border-primary-500 bg-primary-50 scale-[1.02]',
            !dragActive && 'border-slate-300 bg-slate-50/50'
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ALLOWED_TYPES.join(',')}
            onChange={handleInputChange}
            className="hidden"
          />
          
          <div className="flex flex-col items-center gap-4">
            <div className={clsx(
              'w-16 h-16 rounded-full flex items-center justify-center transition-colors',
              dragActive ? 'bg-primary-100 text-primary-600' : 'bg-slate-200 text-slate-500'
            )}>
              <Upload className="w-8 h-8" />
            </div>
            
            <div>
              <p className="text-lg font-semibold text-slate-700">
                Arrastra tu archivo aquí
              </p>
              <p className="text-sm text-slate-500 mt-1">
                o haz clic para seleccionar
              </p>
            </div>
            
            <div className="flex flex-wrap justify-center gap-2 mt-2">
              <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full flex items-center gap-1">
                <FileImage className="w-3 h-3" /> Imágenes
              </span>
              <span className="px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full flex items-center gap-1">
                <FileVideo className="w-3 h-3" /> Videos
              </span>
            </div>
            
            <p className="text-xs text-slate-400 mt-2">
              JPEG, PNG, WebP, MP4, MOV, AVI, WebM • Máx {MAX_SIZE_MB}MB
            </p>
          </div>
        </div>
      )}

      {/* File Preview & Actions */}
      {selectedFile && (
        <div className="bg-white rounded-2xl shadow-lg border border-slate-200 overflow-hidden">
          {/* Preview */}
          <div className="relative aspect-video bg-slate-900 flex items-center justify-center">
            {preview && !isVideo && (
              <img 
                src={preview} 
                alt="Preview" 
                className="max-h-full max-w-full object-contain"
              />
            )}
            {preview && isVideo && (
              <video 
                src={preview} 
                controls 
                className="max-h-full max-w-full"
              />
            )}
            
            {/* Close button */}
            {state === 'idle' && (
              <button
                onClick={clearFile}
                className="absolute top-3 right-3 p-2 bg-black/50 hover:bg-black/70 rounded-full text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            )}
            
            {/* Processing overlay */}
            {(state === 'uploading' || state === 'processing') && (
              <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center">
                <Loader2 className="w-12 h-12 text-white animate-spin mb-4" />
                <p className="text-white font-medium">
                  {state === 'uploading' ? 'Subiendo...' : 'Analizando...'}
                </p>
                {state === 'uploading' && (
                  <p className="text-white/80 text-sm mt-1">
                    {uploadProgress.toFixed(0)}%
                  </p>
                )}
              </div>
            )}
          </div>

          {/* File info */}
          <div className="p-4">
            <div className="flex items-center gap-3 mb-4">
              {isVideo ? (
                <FileVideo className="w-8 h-8 text-purple-500 flex-shrink-0" />
              ) : (
                <FileImage className="w-8 h-8 text-blue-500 flex-shrink-0" />
              )}
              <div className="min-w-0 flex-1">
                <p className="font-medium text-slate-800 truncate">{selectedFile.name}</p>
                <p className="text-sm text-slate-500">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>

            {/* Progress bar */}
            {state === 'uploading' && (
              <div className="mb-4">
                <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-primary-500 to-primary-600 transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Error message */}
            {error && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}

            {/* Success result */}
            {state === 'success' && jobResult?.result && (
              (() => {
                const verdict = getVerdict(jobResult);
                return (
              <div className={clsx(
                'mb-4 p-4 rounded-lg border',
                verdict.tone === 'red'
                  ? 'bg-red-50 border-red-200'
                  : verdict.tone === 'yellow'
                    ? 'bg-yellow-50 border-yellow-200'
                    : 'bg-green-50 border-green-200'
              )}>
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle2 className="w-5 h-5 text-green-600" />
                  <span className="font-medium text-slate-800">Análisis completado</span>
                </div>
                <div className="mt-3">
                  <p className="text-sm text-slate-600">
                    <span className="font-medium">Veredicto:</span> {verdict.label}
                  </p>
                  <p className="text-sm text-slate-600 mb-1">Probabilidad de manipulación:</p>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-3 bg-slate-200 rounded-full overflow-hidden">
                      <div 
                        className={clsx(
                          'h-full transition-all duration-500',
                          verdict.tone === 'red'
                            ? 'bg-red-500'
                            : verdict.tone === 'yellow'
                              ? 'bg-yellow-500'
                              : 'bg-green-500'
                        )}
                        style={{ width: `${jobResult.result.probability * 100}%` }}
                      />
                    </div>
                    <span className={clsx(
                      'font-bold text-lg',
                      verdict.tone === 'red'
                        ? 'text-red-600'
                        : verdict.tone === 'yellow'
                          ? 'text-yellow-600'
                          : 'text-green-600'
                    )}>
                      {(jobResult.result.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-sm text-slate-600 mt-3">
                    <span className="font-medium">Sospecha:</span> {jobResult.result.suspected}
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    {jobResult.result.explanation}
                  </p>
                  <p className="text-xs text-slate-400 mt-2">
                    Modelo: {jobResult.result.model_version} • Ref: {jobResult.result.reference}
                  </p>

                  {/* Feedback */}
                  <div className="mt-4">
                    <p className="text-sm text-slate-700 font-medium mb-2">¿El veredicto fue correcto?</p>

                    {feedbackStatus === 'sent' ? (
                      <p className="text-sm text-slate-600">Gracias, feedback guardado.</p>
                    ) : (
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleFeedback('ai_generated')}
                          disabled={feedbackStatus === 'sending'}
                          className="flex-1 px-3 py-2 bg-slate-100 text-slate-800 rounded-lg font-medium hover:bg-slate-200 transition-colors disabled:opacity-60"
                        >
                          Es IA
                        </button>
                        <button
                          onClick={() => handleFeedback('not_ai_generated')}
                          disabled={feedbackStatus === 'sending'}
                          className="flex-1 px-3 py-2 bg-slate-100 text-slate-800 rounded-lg font-medium hover:bg-slate-200 transition-colors disabled:opacity-60"
                        >
                          No es IA
                        </button>
                      </div>
                    )}

                    {feedbackStatus === 'error' && feedbackError && (
                      <p className="text-xs text-red-600 mt-2">{feedbackError}</p>
                    )}
                    {jobResult.result.analysis_id && (
                      <p className="text-xs text-slate-400 mt-2">ID análisis: {jobResult.result.analysis_id}</p>
                    )}
                  </div>
                </div>
              </div>
                );
              })()
            )}

            {/* Actions */}
            <div className="flex gap-3">
              {state === 'idle' && (
                <>
                  <button
                    onClick={clearFile}
                    className="flex-1 px-4 py-3 border border-slate-300 text-slate-700 rounded-xl font-medium hover:bg-slate-50 transition-colors"
                  >
                    Cancelar
                  </button>
                  <button
                    onClick={handleUpload}
                    className="flex-1 px-4 py-3 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-xl font-medium hover:from-primary-700 hover:to-primary-600 transition-all shadow-lg shadow-primary-500/25"
                  >
                    Analizar
                  </button>
                </>
              )}
              
              {(state === 'success' || state === 'error') && (
                <button
                  onClick={clearFile}
                  className="w-full px-4 py-3 bg-slate-100 text-slate-700 rounded-xl font-medium hover:bg-slate-200 transition-colors"
                >
                  Analizar otro archivo
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
