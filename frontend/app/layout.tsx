import './globals.css';
import type { ReactNode } from 'react';

export const metadata = {
  title: 'Detector de Fraude IA',
  description: 'MVP de detección de deepfakes y contenido sintético'
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
