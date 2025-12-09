const apiBase = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export default function Home() {
  return (
    <main className="page">
      <section className="hero">
        <p className="eyebrow">MVP • IA Forense</p>
        <h1>Detector de deepfakes e IA sintética</h1>
        <p className="lede">
          Sube una imagen o vídeo y obtén un informe técnico con probabilidad de manipulación por IA. Español / English.
        </p>
        <div className="cta">
          <span className="hint">API base:</span>
          <code>{apiBase}</code>
        </div>
      </section>

      <section className="card">
        <h2>Próximos pasos</h2>
        <ol>
          <li>Implementar subida de archivos y almacenamiento S3/MinIO.</li>
          <li>Conectar con el endpoint asíncrono y polling de jobs.</li>
          <li>Integrar detector open-source para imágenes y videos.</li>
        </ol>
      </section>
    </main>
  );
}
