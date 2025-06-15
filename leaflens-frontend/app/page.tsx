'use client';

import { useState, useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { UploadCloud, Loader2, Volume2 } from 'lucide-react';
import { speakText } from '../utils/tts'

type Top3 = { label: string; score: number };
type AnalyzeResult = {
  predicted: string;
  confidence: number;
  top3: Top3[];
  disease_type?: string;
  symptoms?: string;
  prevention?: string;
  treatments?: string;
  fertilizers?: string;
  expected_yield?: string;
  tts_message?: string;
  raw?: string;
};

export default function HomePage() {
  const { t, i18n } = useTranslation('common');
  const [plantName, setPlantName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const changeLanguage = (e: React.ChangeEvent<HTMLSelectElement>) =>
    i18n.changeLanguage(e.target.value);

  const openFileDialog = () => inputRef.current?.click();
  const onSelect = (f: File) => {
    setFile(f);
    setResult(null);
    setPlantName('');
  };

  const onAnalyze = async () => {
    if (!file) return alert(t('upload_prompt'));

    setBusy(true);
    const form = new FormData();
    form.append('file', file);
    form.append('lang', i18n.language);

    try {
      const res = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: form,
      });
      if (!res.ok) throw new Error(await res.text());
      const payload: AnalyzeResult = await res.json();

      setResult(payload);

      const inferred = payload.predicted.split('_')[0];
      setPlantName(inferred);

      // Use the translated TTS message from backend
      const message = payload.tts_message || t('tts_message', {
        plant: inferred,
        disease: payload.predicted.replace(/_/g, ' '),
        confidence: payload.confidence.toFixed(1),
      });

      // Auto-speak with appropriate language
      const langTag = i18n.language === 'hi' 
        ? 'hi-IN' 
        : i18n.language === 'ta' 
        ? 'ta-IN' 
        : 'en-US';
      
      speakText(message, langTag);
    } catch (err) {
      console.error(err);
      alert(t('error_inference'));
    } finally {
      setBusy(false);
    }
  };

  const handleSpeak = () => {
    if (!result) return;

    // Create comprehensive message with all translated content
    let message = result.tts_message || '';

    if (result.symptoms) {
      message += ` Symptoms: ${result.symptoms}.`;
    }
    if (result.prevention) {
      message += ` Prevention: ${result.prevention}.`;
    }
    if (result.treatments) {
      message += ` Treatments: ${result.treatments}.`;
    }
    if (result.fertilizers) {
      message += ` Fertilizers: ${result.fertilizers}.`;
    }
    if (result.expected_yield) {
      message += ` Expected yield: ${result.expected_yield}.`;
    }

    const langTag = i18n.language === 'hi'
      ? 'hi-IN'
      : i18n.language === 'ta'
      ? 'ta-IN'
      : 'en-US';

    speakText(message, langTag);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center py-8 px-4">
      <header className="w-full max-w-3xl flex justify-between items-center mb-6">
        <h1 className="text-5xl font-extrabold text-green-700">{t('title')}</h1>
        <select
          value={i18n.language}
          onChange={changeLanguage}
          className="border rounded-md p-2 bg-white"
        >
          {['en','hi','ta'].map(lng => (
            <option key={lng} value={lng}>{lng.toUpperCase()}</option>
          ))}
        </select>
      </header>

      <div className="w-full max-w-3xl bg-white rounded-2xl shadow-lg p-8 space-y-6">
        {plantName && (
          <div className="text-xl font-medium text-green-800">
            {t('detected_plant')}: {plantName}
          </div>
        )}

        <div
          onClick={openFileDialog}
          className="border-2 border-dashed border-green-400 rounded-xl p-12 flex flex-col items-center justify-center cursor-pointer hover:bg-green-50 transition"
        >
          <UploadCloud className="w-12 h-12 text-green-500 mb-4" />
          <p className="text-lg text-gray-600">
            {file ? file.name : t('upload_prompt')}
          </p>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={e => e.target.files?.[0] && onSelect(e.target.files[0])}
          />
        </div>

        <button
          onClick={onAnalyze}
          disabled={!file || busy}
          className={`w-full flex items-center justify-center py-3 rounded-xl text-white font-semibold transition ${
            !file || busy
              ? 'bg-gray-300 cursor-not-allowed text-gray-600'
              : 'bg-green-600 hover:bg-green-700'
          }`}
        >
          {busy
            ? <><Loader2 className="animate-spin mr-2" /> {t('detecting')}</>
            : t('analyze_button')}
        </button>

        {result && (
          <div className="bg-green-50 border border-green-200 rounded-xl p-6 space-y-4">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold text-green-800">
                {t('predicted')}: {result.predicted.replace(/_/g, ' ')}
              </h2>
              <button
                onClick={handleSpeak}
                className="flex items-center text-sm text-blue-600 hover:underline"
              >
                <Volume2 className="w-4 h-4 mr-1" />
                {t('speak_results')}
              </button>
            </div>
            <p className="text-sm text-green-700">
              {t('confidence')}: {result.confidence.toFixed(1)}%
            </p>
            <div>
              <h3 className="font-semibold text-green-800 mb-2">{t('top3')}:</h3>
              <ul className="list-disc list-inside text-green-700 space-y-1">
                {result.top3.map(p => (
                  <li key={p.label}>
                    {p.label.replace(/_/g, ' ')} — {p.score.toFixed(1)}%
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 space-y-2">
              {result.disease_type && (
                <div><strong>{t('disease_type')}:</strong> {result.disease_type}</div>
              )}
              {result.symptoms && (
                <div><strong>{t('symptoms')}:</strong> {result.symptoms}</div>
              )}
              {result.prevention && (
                <div><strong>{t('prevention')}:</strong> {result.prevention}</div>
              )}
              {result.treatments && (
                <div><strong>{t('treatments')}:</strong> {result.treatments}</div>
              )}
              {result.fertilizers && (
                <div><strong>{t('fertilizers')}:</strong> {result.fertilizers}</div>
              )}
              {result.expected_yield && (
                <div><strong>{t('expected_yield')}:</strong> {result.expected_yield}</div>
              )}
              {result.raw && (
                <pre className="text-sm text-gray-700">{result.raw}</pre>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
