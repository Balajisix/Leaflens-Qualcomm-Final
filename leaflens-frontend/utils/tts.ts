// utils/tts.ts

export function speakText(text: string, lang: string = 'en') {
  if (!window.speechSynthesis) {
    console.warn('TTS not supported in this browser');
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = lang;

  // Optional: Set voice based on language
  const voices = window.speechSynthesis.getVoices();
  const matched = voices.find(v => v.lang.startsWith(lang));
  if (matched) utterance.voice = matched;

  speechSynthesis.speak(utterance);
}
