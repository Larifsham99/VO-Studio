export const EMOTION_LABELS = {
  excited: 'Teruja', happy: 'Gembira', sad: 'Sedih', angry: 'Marah', curious: 'Ingin tahu',
  sarcastic: 'Sindir', crying: 'Menangis', fearful: 'Takut', calm: 'Tenang', confident: 'Yakin',
  serious: 'Serius', dramatic: 'Dramatik', whispers: 'Berbisik', softly: 'Lembut', shouts: 'Menjerit',
  laughs: 'Ketawa', sighs: 'Mengeluh',
}

export const GROUP_LABELS = { emotion: 'Emosi', volume: 'Suara', event: 'Bunyi' }

export const STRATEGIES = [
  ['auto', 'Auto (terbaik untuk model)'],
  ['instruction', 'Arahan gaya'],
  ['exaggeration', 'Intensiti emosi'],
  ['bank', 'Bank emosi suara'],
  ['none', 'Tanpa emosi'],
]

export const STATE_LABELS = {
  idle: 'Sedia', loading: 'Memuatkan model…', busy: 'Sedang menjana…', error: 'Ralat',
}

export const fmtTime = s => {
  if (s == null) return ''
  const m = Math.floor(s / 60), r = Math.round(s % 60)
  return m ? `${m}m ${r}s` : `${r}s`
}
