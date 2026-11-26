import { useCallback, useEffect, useRef, useState } from 'react'
import { api, audioUrl, getToken, refAudioUrl, setToken } from './api.js'
import { EMOTION_LABELS, GROUP_LABELS, STATE_LABELS, STRATEGIES, fmtTime } from './labels.js'

function usePoll(fn, ms, deps = []) {
  useEffect(() => {
    let alive = true
    const tick = async () => { if (alive) await fn() }
    tick()
    const t = setInterval(tick, ms)
    return () => { alive = false; clearInterval(t) }
  }, deps) // eslint-disable-line react-hooks/exhaustive-deps
}

function ErrorBox({ error, onClose }) {
  if (!error) return null
  return <div className="error" role="alert"><span>{error}</span>{onClose && <button className="link" onClick={onClose}>Tutup</button>}</div>
}

// ------------------------------------------------------------------ login
function Login({ onDone }) {
  const [t, setT] = useState('')
  return (
    <div className="center">
      <h1>Latif Voice</h1>
      <p className="muted">Buka pautan yang dipaparkan oleh server pada PC anda, atau tampal token di sini.</p>
      <input placeholder="Token" value={t} onChange={e => setT(e.target.value.trim())} />
      <button className="primary" disabled={!t} onClick={() => { setToken(t); onDone() }}>Sambung</button>
    </div>
  )
}

// --------------------------------------------------------------- projects
function NewProject({ status, refs, onCreated, onCancel }) {
  const installed = status.models.filter(m => m.installed)
  const [f, setF] = useState({
    name: '', mode: 'B', model: status.model || installed[0]?.id || '', language: 'ms',
    voice_id: status.voices[0]?.id || '', reference_id: '',
  })
  const [err, setErr] = useState('')
  const set = (k, v) => setF(o => ({ ...o, [k]: v }))
  const modeARefs = refs.filter(r => r.purpose === 'A')
  const save = async () => {
    try {
      const body = { name: f.name, mode: f.mode, model: f.model, language: f.language }
      if (f.mode === 'A') body.reference_id = f.reference_id
      else body.voice_id = f.voice_id
      onCreated(await api.createProject(body))
    } catch (e) { setErr(e.message) }
  }
  return (
    <div className="card">
      <h2>Projek baharu</h2>
      <label>Nama projek<input value={f.name} onChange={e => set('name', e.target.value)} placeholder="cth. Podcast Episod 1" /></label>
      <div className="seg">
        <button className={f.mode === 'B' ? 'on' : ''} onClick={() => set('mode', 'B')}>
          <b>Mod B</b><small>Suara tetap saya, gaya dari rakaman</small>
        </button>
        <button className={f.mode === 'A' ? 'on' : ''} onClick={() => set('mode', 'A')}>
          <b>Mod A</b><small>Tiru suara &amp; gaya rakaman rujukan</small>
        </button>
      </div>
      {f.mode === 'B' ? (
        status.voices.length ? (
          <label>Suara tetap
            <select value={f.voice_id} onChange={e => set('voice_id', e.target.value)}>
              {status.voices.map(v => <option key={v.id} value={v.id}>{v.id}</option>)}
            </select>
          </label>
        ) : <p className="warn">Belum ada suara tetap. Tambahkan dalam fail senarai suara di PC (lihat Status).</p>
      ) : modeARefs.length ? (
        <label>Rakaman rujukan
          <select value={f.reference_id} onChange={e => set('reference_id', e.target.value)}>
            <option value="">— pilih —</option>
            {modeARefs.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
          </select>
        </label>
      ) : <p className="warn">Muat naik rakaman Mod A dahulu di tab Rakaman.</p>}
      <div className="row2">
        <label>Model
          <select value={f.model} onChange={e => set('model', e.target.value)}>
            {installed.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        </label>
        <label>Bahasa
          <select value={f.language} onChange={e => set('language', e.target.value)}>
            <option value="ms">Bahasa Melayu</option><option value="en">English</option>
          </select>
        </label>
      </div>
      <ErrorBox error={err} />
      <div className="actions">
        <button onClick={onCancel}>Batal</button>
        <button className="primary" disabled={!f.name || !f.model || (f.mode === 'A' ? !f.reference_id : !f.voice_id)} onClick={save}>Cipta</button>
      </div>
    </div>
  )
}

function Projects({ status, refs, onOpen }) {
  const [list, setList] = useState(null)
  const [creating, setCreating] = useState(false)
  const [err, setErr] = useState('')
  const load = useCallback(() => api.projects().then(setList).catch(e => setErr(e.message)), [])
  useEffect(() => { load() }, [load])
  if (creating) return <NewProject status={status} refs={refs} onCancel={() => setCreating(false)}
    onCreated={p => { setCreating(false); onOpen(p) }} />
  return (
    <>
      <button className="primary block" onClick={() => setCreating(true)}>+ Projek baharu</button>
      <ErrorBox error={err} />
      {list?.length === 0 && <p className="muted center-text">Belum ada projek.</p>}
      {list?.map(p => (
        <button key={p.id} className="card item" onClick={() => onOpen(p)}>
          <span className={`badge m${p.mode}`}>Mod {p.mode}</span>
          <span className="grow"><b>{p.name}</b><small className="muted">{p.mode === 'B' ? `Suara: ${p.voice_id}` : 'Klon rujukan'} · {p.model} · {p.language}</small></span>
          <span className="chev">›</span>
        </button>
      ))}
    </>
  )
}

// --------------------------------------------------------------- generate
function TagPalette({ emotions, onInsert }) {
  const groups = {}
  emotions.forEach(e => { (groups[e.group] ||= []).push(e.id) })
  return (
    <div className="tags">
      {Object.entries(groups).map(([g, ids]) => (
        <div key={g} className="tag-group">
          <span className="muted small">{GROUP_LABELS[g] || g}</span>
          {ids.map(id => <button key={id} className="chip" onClick={() => onInsert(id)}>{EMOTION_LABELS[id] || id}</button>)}
        </div>
      ))}
    </div>
  )
}

function JobCard({ job, onDelete }) {
  return (
    <div className="card job">
      <p className="jobtext">{job.text}</p>
      {job.status === 'queued' && <p className="muted small">Dalam giliran…</p>}
      {job.status === 'running' && <div className="bar"><div style={{ width: `${Math.round(100 * job.progress)}%` }} /></div>}
      {job.status === 'error' && <ErrorBox error={job.error} />}
      {job.unknown_tags?.length > 0 && <p className="warn small">Tag tidak dikenali (diabaikan): {job.unknown_tags.join(', ')}</p>}
      {job.status === 'done' && (
        <>
          <audio controls preload="none" src={audioUrl(job.id)} />
          <div className="meta small muted">
            {fmtTime(job.duration_s)} · dijana dalam {fmtTime(job.gen_s)} · {job.model} · emosi: {job.emotion_strategy}
            {job.style_route ? ` · gaya: ${job.style_route}` : ''}
          </div>
          <div className="actions">
            <a className="btn" href={audioUrl(job.id, 'wav', true)} download>WAV</a>
            {job.has_mp3 && <a className="btn" href={audioUrl(job.id, 'mp3', true)} download>MP3</a>}
            <button className="danger" onClick={() => onDelete(job.id)}>Padam</button>
          </div>
        </>
      )}
    </div>
  )
}

function ProjectView({ project, status, refs, onBack }) {
  const [text, setText] = useState(() => { try { return localStorage.getItem(`draft_${project.id}`) || '' } catch { return '' } })
  const [opts, setOpts] = useState({ emotion_strategy: 'auto', normalize: true, style_reference_id: '', style_route: 'auto' })
  const [jobs, setJobs] = useState([])
  const [err, setErr] = useState('')
  const [showOpts, setShowOpts] = useState(false)
  const ta = useRef(null)
  const styleRefs = refs.filter(r => r.purpose === 'B')

  useEffect(() => { try { localStorage.setItem(`draft_${project.id}`, text) } catch {} }, [text, project.id])
  const active = jobs.some(j => j.status === 'queued' || j.status === 'running')
  usePoll(() => api.jobs(project.id).then(setJobs).catch(e => setErr(e.message)), active ? 1000 : 5000, [project.id, active])

  const insert = tag => {
    const el = ta.current
    const s = el?.selectionStart ?? text.length, e = el?.selectionEnd ?? text.length
    const ins = `[${tag}] `
    setText(text.slice(0, s) + ins + text.slice(e))
    requestAnimationFrame(() => { el?.focus(); el?.setSelectionRange(s + ins.length, s + ins.length) })
  }
  const go = async () => {
    setErr('')
    try {
      const body = { project_id: project.id, text, emotion_strategy: opts.emotion_strategy, normalize: opts.normalize }
      if (project.mode === 'B' && opts.style_reference_id) {
        body.style_reference_id = opts.style_reference_id
        body.style_route = opts.style_route
      }
      const j = await api.generate(body)
      setJobs(js => [j, ...js])
    } catch (e) { setErr(e.message) }
  }
  const del = async id => { await api.deleteJob(id).catch(e => setErr(e.message)); setJobs(js => js.filter(j => j.id !== id)) }
  const delProject = async () => {
    if (!confirm(`Padam projek "${project.name}" dan semua audionya?`)) return
    await api.deleteProject(project.id); onBack()
  }

  return (
    <>
      <div className="titlebar">
        <button className="link" onClick={onBack}>‹ Projek</button>
        <span className={`badge m${project.mode}`}>Mod {project.mode}</span>
      </div>
      <h2>{project.name}</h2>
      <div className="card">
        <textarea ref={ta} rows={6} value={text} onChange={e => setText(e.target.value)}
          placeholder="Taip teks di sini. Ketik butang emosi untuk menambah tag, cth. [sad] = Sedih…" />
        <div className="small muted right">{text.length} aksara</div>
        <TagPalette emotions={status.emotions} onInsert={insert} />
        <button className="link" onClick={() => setShowOpts(!showOpts)}>{showOpts ? '▾' : '▸'} Tetapan lanjut</button>
        {showOpts && (
          <div className="opts">
            <label>Cara emosi
              <select value={opts.emotion_strategy} onChange={e => setOpts({ ...opts, emotion_strategy: e.target.value })}>
                {STRATEGIES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </label>
            <label className="check"><input type="checkbox" checked={opts.normalize}
              onChange={e => setOpts({ ...opts, normalize: e.target.checked })} /> Baca nombor, RM, tarikh dengan betul (Melayu)</label>
            {project.mode === 'B' && (
              <>
                <label>Gaya dari rakaman (pilihan)
                  <select value={opts.style_reference_id} onChange={e => setOpts({ ...opts, style_reference_id: e.target.value })}>
                    <option value="">— tiada —</option>
                    {styleRefs.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
                  </select>
                </label>
                {opts.style_reference_id && (
                  <label>Cara pindah gaya
                    <select value={opts.style_route} onChange={e => setOpts({ ...opts, style_route: e.target.value })}>
                      <option value="auto">Auto</option>
                      <option value="native">Terus (model ikut rentak rakaman)</option>
                      <option value="instruction">Arahan (dari analisis rakaman)</option>
                    </select>
                  </label>
                )}
              </>
            )}
          </div>
        )}
        <ErrorBox error={err} onClose={() => setErr('')} />
        <button className="primary block" disabled={!text.trim()} onClick={go}>Jana suara</button>
      </div>
      {jobs.map(j => <JobCard key={j.id} job={j} onDelete={del} />)}
      <button className="danger block subtle" onClick={delProject}>Padam projek</button>
    </>
  )
}

// ------------------------------------------------------------- references
function References({ refs, reload }) {
  const [f, setF] = useState({ name: '', purpose: 'A', transcript: '', consent: '', confirm: false, language: 'ms' })
  const [file, setFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const set = (k, v) => setF(o => ({ ...o, [k]: v }))
  const upload = async () => {
    setBusy(true); setErr('')
    const fd = new FormData()
    fd.append('file', file)
    for (const k of ['name', 'purpose', 'transcript', 'consent', 'language']) fd.append(k, f[k])
    try {
      await api.uploadReference(fd)
      setF({ name: '', purpose: f.purpose, transcript: '', consent: '', confirm: false, language: f.language }); setFile(null)
      reload()
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  const del = async r => {
    if (!confirm(`Padam rakaman "${r.name}"?`)) return
    try { await api.deleteReference(r.id); reload() } catch (e) { setErr(e.message) }
  }
  return (
    <>
      <div className="card">
        <h2>Tambah rakaman</h2>
        <div className="seg">
          <button className={f.purpose === 'A' ? 'on' : ''} onClick={() => set('purpose', 'A')}><b>Untuk Mod A</b><small>Suara + gaya ditiru</small></button>
          <button className={f.purpose === 'B' ? 'on' : ''} onClick={() => set('purpose', 'B')}><b>Untuk Mod B</b><small>Gaya sahaja</small></button>
        </div>
        <label>Nama<input value={f.name} onChange={e => set('name', e.target.value)} placeholder="cth. Pencerita bersemangat" /></label>
        <label>Fail audio (10–20 saat terbaik)
          <input type="file" accept="audio/*" capture onChange={e => setFile(e.target.files[0] || null)} />
        </label>
        <label>Teks yang diucapkan dalam rakaman
          <textarea rows={3} value={f.transcript} onChange={e => set('transcript', e.target.value)}
            placeholder="Kosongkan jika server ada Whisper (transkripsi automatik)" />
        </label>
        <label>Kebenaran (siapa, bila)
          <input value={f.consent} onChange={e => set('consent', e.target.value)} placeholder="cth. Ali bin Abu, bertulis, 26/11/2026" />
        </label>
        <label className="check"><input type="checkbox" checked={f.confirm} onChange={e => set('confirm', e.target.checked)} />
          Saya mengesahkan pemilik suara ini telah membenarkan penggunaannya.</label>
        <ErrorBox error={err} onClose={() => setErr('')} />
        <button className="primary block" disabled={busy || !file || !f.name || !f.consent || !f.confirm} onClick={upload}>
          {busy ? 'Memuat naik & menganalisis…' : 'Muat naik'}
        </button>
      </div>
      {refs.map(r => (
        <div key={r.id} className="card">
          <div className="titlebar"><b>{r.name}</b><span className={`badge m${r.purpose}`}>Mod {r.purpose}</span></div>
          <audio controls preload="none" src={refAudioUrl(r.id)} />
          <p className="small">{r.transcript || <i className="warn">Tiada transkrip: klon mungkin kurang tepat</i>}
            {r.transcript_auto && <span className="muted"> (automatik)</span>}</p>
          {r.profile && (
            <div className="stats small">
              <span>Tempo <b>{r.profile.syllables_per_s?.toFixed(1) ?? '–'}</b> suku kata/s</span>
              <span>Variasi nada <b>{r.profile.f0_std_st?.toFixed(1) ?? '–'}</b> st</span>
              <span>Jeda <b>{r.profile.pauses}</b></span>
              <span>Tekanan <b>{r.profile.emphasis_points}</b></span>
            </div>
          )}
          {r.instruction && <p className="small muted">Gaya dikesan: {r.instruction}</p>}
          <p className="small muted">{fmtTime(r.duration_s)} · kebenaran: {r.consent}</p>
          <button className="danger subtle" onClick={() => del(r)}>Padam</button>
        </div>
      ))}
    </>
  )
}

// ----------------------------------------------------------------- status
function Status({ status, onLogout }) {
  const [err, setErr] = useState('')
  const gpu = status.system.gpus?.[0]
  return (
    <>
      <div className="card">
        <h2>PC anda</h2>
        <p>Keadaan: <b>{STATE_LABELS[status.state] || status.state}</b>{status.queue > 0 && ` · ${status.queue} dalam giliran`}</p>
        <p>Model aktif: <b>{status.model || 'belum dimuatkan'}</b></p>
        <p>Peranti: <b>{gpu ? `${gpu.name} (${gpu.vram_gb} GB)` : status.device}</b></p>
        <p className="small muted">RAM {status.system.ram_total_gb} GB · {status.system.platform}</p>
        {status.device === 'cpu' && <p className="warn">Tiada GPU dikesan. Penjanaan akan sangat perlahan.</p>}
        <ErrorBox error={status.last_error} />
      </div>
      <div className="card">
        <h2>Model</h2>
        {status.models.map(m => (
          <div key={m.id} className="titlebar">
            <span>{m.label}</span>
            {!m.installed ? <span className="muted small">tidak dipasang</span>
              : m.id === status.model ? <span className="badge mA">aktif</span>
              : <button onClick={() => api.loadModel(m.id).catch(e => setErr(e.message))}>Muatkan</button>}
          </div>
        ))}
        <ErrorBox error={err} />
      </div>
      <div className="card">
        <h2>Suara tetap</h2>
        {status.manifest_error && <p className="warn small">{status.manifest_error}</p>}
        {status.voices.map(v => <p key={v.id}><b>{v.id}</b> <span className="small muted">bank emosi: {v.has_emotion_bank.join(', ') || 'tiada'}</span></p>)}
        <p className="small muted">Suara tetap ditetapkan dalam benchmark/refs/manifest.yaml di PC.</p>
      </div>
      <button className="block subtle" onClick={onLogout}>Log keluar dari fon ini</button>
    </>
  )
}

// -------------------------------------------------------------------- app
export default function App() {
  const [authed, setAuthed] = useState(!!getToken())
  const [status, setStatus] = useState(null)
  const [refs, setRefs] = useState([])
  const [tab, setTab] = useState('projects')
  const [project, setProject] = useState(null)
  const [err, setErr] = useState('')

  const loadStatus = useCallback(async () => {
    try { setStatus(await api.status()); setErr('') } catch (e) {
      if (e.status === 401) { setToken(''); setAuthed(false) } else setErr(e.message)
    }
  }, [])
  const loadRefs = useCallback(() => api.references().then(setRefs).catch(() => {}), [])
  usePoll(() => authed && loadStatus(), 4000, [authed])
  useEffect(() => { if (authed) loadRefs() }, [authed, loadRefs])

  if (!authed) return <Login onDone={() => setAuthed(true)} />
  if (!status) return <div className="center"><p className="muted">Menyambung ke PC…</p><ErrorBox error={err} /></div>

  const nav = [['projects', 'Projek', '◧'], ['refs', 'Rakaman', '◉'], ['status', 'Status', '◎']]
  return (
    <div className="app">
      <header>
        <span className="logo">Latif Voice</span>
        <span className={`dot ${status.state}`} title={STATE_LABELS[status.state]} />
        <span className="small muted">{STATE_LABELS[status.state]}</span>
      </header>
      <main>
        <ErrorBox error={err} />
        {tab === 'projects' && (project
          ? <ProjectView project={project} status={status} refs={refs} onBack={() => setProject(null)} />
          : <Projects status={status} refs={refs} onOpen={setProject} />)}
        {tab === 'refs' && <References refs={refs} reload={loadRefs} />}
        {tab === 'status' && <Status status={status} onLogout={() => { setToken(''); setAuthed(false) }} />}
      </main>
      <nav>
        {nav.map(([id, label, icon]) => (
          <button key={id} className={tab === id ? 'on' : ''} onClick={() => { setTab(id); if (id === 'refs') loadRefs() }}>
            <span className="icon">{icon}</span>{label}
          </button>
        ))}
      </nav>
    </div>
  )
}
